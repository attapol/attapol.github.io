class BrowserNGramModel {
  constructor(baseUrl = './model') {
    this.baseUrl = baseUrl;
    this.metadata = null;
    this.tokenToId = new Map();
    this.idToToken = [];
    this.unigramCounts = [];
    this.unigramTop = [];
    this.orders = new Map();
    this.loading = new Map();
  }

  async init(onProgress = () => {}) {
    onProgress('Loading vocabulary…');
    const [metadataResponse, vocabularyResponse] = await Promise.all([
      fetch(`${this.baseUrl}/metadata.json`),
      fetch(`${this.baseUrl}/vocab.tsv`),
    ]);
    if (!metadataResponse.ok || !vocabularyResponse.ok) throw new Error('Could not load the static model');
    this.metadata = await metadataResponse.json();
    const vocabulary = await vocabularyResponse.text();
    for (const line of vocabulary.split('\n')) {
      if (!line) continue;
      const firstTab = line.indexOf('\t');
      const secondTab = line.indexOf('\t', firstTab + 1);
      const id = Number(line.slice(0, firstTab));
      const count = Number(line.slice(firstTab + 1, secondTab));
      const token = line.slice(secondTab + 1);
      this.idToToken[id] = token;
      this.unigramCounts[id] = count;
      this.tokenToId.set(token, id);
    }
    const total = this.unigramCounts.reduce((sum, count = 0) => sum + count, 0) || 1;
    this.unigramTotal = total;
    this.unigramTop = this.unigramCounts
      .map((count = 0, id) => ({ id, count }))
      .filter(item => item.id !== 1 && item.count > 0)
      .sort((left, right) => right.count - left.count)
      .slice(0, 20);
    onProgress('Loading core n-grams…');
    await this.ensureOrder(3, onProgress);
  }

  async ensureOrder(maxOrder, onProgress = () => {}) {
    for (let order = 2; order <= maxOrder; order += 1) {
      if (this.orders.has(order)) continue;
      if (!this.loading.has(order)) this.loading.set(order, this.loadOrder(order, onProgress));
      await this.loading.get(order);
    }
  }

  async loadOrder(order, onProgress) {
    onProgress(`Loading ${order}-gram contexts…`);
    const response = await fetch(`${this.baseUrl}/order_${order}.bin`);
    if (!response.ok) throw new Error(`Could not load order ${order}`);
    const data = new DataView(await response.arrayBuffer());
    const storedOrder = data.getUint32(8, true);
    const recordCount = data.getUint32(12, true);
    if (storedOrder !== order) throw new Error(`Invalid order ${order} model`);
    const contexts = new Map();
    let offset = 16;
    for (let record = 0; record < recordCount; record += 1) {
      const context = [];
      for (let index = 0; index < order - 1; index += 1) {
        context.push(data.getUint32(offset, true));
        offset += 4;
      }
      const total = data.getUint32(offset, true);
      const successorCount = data.getUint32(offset + 4, true);
      offset += 8;
      const successors = [];
      for (let index = 0; index < successorCount; index += 1) {
        successors.push([data.getUint32(offset, true), data.getUint32(offset + 4, true)]);
        offset += 8;
      }
      contexts.set(context.join(','), { total, successors });
    }
    this.orders.set(order, contexts);
    this.loading.delete(order);
  }

  tokenize(text) {
    return text.trim().match(/[\p{L}\p{N}_]+(?:['’\-][\p{L}\p{N}_]+)*|[^\s\p{L}\p{N}_]/gu) || [];
  }

  tokenId(token) {
    return this.tokenToId.get(token) ?? this.tokenToId.get(token.toLowerCase()) ?? 0;
  }

  async complete(text, maxOrder, onProgress = () => {}) {
    maxOrder = Math.max(2, Math.min(Number(maxOrder), Number(this.metadata.max_order)));
    await this.ensureOrder(maxOrder, onProgress);
    const tokens = this.tokenize(text);
    const ids = [...Array(maxOrder - 1).fill(1), ...tokens.map(token => this.tokenId(token))];
    for (let order = maxOrder; order >= 2; order -= 1) {
      const context = ids.slice(-(order - 1));
      const result = this.orders.get(order).get(context.join(','));
      if (result) return this.result(tokens, order, context, result.total, result.successors);
    }
    return this.result(tokens, 1, [], this.unigramTotal, this.unigramTop.map(item => [item.id, item.count]));
  }

  result(inputTokens, sourceOrder, context, total, successors) {
    const predictions = successors.map(([id, count]) => ({
      token: this.idToToken[id] || '<unk>', probability: count / total, count,
    }));
    return {
      input_tokens: inputTokens,
      source_order: sourceOrder,
      matched_context: context.map(id => this.idToToken[id] || '<unk>'),
      context_count: total,
      shown_probability_mass: predictions.reduce((sum, item) => sum + item.probability, 0),
      predictions,
    };
  }

  async generate(text, maxOrder, maxTokens = 100, temperature = 1, onProgress = () => {}) {
    await this.ensureOrder(maxOrder, onProgress);
    const seedTokens = this.tokenize(text);
    const generated = [];
    const trace = [];
    let stopped = 'max_tokens';
    for (let step = 0; step < Math.min(Math.max(maxTokens, 1), 250); step += 1) {
      const completion = await this.complete([...seedTokens, ...generated].join(' '), maxOrder);
      if (!completion.predictions.length) { stopped = 'no_prediction'; break; }
      const weights = completion.predictions.map(item => Math.max(item.probability, 1e-12) ** (1 / Math.max(temperature, .05)));
      const choice = completion.predictions[this.weightedIndex(weights)].token;
      trace.push({ token: choice, source_order: completion.source_order });
      if (choice === '</s>') { stopped = 'end_token'; break; }
      generated.push(choice);
    }
    return { seed: text, tokens: generated, text: this.detokenize([...seedTokens, ...generated]), stopped, trace };
  }

  weightedIndex(weights) {
    let value = Math.random() * weights.reduce((sum, weight) => sum + weight, 0);
    for (let index = 0; index < weights.length; index += 1) {
      value -= weights[index];
      if (value <= 0) return index;
    }
    return weights.length - 1;
  }

  detokenize(tokens) {
    return tokens.join(' ')
      .replaceAll(' @-@ ', '-').replaceAll(' @.@ ', '. ').replaceAll(' @,@ ', ', ')
      .replace(/\s+([,.;:!?%\)])/g, '$1').replace(/([\(\[])\s+/g, '$1')
      .replaceAll(" n't", "n't").replaceAll(" 's", "'s").replaceAll(" 're", "'re");
  }
}

