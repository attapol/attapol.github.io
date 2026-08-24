class BrowserNGramModel {
  constructor(baseUrl = './model') {
    this.baseUrl = baseUrl;
    this.metadata = null;
    this.tokenToId = new Map();
    this.idToToken = [];
    this.unigramCounts = [];
    this.unigramTop = [];
    this.shards = new Map();
    this.loadingShards = new Map();
    this.maxCachedShards = 96;
  }

  async init(onProgress = () => {}) {
    onProgress('Loading full vocabulary…');
    const metadataResponse = await fetch(`${this.baseUrl}/metadata.json`);
    if (!metadataResponse.ok) throw new Error('Could not load model metadata');
    this.metadata = await metadataResponse.json();

    const vocabularyResponse = await fetch(`${this.metadata.data_base}/vocab.tsv.gz`);
    if (!vocabularyResponse.ok) throw new Error('Could not load the full vocabulary');
    const vocabulary = await this.decompressText(vocabularyResponse);
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

    this.unigramTotal = this.unigramCounts.reduce((sum, count = 0) => sum + count, 0) || 1;
    this.unigramTop = this.unigramCounts
      .map((count = 0, id) => ({ id, count }))
      .filter(item => item.id !== 1 && item.count > 0)
      .sort((left, right) => right.count - left.count)
      .slice(0, 20);
  }

  async decompressText(response) {
    const buffer = await this.decompressBuffer(response);
    return new TextDecoder().decode(buffer);
  }

  async decompressBuffer(response) {
    if (!globalThis.DecompressionStream) {
      throw new Error('This browser does not support streaming gzip decompression');
    }
    const stream = response.body.pipeThrough(new DecompressionStream('gzip'));
    return new Response(stream).arrayBuffer();
  }

  tokenize(text) {
    return text.trim().match(/[\p{L}\p{N}_]+(?:['’\-][\p{L}\p{N}_]+)*|[^\s\p{L}\p{N}_]/gu) || [];
  }

  tokenId(token) {
    return this.tokenToId.get(token) ?? this.tokenToId.get(token.toLowerCase()) ?? 0;
  }

  contextHash(context) {
    let value = 1469598103934665603n;
    for (const tokenId of context) {
      value ^= BigInt(tokenId);
      value = BigInt.asUintN(64, value * 1099511628211n);
    }
    return Number(value % BigInt(this.metadata.bucket_count));
  }

  async complete(text, maxOrder, onProgress = () => {}) {
    maxOrder = Math.max(2, Math.min(Number(maxOrder), Number(this.metadata.max_order)));
    const tokens = this.tokenize(text);
    const ids = [...Array(maxOrder - 1).fill(1), ...tokens.map(token => this.tokenId(token))];

    for (let order = maxOrder; order >= 2; order -= 1) {
      const context = ids.slice(-(order - 1));
      const result = await this.lookup(order, context, onProgress);
      if (result) return this.result(tokens, order, context, result.total, result.successors);
    }
    return this.result(tokens, 1, [], this.unigramTotal, this.unigramTop.map(item => [item.id, item.count]));
  }

  async lookup(order, context, onProgress) {
    const bucket = this.contextHash(context);
    const contexts = await this.loadShard(order, bucket, onProgress);
    return contexts.get(context.join(','));
  }

  async loadShard(order, bucket, onProgress) {
    const key = `${order}:${bucket}`;
    if (this.shards.has(key)) {
      const cached = this.shards.get(key);
      this.shards.delete(key);
      this.shards.set(key, cached);
      return cached;
    }
    if (this.loadingShards.has(key)) return this.loadingShards.get(key);

    const loading = this.fetchShard(order, bucket, onProgress);
    this.loadingShards.set(key, loading);
    try {
      const contexts = await loading;
      this.shards.set(key, contexts);
      while (this.shards.size > this.maxCachedShards) {
        this.shards.delete(this.shards.keys().next().value);
      }
      return contexts;
    } finally {
      this.loadingShards.delete(key);
    }
  }

  async fetchShard(order, bucket, onProgress) {
    onProgress(`Loading exact ${order}-gram shard ${bucket + 1}/512…`);
    const part = String(bucket).padStart(3, '0');
    const response = await fetch(`${this.metadata.data_base}/order_${order}/part_${part}.bin.gz`);
    if (!response.ok) throw new Error(`Could not load ${order}-gram shard ${bucket}`);
    const data = new DataView(await this.decompressBuffer(response));
    const contexts = new Map();
    let offset = 0;
    while (offset < data.byteLength) {
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
    return contexts;
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
    const seedTokens = this.tokenize(text);
    const generated = [];
    const trace = [];
    let stopped = 'max_tokens';
    for (let step = 0; step < Math.min(Math.max(maxTokens, 1), 250); step += 1) {
      const completion = await this.complete(
        [...seedTokens, ...generated].join(' '), maxOrder, onProgress,
      );
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

