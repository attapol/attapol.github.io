const state = { mode: 'completion', ready: false };
const model = new BrowserNGramModel('./model');
const context = document.querySelector('#context');
const order = document.querySelector('#order');
const runButton = document.querySelector('#runButton');
const chart = document.querySelector('#chart');
const loadingOverlay = document.querySelector('#loadingOverlay');
const loadingTitle = document.querySelector('#loadingTitle');
const loadingDetail = document.querySelector('#loadingDetail');
let loadingStartedAt = performance.now();
let loadingHideTimer;

document.querySelectorAll('.mode').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('.mode').forEach(item => item.classList.toggle('active', item === button));
  state.mode = button.dataset.mode;
  document.querySelector('#completionPanel').classList.toggle('hidden', state.mode !== 'completion');
  document.querySelector('#generationPanel').classList.toggle('hidden', state.mode !== 'generation');
  setButtonLabel();
}));

order.addEventListener('change', () => {
  const memory = Number(order.value) - 1;
  document.querySelector('#hint').textContent = `The model uses the last ${memory} tokens and backs off when that context is rare.`;
});

document.querySelector('#temperature').addEventListener('input', event => {
  document.querySelector('#tempValue').value = event.target.value;
});

runButton.addEventListener('click', run);
context.addEventListener('keydown', event => {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') run();
});

async function initialize() {
  runButton.disabled = true;
  try {
    await model.init(showProgress);
    state.ready = true;
    const meta = model.metadata;
    document.querySelector('#corpusStats').textContent = `${(meta.encoded_tokens / 1e6).toFixed(1)}M TOKENS · ${meta.vocabulary_size.toLocaleString()} WORDS`;
    await complete();
  } catch (error) {
    document.querySelector('#corpusStats').textContent = 'MODEL UNAVAILABLE';
    chart.innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`;
  } finally {
    runButton.disabled = !state.ready;
    if (state.ready) showCorpusStats();
    setLoading(false);
    setButtonLabel();
  }
}

async function run() {
  if (!state.ready) return;
  runButton.disabled = true;
  showProgress('Working…');
  try {
    if (state.mode === 'completion') await complete(); else await generate();
  } catch (error) {
    const target = state.mode === 'completion' ? chart : document.querySelector('#generatedText');
    target.textContent = `Error: ${error.message}`;
  } finally {
    runButton.disabled = false;
    showCorpusStats();
    setLoading(false);
    setButtonLabel();
  }
}

async function complete() {
  const data = await model.complete(context.value, Number(order.value), showProgress);
  const maximum = Math.max(...data.predictions.map(item => item.probability), 0.00001);
  chart.innerHTML = data.predictions.map((item, index) => `
    <div class="bar-row">
      <div class="word" title="${escapeHtml(item.token)}">${escapeHtml(displayToken(item.token))}</div>
      <div class="track"><div class="fill" style="width:${item.probability / maximum * 100}%;animation-delay:${index * 18}ms"></div></div>
      <div class="percent">${formatProbability(item.probability)}</div>
    </div>`).join('');
  const matched = data.matched_context.map(displayToken).join(' ');
  document.querySelector('#matchInfo').innerHTML = `${data.source_order}-gram used · ${data.context_count.toLocaleString()} examples<br>context: “${escapeHtml(matched || 'unigram fallback')}” · top-20 mass ${(data.shown_probability_mass * 100).toFixed(1)}%`;
}

async function generate() {
  const data = await model.generate(
    context.value, Number(order.value), 100,
    Number(document.querySelector('#temperature').value), showProgress,
  );
  document.querySelector('#generatedText').textContent = data.text;
  const backoffs = data.trace.filter(step => step.source_order < Number(order.value)).length;
  document.querySelector('#generationMeta').textContent = `${data.tokens.length} tokens generated · ${backoffs} backoff steps · stopped: ${data.stopped.replace('_', ' ')}`;
}

function showProgress(message) {
  document.querySelector('#corpusStats').textContent = message.toUpperCase();
  loadingTitle.textContent = message.includes('vocabulary')
    ? 'Learning the vocabulary'
    : message.includes('-gram')
      ? 'Extending model memory'
      : 'Computing the next words';
  loadingDetail.textContent = message;
  setLoading(true);
}
function setLoading(active) {
  clearTimeout(loadingHideTimer);
  if (active) {
    if (!loadingOverlay.classList.contains('active')) loadingStartedAt = performance.now();
    loadingOverlay.classList.add('active');
    loadingOverlay.setAttribute('aria-hidden', 'false');
    return;
  }
  const delay = Math.max(0, 550 - (performance.now() - loadingStartedAt));
  loadingHideTimer = setTimeout(() => {
    loadingOverlay.classList.remove('active');
    loadingOverlay.setAttribute('aria-hidden', 'true');
  }, delay);
}
function showCorpusStats() {
  document.querySelector('#corpusStats').textContent = `${(model.metadata.encoded_tokens / 1e6).toFixed(1)}M TOKENS · ${model.metadata.vocabulary_size.toLocaleString()} WORDS`;
}
function setButtonLabel() { runButton.firstChild.textContent = state.mode === 'completion' ? 'Predict ' : 'Generate '; }
function displayToken(token) { return token === '</s>' ? '〈END〉' : token === '<unk>' ? '〈UNK〉' : token; }
function formatProbability(value) { return value >= .001 ? `${(value * 100).toFixed(2)}%` : `${(value * 100).toPrecision(2)}%`; }
function escapeHtml(value) { const element = document.createElement('span'); element.textContent = value; return element.innerHTML; }

initialize();
