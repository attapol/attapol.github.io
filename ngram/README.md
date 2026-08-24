# N-gram Language Lab — GitHub Pages build

This folder is a self-contained static deployment of the classroom n-gram demo.

- `browser-model.js` performs tokenization, backoff, completion, and generation in the browser.
- The app uses the complete 526 MB compressed model from [`attapol/ngram-language-model-data`](https://github.com/attapol/ngram-language-model-data).
- All 34.4 million retained contexts are available. Each request downloads only one of 512 small gzip shards for the relevant order and context hash.
- Contexts use their exact corpus counts and probabilities; backoff occurs only when the complete trained model has no retained context.
- No server-side runtime or API is required.

The full training pipeline and uncompressed 2.2 GB local model are maintained in the source project rather than this Pages deployment.
