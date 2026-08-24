# N-gram Language Lab — GitHub Pages build

This folder is a self-contained static deployment of the classroom n-gram demo.

- `browser-model.js` performs tokenization, backoff, completion, and generation in the browser.
- `model/` contains a 32 MB compact binary export of frequent contexts from the full 99.2-million-token WikiText-103 model.
- Retained contexts use their exact corpus counts and probabilities; contexts omitted from the web export back off to a shorter retained context.
- No server-side runtime or API is required.

The full training pipeline and 2.2 GB local model are maintained in the source project rather than this Pages deployment.
