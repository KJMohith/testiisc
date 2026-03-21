# Dhristi static demo clone

Dhristi is a lightweight browser-based prototype for glaucoma-risk awareness. This static build keeps the same one-image flow: choose a single eye image, explicitly press **Run scan**, and review a red/yellow/green result with brightness, contrast, redness, and confidence metrics.

## What this prototype includes

- One image only, from upload or a single camera capture.
- An explicit **Run scan** button so inference starts only when the user chooses.
- A responsive dark glassmorphism layout that works on phone and laptop.
- An open model that runs directly in the browser.
- Red / yellow / green outputs with supporting quality metrics.
- Brightness, contrast, redness cue, and confidence readouts.

## Important warning

This is **not** a medical diagnosis tool. It is a screening-style demo for awareness only, and a qualified doctor must make any medical decision.

## Model runtime note

The first scan downloads model files for the open vision pipeline from a public CDN, so the initial run can take longer or fail if the network is unavailable.

## Local run

Serve the project from any static server, for example:

```bash
python3 -m http.server 4173
```

Then open `http://localhost:4173` in a browser.

## Smoke test

```bash
npm test
```

## Project files

- `index.html`
- `styles.css`
- `script.js`
- `README.md`
- `package.json`
- `tests/merge-readiness.mjs`
