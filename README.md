# Dhristi

Dhristi is a lightweight browser-based prototype for quick glaucoma-risk awareness.

## Features

- Upload an eye image or capture one from the device camera.
- Runs a tiny explainable image-analysis heuristic directly in the browser.
- Returns a color-coded status:
  - Red = Danger
  - Yellow = Consult
  - Green = Safe
- Shows simple feature values like brightness, contrast, redness, and confidence.

## Important note

This project is **not a medical diagnosis tool**. It is a demo or hackathon-style concept showing how a tiny AI-inspired workflow can support awareness.

## Run locally

Because browsers usually restrict camera access on plain local files, use a tiny local server:

```bash
python3 -m http.server 4173
```

Then open <http://localhost:4173>.
