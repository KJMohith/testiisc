# Dhristi

Dhristi is a lightweight browser-based prototype for quick glaucoma-risk awareness.

## What changed

- Uses **one image only**.
- The user explicitly clicks **Run scan** to start analysis.
- The scan is faster because the image is resized before processing.
- The layout is responsive for phones and laptops.

## Features

- Upload an eye image or capture one from the device camera.
- Run a tiny explainable image-analysis heuristic directly in the browser.
- Return a color-coded status:
  - Red = Danger
  - Yellow = Consult
  - Green = Safe
- Show simple feature values like brightness, contrast, redness, and confidence.

## Important note

This project is **not a medical diagnosis tool**. It is a demo or hackathon-style concept showing how a tiny AI-inspired workflow can support awareness.

## Run locally

Use a small local server:

```bash
python3 -m http.server 4173
```

Then open <http://localhost:4173>.
