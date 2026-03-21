import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const [indexHtml, scriptJs, readme] = await Promise.all([
  readFile(new URL('../index.html', import.meta.url), 'utf8'),
  readFile(new URL('../script.js', import.meta.url), 'utf8'),
  readFile(new URL('../README.md', import.meta.url), 'utf8'),
]);

assert.match(indexHtml, /<script type="module" src="script\.js"><\/script>/, 'index.html must load script.js as a module');
assert.match(scriptJs, /@xenova\/transformers@2\.17\.2/, 'script.js must import the open model runtime');
assert.match(scriptJs, /pipeline\('zero-shot-image-classification', MODEL_ID\)/, 'script.js must initialize the zero-shot image classifier');
assert.match(scriptJs, /MODEL_LABELS = \[/, 'script.js must define model labels for classification');
assert.match(readme, /public CDN/i, 'README must document the initial model download requirement');

console.log('Merge-readiness checks passed.');
