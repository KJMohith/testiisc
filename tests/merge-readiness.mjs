import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';

const [html, script, readme] = await Promise.all([
  readFile(new URL('../index.html', import.meta.url), 'utf8'),
  readFile(new URL('../script.js', import.meta.url), 'utf8'),
  readFile(new URL('../README.md', import.meta.url), 'utf8'),
]);

assert.match(html, /<script\s+type="module"\s+src="script\.js"><\/script>/i, 'index.html must load script.js as a module');
assert.match(script, /@xenova\/transformers@2\.17\.2/, 'script.js must import @xenova/transformers from the CDN');
assert.match(script, /pipeline\(['"]zero-shot-image-classification['"],\s*MODEL_ID\)/, 'script.js must initialize the zero-shot classifier');
assert.match(readme, /public CDN/i, 'README must document the public CDN download behavior');

console.log('merge readiness checks passed');
