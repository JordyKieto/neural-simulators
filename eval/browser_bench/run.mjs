// Run eval/browser_bench/index.html in headless Chromium and write results/eval/latency_browser.json.
//
//   cd eval/browser_bench && npm install && node run.mjs [--chromium /path/to/chrome]
//
// Serves the repository root with COOP/COEP headers (required for multithreaded WASM, as in the
// deployed demo), then runs the closed-loop benchmark for each execution-provider configuration.
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { extname, join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import os from 'node:os';
import { chromium } from 'playwright-core';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.onnx': 'application/octet-stream', '.json': 'application/json' };
const CONFIGS = [
  { ep: 'wasm', threads: 1 },
  { ep: 'wasm', threads: 4 },
  { ep: 'webgpu', threads: 1 },
];

const argIdx = process.argv.indexOf('--chromium');
const executablePath = argIdx > 0 ? process.argv[argIdx + 1] : undefined;

const server = createServer(async (req, res) => {
  try {
    const path = join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname));
    if (!path.startsWith(ROOT)) throw new Error('outside root');
    const body = await readFile(path);
    res.writeHead(200, {
      'Content-Type': TYPES[extname(path)] || 'application/octet-stream',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Resource-Policy': 'cross-origin',
    });
    res.end(body);
  } catch {
    res.writeHead(404).end();
  }
}).listen(0);
const port = server.address().port;

const browser = await chromium.launch({ executablePath, args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'] });
const results = [];
for (const cfg of CONFIGS) {
  const page = await browser.newPage();
  await page.goto(`http://localhost:${port}/eval/browser_bench/index.html?ep=${cfg.ep}&threads=${cfg.threads}&steps=300`);
  const r = await page.waitForFunction(() => window.__results, null, { timeout: 300_000 }).then((h) => h.jsonValue());
  console.log(JSON.stringify(r));
  results.push(r);
  await page.close();
}
await browser.close();
server.close();

const out = {
  machine: { cpu: os.cpus()[0]?.model, cores: os.cpus().length, platform: `${os.platform()} ${os.release()}` },
  browser: 'Chromium (headless, Playwright)',
  note: 'Closed-loop, batch 1. CARLA step = world_transition. TinySkies step = latent_sim + decoder. Excludes canvas drawing.',
  results,
};
await mkdir(join(ROOT, 'results/eval'), { recursive: true });
await writeFile(join(ROOT, 'results/eval/latency_browser.json'), JSON.stringify(out, null, 2));
console.log('wrote results/eval/latency_browser.json');
