// Capture frames from the deployed browser demo while a scripted "player" holds keys.
//
//   node capture.mjs --mode driving|flying [--url URL] [--seconds 20] [--seed-key n] [--chromium PATH]
//
// Writes results/eval/captures/<mode>_<tag>/page.webm (Playwright page video), canvas.json (the
// canvas bounding box, for cropping) and controls.jsonl (keys held + HUD text, sampled at 10 Hz).
// The page runs inference locally (onnxruntime-web, WASM), exactly as for any visitor.
import { mkdir, writeFile } from 'node:fs/promises';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const arg = (name, fallback) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : fallback;
};
const MODE = arg('mode', 'driving');
const URL = arg('url', 'https://drift-sim-production.up.railway.app/');
const SECONDS = Number(arg('seconds', 20));
const SEEDS = Number(arg('seeds', 0)); // press N this many times before recording
const TAG = arg('tag', '0');

// Human-like scripts: [startSec, endSec, key]. Keys overlap to mix actions.
const taps = (key, start, end, on, period) =>
  Array.from({ length: Math.floor((end - start) / period) }, (_, i) => [start + i * period, start + i * period + on, key]);
const SCRIPTS = {
  driving_gentle: [
    ...taps('w', 0, 20, 0.35, 1.0),
    [6.0, 6.6, 'a'], [11.0, 11.5, 'd'], [15.0, 15.4, 'a'],
  ],
  driving: [
    [0.0, 2.5, 'w'], [3.5, 5.0, 'w'], [6.0, 7.5, 'w'], [8.0, 9.0, 'a'], [9.5, 11.0, 'w'],
    [11.5, 12.2, 'd'], [12.5, 14.0, 'w'], [14.5, 15.5, 's'], [16.0, 18.0, 'w'], [18.5, 19.2, 'a'],
  ],
  flying: [
    [0.0, 2.0, 'w'], [2.0, 3.5, 'a'], [3.5, 5.0, 'w'], [5.0, 6.5, 'd'], [6.5, 8.0, 'w'],
    [8.5, 10.5, 'a'], [11.0, 12.0, 's'], [12.0, 14.0, 'w'], [14.0, 16.0, 'd'], [16.5, 19.5, 'w'],
  ],
};

const out = join(ROOT, 'results/eval/captures', `${MODE}_${TAG}`);
const PAGE_MODE = MODE.startsWith('driving') ? 'driving' : 'flying';
await mkdir(out, { recursive: true });
const browser = await chromium.launch({ executablePath: arg('chromium', undefined) });
const VIEWPORT = { width: 1280, height: 1100 };
const context = await browser.newContext({ viewport: VIEWPORT, recordVideo: { dir: out, size: VIEWPORT } });
const page = await context.newPage();
await page.goto(URL, { waitUntil: 'networkidle' });
if (PAGE_MODE === 'flying') await page.click('.mode-tab[data-mode="flying"]');
const canvas = page.locator('canvas.frame');
await canvas.waitFor({ state: 'visible', timeout: 120_000 });
await page.waitForTimeout(2000);
for (let i = 0; i < SEEDS; i++) {
  await page.keyboard.press('n');
  await page.waitForTimeout(500);
}
await page.keyboard.press('r');
await page.waitForTimeout(300);

const box = await canvas.boundingBox();
const script = SCRIPTS[MODE];
const held = new Set();
const log = [];
const videoStart = await page.evaluate(() => performance.now());
const t0 = Date.now();
while ((Date.now() - t0) / 1000 < SECONDS) {
  const t = (Date.now() - t0) / 1000;
  const want = new Set(script.filter(([a, b]) => t >= a && t < b).map(([, , k]) => k));
  for (const k of held) if (!want.has(k)) { await page.keyboard.up(k); held.delete(k); }
  for (const k of want) if (!held.has(k)) { await page.keyboard.down(k); held.add(k); }
  const hud = await page.evaluate(() => document.body.innerText.match(/fps[^\n]*/i)?.[0] ?? '');
  log.push({ t: +t.toFixed(3), keys: [...held], hud });
  await page.waitForTimeout(100);
}
for (const k of held) await page.keyboard.up(k);
const video = page.video();
await context.close();
await browser.close();
const videoPath = await video.path();
await writeFile(join(out, 'controls.jsonl'), log.map((l) => JSON.stringify(l)).join('\n') + '\n');
await writeFile(join(out, 'canvas.json'), JSON.stringify({ box, video: videoPath, record_seconds: SECONDS, script }, null, 2));
console.log(`recorded ${SECONDS}s -> ${videoPath}`);
