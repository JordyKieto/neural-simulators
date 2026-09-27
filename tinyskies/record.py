"""Record TinySkies gameplay sessions in the browser (frames + WASD controls).

Game: TinySkies (GlobeFly) by Danny Limanseta, https://github.com/dannylimanseta/tinyskies

Opens the game in a headed Chromium window (Playwright, 1280x720), injects a WASD key logger, and
for every step saves a JPEG screenshot (quality 85) together with the key state at the midpoint of
that screenshot's capture (replayed from the key-event log). Controls come from either a random
WASD bot (hold 0.15-1.25 s, gap 0.05-0.45 s, at most 2 keys down) or a human (--manual).

The requested --fps is an upper bound: the effective frame rate is limited by screenshot capture
(about 7.5 fps observed for the released dataset), so always use the recorded timestamps.

Output, one directory per session (the schema of samples/tinyskies/browser_seeds/session_0005):

    recordings/session_XXXX/
        frames/00000000.jpg ...
        samples.jsonl    {frame_idx, timestamp_ms, frame_path, w_down, a_down, s_down, d_down}
        events.jsonl     {type: keydown|keyup, key, timestamp_ms}
        metadata.json    recording settings

With --game-state, each sample also stores screenshot_start_ms / screenshot_end_ms and the game's
own state (window.__tinyskies.getDatasetState, if the game build exposes it; otherwise null).

Next step: python tinyskies/build_dataset.py --recordings recordings --out DIR

Usage (requires `pip install playwright && playwright install chromium`):
    python tinyskies/record.py --sessions 10                 # random-WASD bot, 60 s each
    python tinyskies/record.py --manual --sessions 1         # drive yourself
"""

import argparse
import asyncio
import json
import random
import re
import time
from pathlib import Path

WIDTH, HEIGHT = 1280, 720
KEYS = ("w", "a", "s", "d")
GO_BUTTON = "button#btn-fly.lobby-fly"

RECORDER_JS = """
(() => {
  if (window.__tinyskiesRecorder) return;
  const KEYS = ["w", "a", "s", "d"];
  const nowMs = () => performance.timeOrigin + performance.now();
  window.__recordedEvents = [];
  window.__keysDown = { w: false, a: false, s: false, d: false };

  window.addEventListener("keydown", (event) => {
    const key = event.key.toLowerCase();
    if (!KEYS.includes(key)) return;
    if (!window.__keysDown[key]) window.__recordedEvents.push({ type: "keydown", key, timestamp_ms: nowMs() });
    window.__keysDown[key] = true;
  });
  window.addEventListener("keyup", (event) => {
    const key = event.key.toLowerCase();
    if (!KEYS.includes(key)) return;
    window.__recordedEvents.push({ type: "keyup", key, timestamp_ms: nowMs() });
    window.__keysDown[key] = false;
  });

  window.__tinyskiesRecorder = {
    now: nowMs,
    // Key state at time t, replayed from the event log.
    keysAt(t) {
      const state = { w: false, a: false, s: false, d: false };
      for (const e of window.__recordedEvents) {
        if (e.timestamp_ms > t) break;
        state[e.key] = e.type === "keydown";
      }
      return state;
    },
    // Optional game-side hook; null if this build of the game does not expose it.
    gameState(t) {
      const hook = window.__tinyskies && window.__tinyskies.getDatasetState;
      if (typeof hook !== "function") return null;
      try { return JSON.parse(JSON.stringify(hook(t))); }
      catch (error) { return { hook_error: String(error && error.message ? error.message : error) }; }
    },
  };
})();
"""


def next_session_number(out_dir):
    numbers = [int(m.group(1)) for p in Path(out_dir).glob("session_*") if (m := re.fullmatch(r"session_(\d+)", p.name))]
    return max(numbers, default=0) + 1


async def random_wasd_driver(page, stop, args):
    """Hold 1..max_keys random keys for a random time, release all, wait a random gap, repeat."""
    rng = random.Random(args.seed)
    down = set()

    async def set_keys(target):
        for key in sorted(down - target):
            await page.keyboard.up(key)
            down.discard(key)
        for key in sorted(target - down):
            await page.keyboard.down(key)
            down.add(key)

    async def wait(seconds):
        try:
            await asyncio.wait_for(stop.wait(), timeout=seconds)
            return True
        except asyncio.TimeoutError:
            return False

    try:
        while not stop.is_set():
            await set_keys(set(rng.sample(KEYS, k=rng.randint(1, args.max_keys))))
            if await wait(rng.uniform(args.min_hold, args.max_hold)):
                break
            await set_keys(set())
            if await wait(rng.uniform(args.min_gap, args.max_gap)):
                break
    finally:
        await set_keys(set())


async def record_session(page, session_dir, number, args):
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "game_url": args.url,
        "fps": args.fps,
        "width": WIDTH,
        "height": HEIGHT,
        "record_seconds": args.seconds,
        "created_at_unix": time.time(),
        "session_number": number,
        "total_sessions": args.sessions,
        "manual_mode": args.manual,
        "random_seed": args.seed,
        "post_go_wait_seconds": args.post_go_wait,
        "random_keys": list(KEYS),
        "random_min_hold_seconds": args.min_hold,
        "random_max_hold_seconds": args.max_hold,
        "random_min_gap_seconds": args.min_gap,
        "random_max_gap_seconds": args.max_gap,
        "random_max_keys_down": args.max_keys,
    }
    if args.game_state:
        metadata["game_state"] = True
    (session_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    await page.goto(args.url)
    await page.evaluate(RECORDER_JS)
    stop, bot = asyncio.Event(), None
    tag = f"[{number}/{args.sessions}] {session_dir.name}"
    if args.manual:
        print(f"{tag}: manual mode - click into the game and drive with WASD.")
    else:
        await page.wait_for_selector(GO_BUTTON, state="visible", timeout=30000)
        await page.click(GO_BUTTON)
        await asyncio.sleep(args.post_go_wait)
        bot = asyncio.create_task(random_wasd_driver(page, stop, args))
    print(f"{tag}: recording {args.seconds} s")

    now = "window.__tinyskiesRecorder.now()"
    start, frame_idx = time.time(), 0
    with (session_dir / "samples.jsonl").open("w") as samples:
        while time.time() - start < args.seconds:
            loop_start = time.time()
            frame_path = frames_dir / f"{frame_idx:08d}.jpg"
            shot_start = await page.evaluate(now)
            await page.screenshot(path=str(frame_path), type="jpeg", quality=85)
            shot_end = await page.evaluate(now)
            timestamp = (shot_start + shot_end) / 2.0
            keys = await page.evaluate("(t) => window.__tinyskiesRecorder.keysAt(t)", timestamp)
            sample = {"frame_idx": frame_idx, "timestamp_ms": timestamp, "frame_path": str(frame_path)}
            sample.update({f"{k}_down": keys[k] for k in KEYS})
            if args.game_state:
                sample["screenshot_start_ms"] = shot_start
                sample["screenshot_end_ms"] = shot_end
                sample["game_state"] = await page.evaluate("(t) => window.__tinyskiesRecorder.gameState(t)", timestamp)
            samples.write(json.dumps(sample) + "\n")
            frame_idx += 1
            await asyncio.sleep(max(0.0, 1.0 / args.fps - (time.time() - loop_start)))

    stop.set()
    if bot is not None:
        await bot
    events = await page.evaluate("window.__recordedEvents")
    with (session_dir / "events.jsonl").open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    elapsed = time.time() - start
    print(f"{tag}: {frame_idx} frames in {elapsed:.1f} s ({frame_idx / elapsed:.1f} fps effective)")


async def run(args):
    from playwright.async_api import async_playwright

    first = args.first_session or next_session_number(args.out)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": WIDTH, "height": HEIGHT})
        page = await context.new_page()
        await page.add_init_script(RECORDER_JS)
        for i in range(args.sessions):
            await record_session(page, args.out / f"session_{first + i:04d}", i + 1, args)
        await browser.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="https://tinyskies.vercel.app/")
    parser.add_argument("--out", type=Path, default=Path("recordings"))
    parser.add_argument("--first-session", type=int, help="number of the first session directory (default: next free)")
    parser.add_argument("--sessions", type=int, default=1, help="number of sessions to record")
    parser.add_argument("--seconds", type=float, default=60, help="recording length per session")
    parser.add_argument("--fps", type=float, default=15, help="target rate; capture usually limits it to ~7.5")
    parser.add_argument("--manual", action="store_true", help="drive yourself instead of the random-WASD bot")
    parser.add_argument("--seed", type=int, help="random seed for the bot")
    parser.add_argument("--post-go-wait", type=float, default=5, help="seconds between clicking GO and recording")
    parser.add_argument("--min-hold", type=float, default=0.15)
    parser.add_argument("--max-hold", type=float, default=1.25)
    parser.add_argument("--min-gap", type=float, default=0.05)
    parser.add_argument("--max-gap", type=float, default=0.45)
    parser.add_argument("--max-keys", type=int, default=2, help="maximum keys held at once by the bot")
    parser.add_argument("--game-state", action="store_true", help="also store the game's own state per sample")
    args = parser.parse_args()
    try:
        import playwright  # noqa: F401
    except ImportError:
        parser.exit(1, "playwright is required: pip install playwright && playwright install chromium\n")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
