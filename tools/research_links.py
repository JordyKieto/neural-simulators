"""One link block for every project page: GitHub READMEs, Hugging Face model cards, Hugging Face dataset cards.

Each page gets the same row as https://jordykieto.tech/research (Paper · Code · Models · Datasets · Demo)
followed by links to the other projects. The block sits between two HTML comments and is regenerated in
place, so rerunning is safe.

    python tools/research_links.py block neural-simulators     # print one project's block
    python tools/research_links.py file README.md neural-simulators
    python tools/research_links.py hf                           # update all 8 Hugging Face cards
"""

import re
import sys
from urllib.parse import quote

START, END = "<!-- research-links:start -->", "<!-- research-links:end -->"
SITE = "https://jordykieto.tech/research"
HF, GH = "https://huggingface.co", "https://github.com/JordyKieto"

# Newest first, matching the website.
PROJECTS = {
    "ppo-hyrule": dict(
        title="The Legend of PPO: Learning to Fight in Hyrule",
        paper=("GitHub", f"{GH}/ppo-hyrule/blob/main/paper/paper.md"),
        code=f"{GH}/ppo-hyrule",
        models=[("ppo-hyrule", f"{HF}/jordykieto/ppo-hyrule")],
        datasets=[("ppo-hyrule-data", f"{HF}/datasets/jordykieto/ppo-hyrule-data")],
        demo=("YouTube", "https://youtu.be/FBD1wEsdY3E"),
    ),
    "puffermoba-dyna": dict(
        title="Learning in Dreams, Winning in Reality",
        paper=("arXiv 2610.08033", "https://arxiv.org/abs/2610.08033"),
        code=f"{GH}/puffermoba-dyna",
        models=[("puffermoba-dyna", f"{HF}/jordykieto/puffermoba-dyna")],
        datasets=[("puffermoba-dyna-games", f"{HF}/datasets/jordykieto/puffermoba-dyna-games")],
        demo=("YouTube", "https://youtu.be/Uy0OsZlnd5Y"),
    ),
    "neural-simulators": dict(
        title="Neural Simulators for Interactive Environments",
        paper=("GitHub", f"{GH}/neural-simulators/blob/main/paper/paper.pdf"),
        code=f"{GH}/neural-simulators",
        models=[("neural-simulators", f"{HF}/jordykieto/neural-simulators")],
        datasets=[("tinyskies-gameplay", f"{HF}/datasets/jordykieto/tinyskies-gameplay"),
                  ("carla-driving-demos", f"{HF}/datasets/jordykieto/carla-driving-demos")],
        demo=("YouTube", "https://youtu.be/FcPuaWpXE0w"),
        play="https://drift-sim-production.up.railway.app/",
    ),
    "carla-imitation-learning": dict(
        title="Imitation Learning for Autonomous Driving in CARLA",
        paper=("arXiv 2609.17757", "https://arxiv.org/abs/2609.17757"),
        code=f"{GH}/carla-imitation-learning",
        models=[("carla-imitation-learning", f"{HF}/jordykieto/carla-imitation-learning")],
        datasets=[("carla-driving-demos", f"{HF}/datasets/jordykieto/carla-driving-demos")],
        demo=("YouTube", "https://youtu.be/rr_uS4bf0B4"),
    ),
}

# Hugging Face repo -> the project whose block it carries.
HF_CARDS = {
    ("model", "ppo-hyrule"): "ppo-hyrule",
    ("model", "puffermoba-dyna"): "puffermoba-dyna",
    ("model", "neural-simulators"): "neural-simulators",
    ("model", "carla-imitation-learning"): "carla-imitation-learning",
    ("dataset", "ppo-hyrule-data"): "ppo-hyrule",
    ("dataset", "puffermoba-dyna-games"): "puffermoba-dyna",
    ("dataset", "tinyskies-gameplay"): "neural-simulators",
    ("dataset", "carla-driving-demos"): "carla-imitation-learning",
}

LOGO = {"arXiv": ("arxiv", "b31b1b"), "GitHub": ("github", "24292f"), "YouTube": ("youtube", "ff0000"),
        "HF": ("huggingface", "ffd21e"), "Play": ("googlechrome", "2b7a78")}


def badge(label, message, kind, href):
    def esc(s):
        return quote(s.replace("-", "--").replace("_", "__"), safe="")
    logo, color = LOGO[kind]
    # The Hugging Face mark keeps its own colours; the others are white on their brand colour.
    tint = "" if kind == "HF" else "&logoColor=white"
    src = f"https://img.shields.io/badge/{esc(label)}-{esc(message)}-{color}?logo={logo}{tint}"
    return f'<a href="{href}"><img alt="{label}: {message}" src="{src}"></a>'


def block(key):
    p = PROJECTS[key]
    paper_kind = "arXiv" if p["paper"][1].startswith("https://arxiv.org") else "GitHub"
    row = [badge("Paper", p["paper"][0], paper_kind, p["paper"][1]),
           badge("Code", "GitHub", "GitHub", p["code"])]
    row += [badge("Models", "Hugging Face", "HF", href) for _, href in p["models"]]
    many = len(p["datasets"]) > 1
    row += [badge("Datasets", name if many else "Hugging Face", "HF", href) for name, href in p["datasets"]]
    row.append(badge("Demo", p["demo"][0], "YouTube", p["demo"][1]))
    if p.get("play"):
        row.append(badge("Play", "in your browser", "Play", p["play"]))
    others = " · ".join(f'<a href="{q["code"]}">{q["title"]}</a>' for k, q in PROJECTS.items() if k != key)
    return "\n".join([
        START,
        '<p align="center">',
        *(f"  {b}" for b in row),
        "</p>",
        f'<p align="center"><sub>More research: {others} · <a href="{SITE}">all research ↗</a></sub></p>',
        END,
    ])


def apply(text, key):
    """Replace an existing block, or insert one right after the first H1 (or the first HTML <h1>)."""
    new = block(key)
    if START in text:
        return re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: new, text, flags=re.S)
    m = re.search(r"^# .*$", text, flags=re.M) or re.search(r"^<h1.*?</h1>.*$", text, flags=re.M)
    if not m:
        raise SystemExit("no H1 to anchor the block under")
    return text[:m.end()] + "\n\n" + new + text[m.end():]


def update_hf():
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import pod
    from huggingface_hub import HfApi, hf_hub_download
    token = pod.secret("HUGGING_FACE_KEY")
    api = HfApi(token=token)
    for (kind, name), key in HF_CARDS.items():
        repo = f"jordykieto/{name}"
        path = hf_hub_download(repo, "README.md", repo_type=kind, token=token, force_download=True)
        old = open(path).read()
        new = apply(old, key)
        if new == old:
            print(f"{kind:7s} {repo}: unchanged")
            continue
        api.upload_file(path_or_fileobj=new.encode(), path_in_repo="README.md", repo_id=repo, repo_type=kind,
                        commit_message="Card: standard research link row")
        print(f"{kind:7s} {repo}: updated")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "block":
        print(block(sys.argv[2]))
    elif cmd == "file":
        f, key = sys.argv[2], sys.argv[3]
        text = open(f).read()
        open(f, "w").write(apply(text, key))
    elif cmd == "hf":
        update_hf()
    else:
        print(__doc__)
