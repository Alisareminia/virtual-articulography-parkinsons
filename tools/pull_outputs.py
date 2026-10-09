"""Download only small result files from a Kaggle kernel's output (skips audio and anything large).

Usage: .venv/bin/python tools/pull_outputs.py <owner/kernel-slug> <dest_dir> [--max-mb 20]
"""
import argparse
import json
import urllib.request
from pathlib import Path

KEEP = (".txt", ".csv", ".png", ".json", ".log", ".md", ".pdf", ".svg", ".npz")


def api(url, token):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel")
    ap.add_argument("dest")
    ap.add_argument("--max-mb", type=float, default=20)
    ap.add_argument("--audio", action="store_true", help="also download small .ogg listening clips")
    ap.add_argument("--nested", default=None, help="also take files inside this output subfolder (flattened into dest)")
    a = ap.parse_args()
    token = (Path.home() / ".kaggle" / "access_token").read_text().strip()
    owner, slug = a.kernel.split("/")
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    page, files = "", []
    while True:
        d = api(f"https://www.kaggle.com/api/v1/kernels/output?userName={owner}&kernelSlug={slug}"
                + (f"&pageToken={page}" if page else ""), token)
        files += d.get("files", [])
        if d.get("log"):
            (dest / "kernel.log").write_text(d["log"])
        page = d.get("nextPageToken")
        if not page:
            break
    skipped = 0
    for f in files:
        name = f["fileName"]
        keep = KEEP + ((".ogg",) if a.audio else ())
        if "/" in name and a.nested and name.startswith(a.nested.rstrip("/") + "/") and name.count("/") == 1:
            src, name = name, name.split("/", 1)[1]
        elif "/" in name:
            skipped += 1
            continue
        if not name.lower().endswith(keep):
            skipped += 1
            continue
        if (dest / name).exists() and name.lower().endswith(".ogg"):
            continue   # clips never change once written; skip on re-runs after a dropped connection
        for attempt in range(4):
            try:
                with urllib.request.urlopen(f["url"], timeout=60) as r:
                    size = int(r.headers.get("Content-Length") or 0)
                    if size > a.max_mb * 1e6:
                        skipped += 1
                        break
                    (dest / name).write_bytes(r.read())
                print(f"  {name} ({size / 1e3:.0f} kB)")
                break
            except OSError as ex:
                if attempt == 3:
                    print(f"  FAILED {name}: {ex}")
    print(f"downloaded to {dest}; skipped {skipped} file(s) (audio/large/nested)")


if __name__ == "__main__":
    main()
