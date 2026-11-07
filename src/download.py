"""Resumable download of the Hateful Memes splits we use (train / dev_seen / test_seen).

Source: HF dataset repo "cs5242-hateful-memes/hateful-memes-data" (course-recommended mirror).
- Downloads only train.jsonl, dev_seen.jsonl, test_seen.jsonl (+ README/LICENSE) and the
  images those three files reference.
- Skips the *unseen* splits and the data/*.parquet files (a duplicate parquet copy of all
  splits with the images embedded, ~4.2 GB).
- Resumable: any file already present on disk (non-empty) is not downloaded again.

Usage:  python -m src.download
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ID = "cs5242-hateful-memes/hateful-memes-data"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "hateful_memes"
SPLIT_FILES = ["train.jsonl", "dev_seen.jsonl", "test_seen.jsonl"]
META_FILES = ["README.md", "LICENSE.txt"]


def _exists(rel: str) -> bool:
    p = DATA_DIR / rel
    return p.is_file() and p.stat().st_size > 0


def _fetch(rel: str, retries: int = 5) -> str:
    for attempt in range(retries):
        try:
            hf_hub_download(REPO_ID, rel, repo_type="dataset", local_dir=DATA_DIR)
            return rel
        except Exception as e:  # network hiccup / rate limit -> back off and retry
            if attempt == retries - 1:
                raise RuntimeError(f"failed to download {rel}: {e}") from e
            time.sleep(2 ** attempt * 5)
    return rel


def needed_images() -> list[str]:
    imgs = set()
    for f in SPLIT_FILES:
        with open(DATA_DIR / f, encoding="utf-8") as fh:
            imgs.update(json.loads(line)["img"] for line in fh if line.strip())
    return sorted(imgs)


def main(workers: int = 8) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for f in SPLIT_FILES + META_FILES:
        if not _exists(f):
            _fetch(f)

    imgs = needed_images()
    todo = [i for i in imgs if not _exists(i)]
    print(f"{len(imgs)} images referenced; {len(imgs) - len(todo)} already on disk; {len(todo)} to download")
    failed = []
    if todo:
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(_fetch, rel): rel for rel in todo}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="images"):
                try:
                    fut.result()
                except Exception as e:
                    failed.append(futs[fut])
                    tqdm.write(str(e))
    print(f"done; {len(failed)} failed" + (" (re-run to retry)" if failed else ""))


if __name__ == "__main__":
    main()
