# AI Use Log — ARI 510 Lab 3

Tool: Claude Code (Claude Opus 5.5, agentic coding assistant in VS Code). One entry per step; used for the Generative AI Use Statement (PDF §11).

---

## 2026-11-07 — Step 1: Project setup, dataset download, verification

**Mode:** Creation (code, config, docs) + Information (inspecting the dataset repo).

**What I asked:** Step 1 only, with no analysis, embeddings or training: (1) write CLAUDE.md with the project context (models, splits, rules, layout, this log); (2) detect OS/Python/GPU, create a `.venv`, install torch/torchvision (CUDA build if a GPU exists), transformers, open_clip_torch, huggingface_hub, pandas, scikit-learn, matplotlib, pillow, tqdm, ipykernel, jupyter, and write requirements.txt and .gitignore; (3) inspect HF repo `cs5242-hateful-memes/hateful-memes-data` before downloading anything large, then download only train/dev_seen/test_seen and the images they reference, resumably (fallback `neuralcatcher/hateful_memes` if broken); (4) verify the download (row counts, % hateful, columns, missing images, 5 random PIL opens, disk size) without printing meme texts; (5) save the checks as `src/data.py` with `load_split(name)` and `python -m src.data`.

**What Claude did:**
- Read the lab PDF. Pointed out an inconsistency: Part 0 says to hand-label 10 dev memes, but the results checklist says 20 (recorded in CLAUDE.md: do 20).
- Environment: Windows 11; Python 3.11.9 picked for the venv (3.14 also installed but too new for some ML wheels); GPU = NVIDIA RTX 5060 Laptop, 8 GB, compute capability 12.0 (Blackwell), so installed the PyTorch CUDA 12.8 wheels (torch 2.11.0+cu128). Confirmed `torch.cuda.is_available()` and ran a matmul on the GPU. Registered a Jupyter kernel "Python (hateful-memes)".
- Inspected the repo with the HF API: 10,019 files / 7.79 GB = `img/` (10,000 PNGs, 3.6 GB) + 5 jsonl files + `data/` (11 parquet files, 4.2 GB). The repo README confirms `data/` is a parquet copy of the same splits, so it was skipped. The README also says the neuralcatcher mirror is missing ~1,941 images that this mirror recovered from Kaggle, so the fallback was not needed and would have been worse.
- All 10,000 images in `img/` are referenced by train+dev_seen+test_seen (8,500+500+1,000, no overlap), so the whole `img/` folder was needed (~3.6 GB). Wrote `src/download.py`: it downloads only the needed files with `hf_hub_download` and skips anything already on disk. Re-running it downloaded 0 files.
- Wrote `src/data.py` (`load_split`, `verify`), CLAUDE.md, requirements.txt (pinned), .gitignore, and empty `embeddings/ notebooks/ figures/ results/` folders.
- Slip: while peeking at the jsonl format, the first few words of 3 training meme texts were printed (a redaction filter in the shell command was applied in the wrong order). No other texts were printed.

**Results:** train 8,500 (35.5% hateful), dev 500 (49.4%), test 1,000 (49.0%); columns id/img/label/text; 0 missing images; all 10,000 images fully decode (9,995 RGB, 5 RGBA); sizes 94–825 px; data/ = 3.61 GB.

**How I checked it:** _(fill in)_
