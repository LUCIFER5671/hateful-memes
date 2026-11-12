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

---

## 2026-11-07 — Step 2: Git setup + Part 0 scaffolding (label distribution, examples, blind "be the model")

**Mode:** Creation (code, notebook, templates).

**What I asked:** git init + first commit + private GitHub repo via `gh` (or the commands if gh isn't available); create `notebooks/lab3.ipynb` (short cells, logic in `src/`); Part 0.1 label-distribution plot (`src/plots.py`, `figures/label_distribution.png`); Part 0.2 `show_meme` helper + a 12-meme training grid (6/6, fixed seed) + an empty-headings `notes/part0_notes.md`; Part 0.3 pick 20 dev memes blind (ids to `results/human_eval_ids.csv`, an empty `results/human_labels.csv`, a labeling cell without labels, a scoring cell using a new `src/metrics.compute_metrics`); an empty `notes/work_log.md` template; execute the notebook; don't describe meme content.

**What Claude did:**
- `git init -b main`; confirmed via `git check-ignore` that .venv/, data/, embeddings/, __pycache__/ and .ipynb_checkpoints/ are ignored; first commit; confirmed 0 tracked files under data/ or embeddings/. **gh CLI is not installed**, so the GitHub repo was not created; Claude gave me the commands to run myself.
- Added `show_meme`, `show_memes` (grid) and `sample_balanced` to `src/data.py`; `src/plots.py` (`plot_label_distribution`); `src/metrics.py` (`compute_metrics`); `src/human_eval.py` (`prepare()` picks 20 dev ids with a plain random sample, seed 510, and writes the id list + an empty sheet without overwriting an existing one; `score()` prints "fill in … first" if any guess is missing, else joins the true labels and returns metrics + the wrong memes).
- Built `notebooks/lab3.ipynb` (Setup, 0.1, 0.2, 0.3 labeling + scoring cells) and executed it. All cells ran; the scoring cell printed the expected "fill in results/human_labels.csv first".
- Fixes during the step: (1) the first version of the label plot had the "% hateful" labels overlapping the bars for dev/test (dev is 17x smaller than train on a shared y-axis), so it was redrawn as one panel per split with its own y-scale; (2) the executed notebook was 7.3 MB because the meme grids were embedded as PNG, so inline figures were switched to JPEG (now ~1.2 MB).
- Smoke-tested `score()` on a throw-away copy of the sheet filled with random guesses (in a temp folder; nothing about labels printed) and `compute_metrics` on a toy example.
- Claude did not view or describe any meme images or texts.

**How I checked it:** _(fill in)_

---

## 2026-11-12 — Step 3: Compute and cache embeddings (BERT, ResNet-18, CLIP)

**Mode:** Creation (code) + Advice/feedback (Claude flagged a CLIP configuration problem).

**What I asked:** Write `src/embed.py` with three frozen encoders on the GPU (BERT-base-uncased [CLS] at max_length 128; ResNet-18 with fc=Identity and its own transforms; open_clip ViT-B-32 "openai", with image and text saved separately and raw). Cache to `embeddings/{split}_{encoder}_{tag}.pt` + meta csv, loading instead of recomputing, with a CLI. Run on a 200-item subset first and check shapes / NaN / row order / determinism (re-embed 4 items, allclose atol 1e-4), then run on everything. Add `src/features.py` `load_features`, add an "Embeddings" notebook section with an empty "Why remove ResNet's final layer?" cell for me, commit and push.

**What Claude did:**
- Wrote `src/embed.py` (encoders, caching, `--check`) and `src/features.py`. Determinism measures: TF32 off, deterministic cuDNN, fixed seeds, and BERT padded to a fixed 128 tokens so a text's vector doesn't depend on the other texts in its batch. Image DataLoader uses num_workers=0 (Windows).
- **Problem found:** the first subset run printed `QuickGELU mismatch ... pretrained tag 'openai' (quick_gelu=True)`. OpenAI's CLIP weights were trained with QuickGELU but open_clip's plain "ViT-B-32" config uses GELU, so the CLIP vectors would have been silently slightly wrong. Fixed with `force_quick_gelu=True`, deleted the bad CLIP cache and recomputed it. Verified the fix: CLIP picks each meme's own text out of 200 candidates 62–65% of the time (chance = 0.5%).
- Subset (200) checks: all shapes right, all finite, ids/labels match jsonl order, determinism max |diff| ≤ 2.5e-5. Then the full run (timings below) passed the same checks; the first 200 rows of the full run match the subset run (≤ 2.6e-5).
- Extra check: many rows are exact duplicates. Claude counted them (counts only, no text) and found the dataset's confounder design: e.g. in dev, 79 groups of memes share identical text, and 78 of those groups contain both labels; 22% of dev texts and 19% of test texts also appear in train. Identical images are rarer (dev: 3 groups).
- Executed the notebook; the "Embeddings" section prints all 12 shapes. The ResNet explanation cell was left empty for me.

**Results:** full-run time 317 s on the RTX 5060 (BERT 48 s, ResNet-18 110 s, CLIP 144 s for all splits); shapes train 8500 / dev 500 / test 1000 × (768 BERT, 512 ResNet, 512 CLIP image, 512 CLIP text); `embeddings/` = 94 MB (30 files incl. the 200-subsets), git-ignored.

**How I checked it:** _(fill in)_

---

## 2026-11-12 — Part 0.3: Human baseline ("be the model", 20 dev memes)

**Mode:** Creation (data entry, small code change).

**What I asked:** Enter my own blind guesses (`my_guess`), `needed_both`, and a new `modality` column (both/image/text) into `results/human_labels.csv` in row order without changing ids; confirm 20 rows in the same id order as `results/human_eval_ids.csv`; run the scoring cell; report the metrics, the ids I got wrong (true label + modality) and accuracy per modality group; commit.

**What Claude did:**
- Entered my values exactly as given (the labeling decisions were mine, made blind before seeing labels). Verified 20 rows, ids unchanged and in the same order as `human_eval_ids.csv`, and that `needed_both = y` exactly where `modality = both`.
- Extended `src/human_eval.score()` to check the id order and return per-modality accuracy; updated the notebook's scoring cell to show it plus a table of the wrong ids; re-executed the notebook.

**Results:** accuracy 0.65 (13/20), precision 0.444, recall 0.667, F1 0.533. Wrong (my guess → true label): 89425 (0→1, both), 73021 (1→0, both), 37160 (1→0, both), 61085 (1→0, image), 23810 (0→1, text), 41728 (1→0, both), 37420 (1→0, both). Accuracy by modality: both 4/9 (0.44), image 4/5 (0.80), text 5/6 (0.83). The 20-meme sample has 6 hateful (30%) vs 49% in the full dev set.

**How I checked it:** _(fill in)_
