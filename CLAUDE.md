# ARI 510 Lab 3 — Multimodal Classification with Pretrained Encoders (Hateful Memes)

Lab spec: `ARI410-510_Lab3-1.pdf` (in repo root). The user is doing the lab in explicit steps; do only the step asked for, then stop.

## Task
Binary classification (1 = hateful, 0 = not hateful) of memes, using **frozen** pretrained encoders + small trainable classifiers.

- **Text-only:** BERT `bert-base-uncased`, last-layer `[CLS]` hidden state → 768-d.
- **Image-only:** torchvision ResNet-18 (ImageNet weights) with `fc = nn.Identity()` → 512-d.
- **Early fusion:** `torch.cat((text, image), dim=1)` → 1280-d, new classifier.
- **Late fusion:** average the predicted probabilities of the trained text-only and image-only classifiers (no new training).
- **Classifier:** small feedforward net (Linear → ReLU → Linear → 1 logit), mini-batches, `nn.BCEWithLogitsLoss`, Adam. Write ONE reusable train function (train/dev vectors + labels → dev metrics + loss curves); it's meant to be reused in project Phase 2b.
- **Metrics (dev, and once on test):** accuracy, precision, recall, F1, ROC-AUC. Also an always-"not hateful" baseline. Class balance differs between train and dev/test — keep that in mind when deciding which metrics to trust.
- **Sanity checks (text model):** dumb baseline; initial loss ≈ ln 2 ≈ 0.69; overfit 20 examples to ~0 loss / 100% acc; train/dev loss curve per epoch. Every model needs a train/dev loss plot.
- **Part 0 (before any training):** label-distribution plot per split; look at ≥10 train memes; "be the model" on dev memes for a human baseline. NOTE: PDF Part 0 says 10 dev memes, but the Results checklist says 20 — do 20 to satisfy both.
- **ARI 510 design change:** (a) CLIP ViT-B/32 embeddings (via `open_clip`) as an alternative encoder, and (b) weighted late fusion `p = w·p_text + (1−w)·p_image` with `w` tuned on dev. Write the hypothesis (which modality is more useful; how much fusion helps) BEFORE running Part 3.
- **ARI 410 analysis (also doing it):** find dev memes where fusion and single-modality models disagree — ≥2 where fusion was right but a single-modality model was wrong, ≥1 where fusion was wrong; record ids + descriptions.

## Splits
- `train.jsonl` = train (8,500), `dev_seen.jsonl` = dev (500), `test_seen.jsonl` = test (1,000).
- Ignore all `*unseen*` files.
- Columns: `id`, `img` (relative path like `img/42953.png`), `text`, `label`.

## Rules
- Every decision (hidden size, epochs, lr, fusion weight, threshold, …) is made on **dev**.
- The **test set is evaluated exactly once**, at the very end, for all models together.
- Fix random seeds (python, numpy, torch, cuda).
- Compute embeddings **once** and cache them to `embeddings/*.pt`; load from cache afterwards.
- Develop on a small subset (~200 examples) first, then run on everything.
- Content: the dataset contains hateful content. Never print meme texts in bulk; show counts/ids, and only specific examples when the user asks.

## Layout
- `data/hateful_memes/` — dataset (jsonl + `img/`), git-ignored. Download with `python -m src.download` (resumable).
- `embeddings/` — cached embedding tensors (`.pt`), git-ignored.
- `src/` — reusable code. `src/data.py`: `load_split(name)` → DataFrame with absolute `path` column; `python -m src.data` re-runs dataset checks.
- `src/plots.py` (figures), `src/metrics.py` (`compute_metrics(y_true, y_pred, y_prob=None)` → accuracy/precision/recall/F1[/AUROC], zero_division=0 — use for every model), `src/human_eval.py` (Part 0.3 blind labeling; never overwrites `results/human_labels.csv`).
- `notebooks/lab3.ipynb` — the single submitted notebook (kernel "Python (hateful-memes)"); later steps append sections. Keep cells short (imports, calls, plots); logic goes in `src/`. Inline figures are JPEG to keep it small. Execute with `jupyter nbconvert --to notebook --execute --inplace notebooks/lab3.ipynb`.
- `notes/` — the user's own writing (`part0_notes.md`, `work_log.md`). Do NOT fill these in.
- `figures/`, `results/` (`human_eval_ids.csv`, `human_labels.csv` filled in by the user).
- Git: local repo on `main`; private GitHub remote `hateful-memes` (never public). `embeddings/` is git-ignored, so code must `mkdir` it.
- Part 0.3 blindness: never reveal labels of the 20 ids in `results/human_eval_ids.csv` in chat, filenames, or the labeling cell until the user has filled in `human_labels.csv`.
- `ai_log.md` — after every step, append: date, step, what the user asked, what Claude did. Needed for the required Generative AI Use Statement (PDF §11).

## Environment
- Windows 11, Python 3.11 venv in `.venv` (`.venv\Scripts\python.exe`). Python 3.14 is also installed but not used.
- GPU: NVIDIA RTX 5060 Laptop (8 GB, Blackwell sm_120) → requires PyTorch CUDA 12.8 wheels (`--index-url https://download.pytorch.org/whl/cu128`).
- Dataset source: HF `cs5242-hateful-memes/hateful-memes-data` (repo_type=dataset). Its `data/*.parquet` is a duplicate of the jsonl+img content and is not downloaded.
