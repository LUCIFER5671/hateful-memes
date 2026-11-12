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

**How I checked it:** Confirmed the ResNet-18 weights version (ResNet18_Weights.DEFAULT) by checking it manually. Verified all 10,000 images exist using a terminal command (no missing files). Cross-checked the installed dependencies manually in the terminal.

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

**How I checked it:** Checked that the blind grid in 0.3 showed no labels before I labeled. Compared the label-distribution plot against the split percentages (35.5 / 49.4 / 49.0%). Confirmed the scoring cell refused to run until human_labels.csv was filled, and that the 20 rows matched the order of the printed sheet.

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

**How I checked it:** Checked shapes 8500/500/1000 × 768/512; noted the QuickGELU fix and asked why it mattered.

---

## 2026-11-12 — Part 0.3: Human baseline ("be the model", 20 dev memes)

**Mode:** Creation (data entry, small code change).

**What I asked:** Enter my own blind guesses (`my_guess`), `needed_both`, and a new `modality` column (both/image/text) into `results/human_labels.csv` in row order without changing ids; confirm 20 rows in the same id order as `results/human_eval_ids.csv`; run the scoring cell; report the metrics, the ids I got wrong (true label + modality) and accuracy per modality group; commit.

**What Claude did:**
- Entered my values exactly as given (the labeling decisions were mine, made blind before seeing labels). Verified 20 rows, ids unchanged and in the same order as `human_eval_ids.csv`, and that `needed_both = y` exactly where `modality = both`.
- Extended `src/human_eval.score()` to check the id order and return per-modality accuracy; updated the notebook's scoring cell to show it plus a table of the wrong ids; re-executed the notebook.

**Results:** accuracy 0.65 (13/20), precision 0.444, recall 0.667, F1 0.533. Wrong (my guess → true label): 89425 (0→1, both), 73021 (1→0, both), 37160 (1→0, both), 61085 (1→0, image), 23810 (0→1, text), 41728 (1→0, both), 37420 (1→0, both). Accuracy by modality: both 4/9 (0.44), image 4/5 (0.80), text 5/6 (0.83). The 20-meme sample has 6 hateful (30%) vs 49% in the full dev set.

**How I checked it:** Confirmed my 20 guesses matched the sheet before scoring.

---

## 2026-11-12 — Notes saved for me (between Steps 3 and 4)

**Mode:** Creation (transcription / drafting).
- ResNet final-layer answer: written by me; Claude pasted it word for word into the notebook cell.
- `notes/hypothesis_510.md`: written by me before any fusion results; Claude saved it word for word and committed it on its own (`38e954c`, "510 hypothesis (written before Part 3)").
- `notes/work_log.md` 2026-11-12 entry: at my request, **Claude drafted this entry** from facts in this session (embedding timings, the CLIP QuickGELU problem and fix, my 65% / 13 of 20 human baseline). Committed as "Work log: embedding timings, CLIP QuickGELU fix, human baseline 65% (13/20)".
- I also asked Claude a conceptual question ("why does early fusion work better than late fusion?"). Claude explained the usual reasoning (early fusion can learn text-image interactions; late fusion only averages two single-modality opinions), pointed out that we had not run fusion yet so it isn't known for our data, and listed reasons it might not hold. Mode: Information.

---

## 2026-11-12 — Step 4: Shared training function + Part 1 (text-only, BERT) + sanity checks

**Mode:** Creation (code) + Information (Claude's reading of the results).

**What I asked:** `src/models.py` (PDF classifier + optional dropout); `src/train.py` `train_classifier(...)` reused for every model (train-only standardization stored with the model, BCEWithLogitsLoss with optional pos_weight, Adam, shuffled mini-batches, seeds, per-epoch train/dev loss with dev loss always unweighted, keep the lowest-dev-loss epoch, return model/curves/best_epoch/dev metrics) + save/load; text-model sanity checks (dumb baseline, initial loss vs ln 2, overfit 20 examples for 300 epochs) saved to `results/sanity_text.json`; a 48-config grid selected by dev AUROC; retrain the chosen config with seeds 0/1/2; loss-curve plot; `results/dev_results.csv`; a "Part 1" notebook section; never load the test split, with an assert guarding it.

**What Claude did:**
- Wrote `src/models.py` (`Classifier`, `StandardizedClassifier` holding train mean/std as buffers), `src/train.py`, `src/experiments.py`; added `plot_loss_curves` to `src/plots.py`; added `results/models/` to .gitignore.
- Test guard: `load_features("test", ...)` now raises an AssertionError until `unlock_test()` is called (final step only); verified it raises. The Step 3 "Embeddings" cell used to load test features for shapes, so it was changed to `feature_shapes()` (shapes only, no labels).
- Design choices to note: train loss in the curves is computed after each epoch in eval mode, unweighted, on the full train set, so it's directly comparable to dev loss; the per-batch training objective (with dropout and pos_weight) is kept separately as `train_objective`. The grid is cached in `results/grid_text_bert.csv` so re-running the notebook doesn't re-run it. `record_dev_result` replaces a model's row in place so re-runs don't duplicate rows.
- Ran the grid (48 configs, ~10 min on GPU), the 3-seed retraining and the notebook.

**Results:**
- Sanity: (a) always "not hateful" on dev: accuracy 0.506, precision/recall/F1 0, AUROC 0.5; (b) initial loss 0.658 vs ln 2 = 0.693; (c) 20 examples × 300 epochs: final train loss 4.9e-5, train accuracy 100%.
- Chosen config (best dev AUROC 0.6336, seed 0): hidden 256, lr 1e-3, dropout 0.3, weight decay 1e-4, pos_weight 1.8155 (= n_neg/n_pos).
- Dev, 3 seeds: accuracy 0.575 ± 0.015, precision 0.595 ± 0.038, recall 0.453 ± 0.059, F1 0.512 ± 0.028, AUROC 0.621 ± 0.013.
- Observations Claude flagged: every grid run had its lowest dev loss at epoch 1 or 2 (38 at epoch 1, 10 at epoch 2), and dev loss then rises steadily (seed 0: 0.72 at epoch 2 → ~1.36 at epoch 50) while train loss falls to ~0.2, which is fast overfitting. The best dev loss in the whole grid (0.696) is no better than predicting 0.5 for everything (ln 2 = 0.693). The top 5 configs differ by only 0.002 AUROC, much less than the seed-to-seed std (0.013), so the choice among them is essentially noise. pos_weight helped on average (mean grid AUROC 0.622 vs 0.612).

**How I checked it:** I checked whether the loss curves behaved as expected (initial loss, tiny overfit, dev-loss minimum) and asked Claude to verify. The check showed best_epoch was always epoch 1–2 and that the untrained model often had a lower dev loss than any trained epoch, even though dev AUROC rose from ~0.5 to 0.62. I concluded that dev-loss early stopping conflicted with selecting by AUROC and changed it to stop on dev AUROC, with slower learning rates and up to 100 epochs. Confirmed with git grep that the training code never loads the test split.

---

## 2026-11-12 — Report: Section 1 (Looking at the Data) + Figure 1

**Mode:** Creation (file setup) + Advice/feedback (number check).

**What I asked:** Put my Section 1 text into the report with `figures/label_distribution.png` as Figure 1, and record in the AI-use table (Report writing) that I had Claude polish this section from my own notes, checked by verifying every number against `results/human_labels.csv` and the notebook output.

**What Claude did:** Created `report/report.md` with the report structure from the lab PDF (§10–11). Inserted my Section 1 text word for word with Figure 1 and a caption; left all other sections empty for me. Filled the "Report writing" row of the AI-use table as I worded it. Before inserting, Claude re-computed every number in the section (split sizes and % hateful, my 13/20 and precision/recall/F1, the 9/5/6 modality counts and 4/9, 4/5, 5/6 accuracies, the 5 false alarms and the two missed ids, 6/20 hateful in my sample); all matched. Note: the text arrived already polished; I polished it with Claude (in a separate chat) and Grammarly, not in this Claude Code session.

**How I checked it:** Verified every number against `results/human_labels.csv` and the notebook output.

---

## 2026-11-12 — Step 4 fix: early stopping on dev AUROC, slower learning rates

**Mode:** Creation (code) + Advice/feedback (Claude checked its own results and corrected them).

**What I asked:** In `train_classifier`, add `early_stop_metric` ("auroc" default, or "loss") and record epoch 0 and dev AUROC per epoch; add lr 1e-4 and 3e-5 to the grid with up to 100 epochs; rerun the grid and the 3-seed retrain with AUROC early stopping, re-save `text_bert.pt`, update `dev_results.csv`; make the curves figure two panels (loss, dev AUROC) with best_epoch marked; rerun the sanity checks; add a markdown note explaining why we stop on dev AUROC; no test split.

**What Claude did:**
- `src/train.py`: `early_stop_metric` in {"auroc" (default), "loss", "last"}; "last" was added so the tiny-batch overfit check still reports the final epoch (with "auroc" it would return whichever early epoch first hit AUROC 1.0). Curves now include epoch 0 (untrained) and `dev_auroc`; best_epoch can be 0. Per-batch loss is accumulated on the GPU (fewer syncs).
- `src/experiments.py`: grid lr {1e-3, 3e-4, 1e-4, 3e-5}, 100 epochs, AUROC early stopping (96 configs, 25 min); the grid CSV records the early-stopping setting, the max epochs and the min-dev-loss epoch, and a cached grid made with different settings now raises instead of being silently reused. `src/plots.py`: two-panel figure.
- **Bug found and fixed by Claude:** the first 3-seed retrain gave seed-0 AUROC 0.6358, but the identical grid run gave 0.6348. Claude checked that training is deterministic (same AUROC to 6 decimals within and across processes), and traced the difference to the grid CSV storing pos_weight rounded to "1.8155" while training had used the exact 1.8155018…; the retrain read back the rounded value. Fixed by storing the exact value (repr) and correcting the existing CSV in place (the grid itself had used the exact value, so its results stand). After the fix, seed 0 reproduces the grid's 0.6348 exactly.
- **Note corrected by Claude:** the first draft of the notebook note said dev AUROC "keeps improving for longer" than dev loss; the actual curves show AUROC peaks at epoch 2 and then also declines, so the note was rewritten to match (plus the finding that slower learning rates delay the AUROC peak but don't raise it).
- Sanity checks rerun in the notebook with explicit asserts: unchanged and PASS. Test split never loaded.

**Results:**
- Chosen config (best dev AUROC 0.6348): hidden 64, lr 1e-3, dropout 0, weight decay 0, pos_weight = n_neg/n_pos = 1.8155, early stopping on dev AUROC.
- best_epoch (max dev AUROC) per seed: 2, 2, 2. Epoch of minimum dev loss: 0, 0, 1.
- Dev, 3 seeds: accuracy 0.574 ± 0.012, precision 0.602 ± 0.016, recall 0.408 ± 0.035, F1 0.485 ± 0.027, AUROC 0.627 ± 0.007 (previous dev-loss-stopping version: AUROC 0.621 ± 0.013).
- Grid: dev AUROC 0.620–0.635 across all 96 configs; top 5 within 0.003 (less than the seed std). Mean AUROC by lr: 1e-3 0.629, 3e-4 0.626, 1e-4 0.626, 3e-5 0.625. Mean best epoch by lr: 2.0 / 2.7 / 6.2 / 19.0. In 68 of 96 runs the untrained model (epoch 0) has the lowest dev loss.

**How I checked it:** Checked that the sanity checks still passed after the change (initial loss 0.659, overfit 100%). Confirmed best_epoch was 2 for all three seeds from the per-seed table in the notebook, and saw on the seed-0 two-panel figure that AUROC also declines after its peak. The note's first claim contradicted this, so I had it rewritten. Noticed the retrain didn't reproduce the grid's 0.6348, and traced it to pos_weight being rounded in the CSV.

---

## 2026-11-12 — Step 5: Part 2 (image-only) + Part 3 (early and late fusion), dev only

**Mode:** Creation (code) + Information (Claude's summary of the results).

**What I asked:** Save dev predictions (id, label, prob per seed) for every model; image-only (ResNet-18) and early fusion (raw BERT + ResNet concatenated, 1280-d, standardized after concatenation) with the same 96-config grid, AUROC selection and early stopping, 3 seeds, saved grids/models/two-panel curves; late fusion as the per-seed average of the text and image probabilities (no training, threshold 0.5); add all to `dev_results.csv`; notebook sections "Part 2: Image-only" and "Part 3: Fusion" with an empty "Hypothesis check (dev)" cell for me; report without interpreting against my hypothesis; never touch the test split.

**What Claude did:**
- Added `load_concat` (`src/features.py`; checks that both encoders' rows are the same memes) and, in `src/experiments.py`, `save_dev_preds` / `load_dev_preds`, `late_fusion` (per-seed average of the saved dev probabilities, after checking ids and labels line up), `seed_metrics_from_preds`, `seed_table`, and `fit_select_record` (one pipeline for every trained model). The text model's dev predictions are now also saved from Part 1.
- Ran both grids (image-only 31 min, early fusion ~30 min), then the notebook. Late fusion uses the saved per-seed predictions, which come from the per-seed models (only the seed-0 model of each is saved to `results/models/`).
- Checks: the saved prediction files reproduce every dev_results number exactly; late_avg equals (text + image) / 2 for each seed; the test split stayed locked.
- Fixed a figure bug: in the early-fusion figure the "best_epoch" label overlapped the panel title; labels now go below the marker. All three curve figures were regenerated (numbers unchanged).
- Left the "Hypothesis check (dev)" cell empty for me.

**Results (dev, mean ± std over seeds 0/1/2):**

| model | accuracy | precision | recall | F1 | AUROC |
|---|---|---|---|---|---|
| always_not_hateful | 0.506 | 0.000 | 0.000 | 0.000 | 0.500 |
| text_bert | 0.574 ± 0.012 | 0.602 ± 0.016 | 0.408 ± 0.035 | 0.485 ± 0.027 | 0.627 ± 0.007 |
| image_resnet | 0.515 ± 0.001 | 0.569 ± 0.065 | 0.128 ± 0.098 | 0.194 ± 0.125 | 0.537 ± 0.006 |
| early_concat | 0.580 ± 0.014 | 0.663 ± 0.026 | 0.304 ± 0.021 | 0.416 ± 0.025 | 0.664 ± 0.009 |
| late_avg | 0.557 ± 0.010 | 0.639 ± 0.040 | 0.247 ± 0.067 | 0.352 ± 0.063 | 0.627 ± 0.011 |

- Chosen configs: image_resnet hidden 512, lr 1e-3, dropout 0, wd 0, no pos_weight (best_epoch 1, 1, 2); early_concat hidden 256, lr 3e-4, dropout 0, wd 0, no pos_weight (best_epoch 30, 3, 5).
- Things Claude flagged: image-only is close to chance (AUROC 0.537; grid range 0.524–0.542) and its dev AUROC falls *below* 0.5 after a few epochs (~0.46 by the end); the image grid's top 5 are within 0.001 and the chosen config (no pos_weight) has recall 0.04 at seed 0, while equally good configs with pos_weight have recall ~0.5; early fusion has the highest AUROC but lower F1 than text-only because its recall is low (no pos_weight, mean predicted probability 0.24 vs 49% hateful in dev); early fusion memorizes train (train loss → 0) and its best epoch varies a lot by seed (30 / 3 / 5), with dev loss up to 2.2 at the chosen epoch; late fusion's AUROC equals text-only's (0.627) and its F1 is lower.

**How I checked it:** Opened preds_dev_late_avg.csv and confirmed it has 500 rows with one probability column per seed. Checked the curves: image-only dev AUROC falls below 0.5 as training continues, and early fusion overfits (train loss ~0, dev loss ~2.2). Noticed that early fusion has the best AUROC but a lower F1 than text-only, which comes from its low recall without pos_weight, so I rely on AUROC to compare models.

---

## 2026-11-12 — Step 6: ARI 510 design change (CLIP encoders) + ARI 410 disagreement analysis, dev only

**Mode:** Creation (code) + Information (Claude's summary of the results).

**What I asked:** Train clip_text (512-d), clip_image (512-d), clip_early_concat (1024-d raw concat) and clip_late_avg (per-seed average) with the exact same train function, 96-config grid, AUROC selection/early stopping and 3 seeds; save grids, models, curves and dev predictions and add all four to `dev_results.csv`; a 9-model comparison table and AUROC bar chart (`figures/dev_auroc_comparison.png`) with F1 too; state the AUROC change of clip_early_concat vs early_concat as numbers only and whether it exceeds the seed std, without interpreting against my hypothesis; for the 410 analysis, seed-0 dev predictions at threshold 0.5 → `results/disagreements_dev.csv`, pick 3 memes, show them with `show_meme` and their predictions, and leave empty "410: meme <id>" cells for me; never touch the test split.

**What Claude did:**
- Added `auroc_gap` (`src/experiments.py`), `plot_auroc_comparison` (`src/plots.py`) and `src/analysis.py` (`seed_predictions`, `disagreements`, `pick_examples`, `summary`). Reused `fit_select_record` and `late_fusion` from Step 5 unchanged.
- Ran the three CLIP grids (24, 27 and 27 min), then the notebook. Moved the legends of the loss/AUROC curve figures below the axes because the best_epoch label overlapped the legend in the CLIP early-fusion figure; all curve figures were regenerated (numbers unchanged).
- 410 picks were made mechanically (fixed random seed 510, without looking at the memes): two where fusion was right and text-only was wrong (preferring memes where both fusion models were right), and one where early fusion was wrong while both single-modality models were right. Claude did not look at or describe the memes; the "410: meme <id>" cells are left empty for me.
- Checks: no notebook errors; the test split stayed locked.

**Results (dev, mean ± std over seeds 0/1/2):**

| model | AUROC | F1 |
|---|---|---|
| always_not_hateful | 0.500 | 0.000 |
| text_bert | 0.627 ± 0.007 | 0.485 ± 0.027 |
| image_resnet | 0.537 ± 0.006 | 0.194 ± 0.125 |
| early_concat | 0.664 ± 0.009 | 0.416 ± 0.025 |
| late_avg | 0.627 ± 0.011 | 0.352 ± 0.063 |
| clip_text | 0.645 ± 0.005 | 0.383 ± 0.003 |
| clip_image | 0.679 ± 0.003 | 0.534 ± 0.031 |
| clip_early_concat | 0.749 ± 0.005 | 0.591 ± 0.009 |
| clip_late_avg | 0.695 ± 0.005 | 0.486 ± 0.019 |

- Chosen configs: clip_text hidden 64, lr 3e-4, dropout 0.3, wd 1e-4, no pos_weight (best_epoch 5, 6, 5); clip_image hidden 512, lr 1e-4, dropout 0, wd 0, pos_weight 1.82 (best_epoch 33, 18, 69); clip_early_concat hidden 256, lr 1e-3, dropout 0, wd 1e-4, pos_weight 1.82 (best_epoch 62, 27, 90).
- clip_early_concat vs early_concat: +0.085 AUROC (0.749 vs 0.664); larger seed std 0.009, so the gap is about 9.6× the seed std.
- 410: 139 dev memes where a fusion model was right and a single-modality model wrong (69 with text wrong, 89 with image wrong); 202 where early_concat was wrong (7 of them with both single models right). Picked ids: 50379, 01967 (fusion right), 57208 (fusion wrong).

**How I checked it:** _(to fill in)_
