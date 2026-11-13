# ARI 510 Lab 3: Multimodal Classification with Pretrained Encoders

## 1. Looking at the Data

**Label distribution.** The training set has 8,500 memes, of which 35.5% are hateful. The dev (dev_seen) and test (test_seen) sets are much more balanced, at 49.4% of 500 and 49.0% of 1,000 respectively (Figure 1). This shift matters: a model trained on the training set learns that most memes are not hateful, so at a 0.5 threshold it will lean toward predicting "not hateful" on dev and test, which lowers recall. It also means accuracy alone is misleading, and I rely on AUROC as the primary metric because it does not depend on the threshold or the class balance.

![Figure 1: Label distribution per split (note the different y-scales).](../figures/label_distribution.png)

*Figure 1: Label distribution per split: number of not-hateful and hateful memes, with % hateful (note the different y-scales).*

**Looking at examples.** I looked through 12 training memes (6 hateful, 6 not). The images were mostly cartoons, illustrations, and stock-style pictures, though some used real photos (e.g. id 06435) such as a child, a well-known film character, or animals, which were harmless on their own. The hateful memes typically ended in a punchline that carried the attack, mostly targeting Muslims and Black people, with one anti-feminist example (id 80945). The non-hateful memes were ordinary memes without that punchline structure. In most hateful examples, the image alone was neutral and the meaning only became hateful once paired with the text (e.g. id 58609). I agreed with every label in this sample, and none of the images were blurry or confusing.

**Being the model.** I labeled 20 randomly chosen dev memes without seeing their labels and reached 65% accuracy (13/20; precision 0.44, recall 0.67, F1 0.53). For each meme I also recorded which modality I needed: both image and text for 9, the image alone for 5, and the text alone for 6. I was right on 5 of 6 text-only and 4 of 5 image-only memes, but on only 4 of the 9 memes that needed both (44%), which is about chance. These are the confounder-style memes the dataset was designed around.

My errors fell into two groups. Five of the seven were false alarms: I reacted to a sensitive topic (Jewish people, the number six million, Muslims, Lebanon, drugs) rather than to an actual attack on a group. One (61085) I labeled hateful because it seemed to promote cocaine, which isn't hate under the dataset's definition. My two misses (89425, 23810) were the opposite problem: I could not understand the image or the reference in the text, which I believe depends on American cultural context I am not familiar with. One caveat is that my sample was only 30% hateful compared with 49% in the full dev set, and with 20 memes each error moves accuracy by 5 points, so 65% is a rough baseline. Even so, the task was clearly hard for me as a human, and it needs both world knowledge and careful reading of how the image and text combine.

## 2. Approach

**Encoders.** All encoders were used frozen (eval mode, no gradients) and embeddings were computed once on an RTX 5060 Laptop GPU and cached to disk (~5 minutes total). Text was encoded with bert-base-uncased, keeping the [CLS] vector from the last layer (768-d, max length 128). Images were encoded with ResNet-18 (ImageNet weights) with its final classification layer replaced by an identity, giving a 512-d vector per image. For the 510 design change I also encoded both modalities with CLIP ViT-B/32 (OpenAI weights, 512-d each). CLIP's first run warned that these weights expect the QuickGELU activation, which open_clip's default configuration does not use; I forced QuickGELU and recomputed, then checked that CLIP matches each meme's image to its own text 62–65% of the time among 200 candidates (0.5% by chance). Re-embedding a sample reproduced the cached vectors to within 2.5e-5.

**Classifier and training.** Every model uses the same small feedforward network (Linear → ReLU → [dropout] → Linear to one logit) trained with BCEWithLogitsLoss, Adam, and mini-batches of 64, through a single shared training function. Features are standardized using statistics from the training set only. All choices were made on the dev set; the test set was not loaded until the final comparison.

**Hyperparameter selection.** For each model I searched hidden size {64, 256, 512}, learning rate {1e-3, 3e-4, 1e-4, 3e-5}, dropout {0, 0.3}, weight decay {0, 1e-4}, and pos_weight {none, n_neg/n_pos}, for up to 100 epochs, selecting by dev AUROC. The chosen configuration was retrained with 3 seeds and I report mean ± std. Chosen text-only config: hidden 64, lr 1e-3, dropout 0, weight decay 0, pos_weight ≈ 1.82 (n_neg/n_pos); best epoch 2 for all 3 seeds.

**Sanity checks (text model).** An always-"not hateful" baseline scores 0.506 accuracy and 0 F1 on dev. A freshly initialized network had a starting loss of 0.659, close to ln 2 ≈ 0.693. Trained on 20 examples, it reached 5e-5 training loss and 100% training accuracy, confirming the training loop works.

**Changes from the instructions.** My first runs stopped training at the epoch with the lowest dev loss. For all three seeds this was epoch 1–2, and the untrained network often had a lower dev loss than any trained epoch, even though dev AUROC rose from about 0.5 to 0.62. Dev loss was rising because the model overfits quickly and is miscalibrated for dev's 49% hateful rate after training on 35%. Since I select models by AUROC, I switched early stopping to the epoch with the highest dev AUROC (Figure 2). I also tried slower learning rates (down to 3e-5); they delayed the AUROC peak (to about epoch 19) but did not raise it, and in 68 of 96 runs the untrained model had the lowest dev loss. All 96 configurations fell between 0.620 and 0.635 AUROC, within seed noise. I also labeled 20 dev memes for the human baseline (the PDF says 10 in Part 0 and 20 in the checklist).

![Figure 2: Text-only (BERT) training curves.](../figures/curves_text_bert.png)

*Figure 2: Text-only (BERT [CLS]) model, chosen configuration, seed 0. (a) Train and dev loss per epoch (epoch 0 = untrained). (b) Dev AUROC per epoch. The dashed line marks best_epoch (highest dev AUROC).*

## 3. Results

All numbers are mean ± std over three training seeds (0, 1, 2) at the fixed 0.5 threshold; the always-"not hateful" baseline is a single deterministic run. Every configuration, epoch and threshold was chosen on dev (Table 1) before the test set was evaluated once (Table 2).

*Table 1: Dev (dev_seen, 500 memes) results.*

| model | accuracy | precision | recall | F1 | AUROC |
|---|---|---|---|---|---|
| always_not_hateful | 0.506 | 0.000 | 0.000 | 0.000 | 0.500 |
| text_bert | 0.574 ± 0.012 | 0.602 ± 0.016 | 0.408 ± 0.035 | 0.485 ± 0.027 | 0.627 ± 0.007 |
| image_resnet | 0.515 ± 0.001 | 0.569 ± 0.065 | 0.128 ± 0.098 | 0.194 ± 0.125 | 0.537 ± 0.006 |
| early_concat | 0.580 ± 0.014 | 0.663 ± 0.026 | 0.304 ± 0.021 | 0.416 ± 0.025 | 0.664 ± 0.009 |
| late_avg | 0.557 ± 0.010 | 0.639 ± 0.040 | 0.247 ± 0.067 | 0.352 ± 0.063 | 0.627 ± 0.011 |
| clip_text | 0.571 ± 0.006 | 0.661 ± 0.019 | 0.270 ± 0.002 | 0.383 ± 0.003 | 0.645 ± 0.005 |
| clip_image | 0.613 ± 0.005 | 0.660 ± 0.015 | 0.451 ± 0.052 | 0.534 ± 0.031 | 0.679 ± 0.003 |
| **clip_early_concat (510 design change)** | 0.671 ± 0.005 | 0.769 ± 0.033 | 0.480 ± 0.024 | 0.591 ± 0.009 | **0.749 ± 0.005** |
| clip_late_avg | 0.611 ± 0.012 | 0.700 ± 0.018 | 0.372 ± 0.018 | 0.486 ± 0.019 | 0.695 ± 0.005 |

![Figure 3: Dev AUROC by model.](../figures/dev_auroc_comparison.png)

*Figure 3: Dev AUROC by model (mean ± std over 3 seeds). Blue: BERT / ResNet-18 encoders; purple: CLIP ViT-B/32 encoders; grey: always-"not hateful" baseline.*

*Table 2: Test (test_seen, 1,000 memes) results, evaluated once.*

| model | accuracy | precision | recall | F1 | AUROC |
|---|---|---|---|---|---|
| always_not_hateful | 0.510 | 0.000 | 0.000 | 0.000 | 0.500 |
| text_bert | 0.616 ± 0.003 | 0.649 ± 0.025 | 0.475 ± 0.052 | 0.547 ± 0.026 | 0.676 ± 0.009 |
| image_resnet | 0.516 ± 0.002 | 0.533 ± 0.019 | 0.131 ± 0.102 | 0.196 ± 0.126 | 0.544 ± 0.005 |
| early_concat | 0.611 ± 0.006 | 0.692 ± 0.009 | 0.373 ± 0.025 | 0.485 ± 0.020 | 0.701 ± 0.005 |
| late_avg | 0.583 ± 0.014 | 0.674 ± 0.025 | 0.296 ± 0.085 | 0.406 ± 0.074 | 0.676 ± 0.010 |
| clip_text | 0.592 ± 0.007 | 0.667 ± 0.014 | 0.333 ± 0.014 | 0.444 ± 0.013 | 0.682 ± 0.002 |
| clip_image | 0.612 ± 0.007 | 0.665 ± 0.024 | 0.424 ± 0.036 | 0.516 ± 0.021 | 0.673 ± 0.005 |
| **clip_early_concat (510 design change)** | 0.681 ± 0.009 | 0.742 ± 0.016 | 0.536 ± 0.028 | 0.622 ± 0.017 | **0.774 ± 0.009** |
| clip_late_avg | 0.611 ± 0.004 | 0.702 ± 0.008 | 0.359 ± 0.007 | 0.475 ± 0.008 | 0.711 ± 0.004 |

![Figure 4: Test AUROC by model.](../figures/test_auroc_comparison.png)

*Figure 4: Test AUROC by model (mean ± std over 3 seeds), same models and colours as Figure 3.*

## 4. Discussion

**Text vs. images, and how the answer depended on the encoder.** With the baseline encoders, text was clearly more useful than images. BERT reached 0.676 test AUROC (0.627 dev), while ResNet-18 reached only 0.544 (0.537 dev), barely above the 0.500 baseline. Switching to CLIP barely changed the text model (+0.006 test, +0.018 dev) but transformed the image model (+0.129 test, +0.142 dev): CLIP image reached 0.673 on test. On dev the CLIP image model even beat the CLIP text model (0.679 vs 0.645), the opposite of my hypothesis that text is more useful. On test the two were roughly level (text 0.682, image 0.673), so the robust finding is not that images beat text, but that the image modality went from near chance to on par with text once the encoder changed. My hypothesis was therefore really a statement about ResNet, not about images. ResNet-18 was trained to sort ImageNet photos into 1,000 object categories, so its features describe which objects appear (a dog, a car, a person). Whether a meme is hateful rarely depends on that: the same kind of stock photo or cartoon appears in both hateful and harmless memes, and the ResNet model's dev AUROC even fell below 0.5 as training continued. CLIP was trained on hundreds of millions of image–caption pairs from the web, which is much closer to how memes are made, so its image features carry the kind of concepts people describe in words. A weak encoder can make a modality look useless when it is not.

**Did fusion help, and why did early fusion beat late fusion?** Early fusion was the best model for both encoder pairs: on test it improved on the best single model by 0.025 AUROC with BERT + ResNet (0.701 vs 0.676) and by 0.092 with CLIP (0.774 vs 0.682). Late fusion helped much less: with BERT + ResNet it only matched text-only (0.676), because averaging in a near-chance image model dilutes the text model's predictions; with CLIP it improved by 0.029 (0.711), but early fusion was 0.063 higher still. This matches my hypothesis, and the reason follows from how the dataset was built. Hateful Memes contains "confounders": the same text paired with a different image, or the same image with a different text, flips the label (in dev, 78 of the 79 groups of memes with identical text contain both labels). In late fusion, the text and image classifiers are trained separately and each judges the meme from one modality alone; averaging two such verdicts can never recover meaning that exists only in the combination, because each classifier gives identical inputs identical scores. Early fusion trains one classifier on the concatenated text and image features, so it can learn how the two interact. The interaction is not always learned correctly: dev meme 57208 is harmless in both modalities, and both single models got it right, but early fusion read the combination as an attack on the person shown and predicted hateful.

**Comparison with the original paper.** Kiela et al. (2020; NeurIPS version, arXiv 2005.04790v3, Table 1) report AUROC on a 500-meme dev ("validation") set and a 1,000-meme test set; their test numbers are the mean ± std over three seeds. All paper numbers below are from that version.

| model | paper dev | ours dev | paper test | ours test |
|---|---|---|---|---|
| Image-only (paper: ResNet-152 grid; ours: ResNet-18) | 52.33 | 53.7 | 53.71 ± 2.04 | 54.4 ± 0.5 |
| Text BERT | 65.05 | 62.7 | 69.00 ± 0.11 | 67.6 ± 0.9 |
| Late fusion | 65.07 | 62.7 | 69.30 ± 0.33 | 67.6 ± 1.0 |
| Concat BERT (early fusion) | 65.88 | 66.4 | 67.77 ± 0.87 | 70.1 ± 0.5 |
| Best paper model (Visual BERT COCO) | 74.14 | — | 75.44 ± 1.86 | — |
| CLIP early fusion (our 510 design change) | — | 74.9 | — | 77.4 ± 0.9 |

Our baselines land within about 2.5 AUROC points of the paper's in every case, sometimes above and sometimes below: text-only is 2.4 points lower on dev and 1.4 lower on test, while early fusion is 0.5 higher on dev and 2.3 higher on test. I read these gaps as noise rather than as one setup being better. With about 250 hateful memes in dev and 490 in test, the sampling error of an AUROC around 0.65 is roughly ±2.4 points on dev and ±1.7 points on test (Hanley–McNeil standard error), so a 2-point gap is about one standard error. That is much larger than our seed-to-seed std (about 0.5–1 point), so which memes happen to be in a split matters more than the training seed. The splits themselves also move the numbers: our text model rose about 5 points from dev to test (62.7 → 67.6), and the paper's Text BERT rose about 4 (65.05 → 69.00). Our setup also differs from the paper's: the paper fine-tuned all models fully end-to-end, BERT included ("We fine-tune all models fully end-to-end without freezing anything", Appendix A), on 16 V100 GPUs, while our encoders are frozen and only a small classifier on top is trained, and our image encoder is smaller (ResNet-18 vs. ResNet-152). I would therefore expect our baselines to be slightly weaker, and they land within about one standard error of the paper either way. The overall pattern matches: images alone are near chance and text alone is around 65–69. One difference is that the paper's Concat BERT scored below its Text BERT on test, while our early fusion beat text-only on both dev and test.

The CLIP design change is the clearest result: frozen CLIP features with a small classifier scored 74.9 on dev and 77.4 on test, beating every model in the paper on both splits, including its best, Visual BERT COCO (74.14 dev, 75.44 test), a multimodally pretrained transformer fine-tuned end-to-end. Our model achieves this without fine-tuning any encoder weights. CLIP was released after the paper and pretrained on far more image–text data. Still, it shows how much the choice of encoder matters on this task: the same small classifier went from 70.1 (BERT + ResNet) to 77.4 (CLIP) on test.

**Comparison with the dumb baseline and with a human.** Every trained model beat the always-"not hateful" baseline (0.510 test accuracy, 0 F1, 0.500 AUROC), though image-only ResNet only barely (0.516 accuracy, 0.544 AUROC). My own blind labeling of 20 dev memes gave 65% accuracy (13/20). The best models are around that level on test accuracy (text-only BERT 61.6%, CLIP early fusion 68.1%), but my sample was small (each meme is 5 points) and the paper's trained annotators reached 84.7% on the test set. So even the best model here is well below a careful human, and the memes I found hardest, the ones that needed both modalities, are exactly the cases the confounders were built around. That a small classifier on frozen BERT vectors, never trained on memes or hate speech, reaches 0.676 AUROC shows that general-purpose text features already carry a lot of the signal; the near-chance ResNet result shows that this is not automatic and depends on what the encoder was trained to notice.

**Which metrics I relied on, and why AUROC.** I used AUROC (area under the ROC curve) as the primary metric, both for model selection and for comparison. AUROC measures how well a model ranks hateful memes above non-hateful ones across all possible thresholds, so it does not depend on the 0.5 decision threshold or on the class balance. That matters here because training is 35.5% hateful but dev and test are about 49%: models trained on the skewed training set lean toward "not hateful" at 0.5, which lowers recall and F1 without the model ranking memes any worse. For example, BERT + ResNet early fusion has a higher test AUROC than text-only (0.701 vs 0.676) but a lower F1 (0.485 vs 0.547), because its recall at 0.5 is lower (0.373 vs 0.475). Accuracy is also misleading on its own, since the always-"not hateful" baseline already gets 51%. The paper recommends AUROC as the main metric for the same reasons, with accuracy as a secondary, easier-to-read number. Finally, test AUROC came out higher than dev AUROC for most models (by more than 0.03 for text-only, early fusion, late fusion and CLIP text), but the main conclusions held on both splits: CLIP early fusion was the best model, early fusion beat late fusion and the best single model for both encoder pairs, and ResNet image-only stayed near chance. This gives me some confidence that the conclusions above are not an artifact of the dev set.

## ARI 510: Hypothesis and Design Change

### Hypothesis (written before Part 3)

I wrote this hypothesis after Part 0 and the text-only model, before training any image-only or fusion model, and committed it to the repository at that point (`notes/hypothesis_510.md`).

*Evidence.* In the dev set, 78 of the 79 groups of memes that share identical text contain both a hateful and a non-hateful example. Text alone cannot separate them: any text-only model must get at least one meme wrong in each of those groups, which puts a hard ceiling on text-only performance. My own labeling points the same way: 9 of my 20 dev memes (45%) needed both the image and the text to judge, and I got only 4 of those 9 right.

1. **Single modality:** I expect text to be more useful on its own than images. When labeling, I found the text much easier to judge as hateful or not; images were often needed to confirm the meaning, but on their own they rarely decided it.
2. **Fusion:** I expect fusion to beat the best single-modality model, but only modestly, by roughly 2–5 AUROC points. The hateful meaning often lies in the combination of text and image, but BERT and ResNet were never trained on memes, and a small classifier on their frozen vectors can only learn simple interactions between them.
3. **Early vs. late fusion:** I expect early fusion to work better. It sees the text and image features together in one model, so it can in principle learn how they interact. Late fusion trains the two classifiers separately and only averages their verdicts, so each one still judges from a single modality and neither can catch a meme whose hatefulness exists only in the combination.

### Design change: CLIP encoders for both modalities

**Change.** I replaced BERT (text) and ResNet-18 (images) with CLIP ViT-B/32 (OpenAI weights), for both modalities, giving a 512-d text vector and a 512-d image vector per meme.

**Justification.** CLIP was trained on 400 million image–text pairs from the web, so its image and text vectors live in one shared space, and it has seen web images that contain text, like memes. ResNet-18 was trained only to recognize ImageNet object categories, and BERT only on books and Wikipedia. After the text-only and image-only results, the image encoder looked like the weakest link (ResNet image-only was barely above chance), and a fusion model can only combine what its encoders capture. So rather than changing how the two vectors are combined, I changed what the vectors contain, and kept everything else fixed so that any difference comes from the encoder.

**Setup.** Everything except the encoder is identical to the baseline: the same training function, the same 96-configuration grid, selection and early stopping by dev AUROC, and three seeds. I trained four CLIP models that mirror the baseline ones: text-only (`clip_text`), image-only (`clip_image`), early fusion on the concatenated 1,024-d vector (`clip_early_concat`, the design change), and late fusion averaging the seed-paired text and image probabilities (`clip_late_avg`). The chosen early-fusion configuration was hidden size 256, learning rate 1e-3, no dropout, weight decay 1e-4 and pos_weight ≈ 1.82 (best epochs 62, 27 and 90 for the three seeds).

### Results of the design change

AUROC, mean ± std over 3 seeds (all metrics are in Tables 1 and 2):

| model | BERT + ResNet, dev | CLIP, dev | BERT + ResNet, test | CLIP, test |
|---|---|---|---|---|
| text-only | 0.627 ± 0.007 | 0.645 ± 0.005 | 0.676 ± 0.009 | 0.682 ± 0.002 |
| image-only | 0.537 ± 0.006 | 0.679 ± 0.003 | 0.544 ± 0.005 | 0.673 ± 0.005 |
| early fusion | 0.664 ± 0.009 | **0.749 ± 0.005** | 0.701 ± 0.005 | **0.774 ± 0.009** |
| late fusion | 0.627 ± 0.011 | 0.695 ± 0.005 | 0.676 ± 0.010 | 0.711 ± 0.004 |

On dev, CLIP early fusion improved on the baseline early fusion by 0.085 AUROC (0.749 vs. 0.664), about ten times the larger seed std (0.009), and it was the best model on dev. It was then evaluated once on test, where it remained the best model: 0.774 vs. 0.701 for the baseline early fusion (+0.073), with the highest accuracy (0.681) and F1 (0.622) of all models as well. CLIP also improved late fusion (+0.068 on dev, +0.035 on test), but much less than early fusion.

### Hypothesis revisited

1. **Text more useful than images: held only for BERT/ResNet.** With the baseline encoders text clearly won (test 0.676 vs. 0.544). With CLIP the image model caught up: it beat CLIP text on dev (0.679 vs. 0.645) and was roughly level on test (0.673 vs. 0.682). My prediction was really about ResNet, not about images (see Section 4).
2. **Fusion helps by 2–5 points: held for BERT/ResNet early fusion, exceeded with CLIP.** Baseline early fusion beat the best single model by 3.7 points on dev and 2.5 on test, inside my range. With CLIP the gain was larger than I expected: 7.0 points on dev and 9.2 on test. Late fusion did not reach my range with BERT/ResNet (no gain on either split), because averaging in a near-chance image model adds nothing.
3. **Early fusion beats late fusion: held for both encoder pairs, on dev and on test** (BERT/ResNet: 0.664 vs. 0.627 dev, 0.701 vs. 0.676 test; CLIP: 0.749 vs. 0.695 dev, 0.774 vs. 0.711 test). This supports my reasoning that a classifier seeing both feature vectors together can learn how text and image combine, which averaging two separate verdicts cannot, though, as meme 57208 shows, this can also produce a hateful reading that isn't there.

## 5. Reflection

## 6. Generative AI Use Statement

| Phase | Used AI? | How (mode + what you asked) | How you checked / corrected / overrode it |
|---|---|---|---|
| Planning & understanding | | | |
| Coding and Debugging | | | |
| Analysis & interpretation | | | |
| Report writing, formatting, editing | Yes | Advice / feedback: I had Claude and Grammarly polish Section 1 (Looking at the Data) from my own notes. | I verified every number against `results/human_labels.csv` and the notebook output. |
