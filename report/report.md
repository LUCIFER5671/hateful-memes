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

**Which modality is more useful depended on the encoder.** With the baseline encoders, text was clearly more useful than images: BERT reached 0.627 dev AUROC and ResNet-18 only 0.537, barely above chance. This matched my hypothesis that text would be the more useful modality. With CLIP the order reversed: the image model reached 0.679 and the text model 0.645, a gap of 0.034, several times the seed standard deviation (0.003 and 0.005). Switching encoders barely changed the text model (+0.018 AUROC, BERT → CLIP) but greatly improved the image model (+0.142, ResNet-18 → CLIP). So my hypothesis was not wrong about memes in general. The image side was held back by the encoder, not by the modality.

I think the reason is what each image encoder was trained to do. ResNet-18 was trained to classify ImageNet photos into 1,000 object categories, so its features describe which objects appear (a dog, a car, a person). The hateful meaning of a meme rarely depends on that: the same kind of stock photo or cartoon appears in both hateful and harmless memes, and the meaning depends on who is shown and how the image relates to the caption. The ResNet image model's dev AUROC even fell below 0.5 as training continued, which suggests that its features did not carry information that generalizes to dev. CLIP was trained on image–caption pairs from the web, which is much closer to how memes are made, so its image features capture the kind of concepts people describe in words. The lesson is that "is text or image more useful?" cannot be answered separately from the choice of encoder. A weak image encoder can make a modality look useless when it is not.

## ARI 510: Hypothesis and Design Change

## 5. Reflection

## 6. Generative AI Use Statement

| Phase | Used AI? | How (mode + what you asked) | How you checked / corrected / overrode it |
|---|---|---|---|
| Planning & understanding | | | |
| Coding and Debugging | | | |
| Analysis & interpretation | | | |
| Report writing, formatting, editing | Yes | Advice / feedback: I had Claude and Grammarly polish Section 1 (Looking at the Data) from my own notes. | I verified every number against `results/human_labels.csv` and the notebook output. |
