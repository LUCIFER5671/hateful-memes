# ARI 510 Hypothesis (written before Part 3)

## Evidence
In the dev set, 78 of the 79 groups of memes that share identical text contain both a hateful and a non-hateful example. Text alone cannot separate them: any text-only model must get at least one meme wrong in each of those groups, which puts a hard ceiling on text-only performance. My own labeling points the same way: 9 of my 20 dev memes (45%) needed both the image and the text to judge, and I got only 4 of those 9 right.

## Hypothesis
1. **Single modality:** I expect text to be more useful on its own than images. When labeling, I found the text much easier to judge as hateful or not; images were often needed to confirm the meaning, but on their own they rarely decided it.
2. **Fusion:** I expect fusion to beat the best single-modality model, but only modestly, by roughly 2–5 AUROC points. The hateful meaning often lies in the combination of text and image, but BERT and ResNet were never trained on memes, and a small classifier on their frozen vectors can only learn simple interactions between them.
3. **Early vs. late fusion:** I expect early fusion to work better. It sees the text and image features together in one model, so it can in principle learn how they interact. Late fusion trains the two classifiers separately and only averages their verdicts, so each one still judges from a single modality and neither can catch a meme whose hatefulness exists only in the combination.
