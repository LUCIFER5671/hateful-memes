# ARI 510 Lab 3: Multimodal Classification with Pretrained Encoders

## 1. Looking at the Data

**Label distribution.** The training set has 8,500 memes, of which 35.5% are hateful. The dev (dev_seen) and test (test_seen) sets are much more balanced, at 49.4% of 500 and 49.0% of 1,000 respectively (Figure 1). This shift matters: a model trained on the training set learns that most memes are not hateful, so at a 0.5 threshold it will lean toward predicting "not hateful" on dev and test, which lowers recall. It also means accuracy alone is misleading, and I rely on AUROC as the primary metric because it does not depend on the threshold or the class balance.

![Figure 1: Label distribution per split (note the different y-scales).](../figures/label_distribution.png)

*Figure 1: Label distribution per split: number of not-hateful and hateful memes, with % hateful (note the different y-scales).*

**Looking at examples.** I looked through 12 training memes (6 hateful, 6 not). The images were mostly cartoons, illustrations, and stock-style pictures, though some used real photos (e.g. id 06435) such as a child, a well-known film character, or animals, which were harmless on their own. The hateful memes typically ended in a punchline that carried the attack, mostly targeting Muslims and Black people, with one anti-feminist example (id 80945). The non-hateful memes were ordinary memes without that punchline structure. In most hateful examples, the image alone was neutral and the meaning only became hateful once paired with the text (e.g. id 58609). I agreed with every label in this sample, and none of the images were blurry or confusing.

**Being the model.** I labeled 20 randomly chosen dev memes without seeing their labels and reached 65% accuracy (13/20; precision 0.44, recall 0.67, F1 0.53). For each meme I also recorded which modality I needed: both image and text for 9, the image alone for 5, and the text alone for 6. I was right on 5 of 6 text-only and 4 of 5 image-only memes, but on only 4 of the 9 memes that needed both (44%), which is about chance. These are the confounder-style memes the dataset was designed around.

My errors fell into two groups. Five of the seven were false alarms: I reacted to a sensitive topic (Jewish people, the number six million, Muslims, Lebanon, drugs) rather than to an actual attack on a group. One (61085) I labeled hateful because it seemed to promote cocaine, which isn't hate under the dataset's definition. My two misses (89425, 23810) were the opposite problem: I could not understand the image or the reference in the text, which I believe depends on American cultural context I am not familiar with. One caveat is that my sample was only 30% hateful compared with 49% in the full dev set, and with 20 memes each error moves accuracy by 5 points, so 65% is a rough baseline. Even so, the task was clearly hard for me as a human, and it needs both world knowledge and careful reading of how the image and text combine.

## 2. Approach

## 3. Results

## 4. Discussion

## ARI 510: Hypothesis and Design Change

## 5. Reflection

## 6. Generative AI Use Statement

| Phase | Used AI? | How (mode + what you asked) | How you checked / corrected / overrode it |
|---|---|---|---|
| Planning & understanding | | | |
| Coding and Debugging | | | |
| Analysis & interpretation | | | |
| Report writing, formatting, editing | Yes | Advice / feedback: I had Claude and Grammarly polish Section 1 (Looking at the Data) from my own notes. | I verified every number against `results/human_labels.csv` and the notebook output. |
