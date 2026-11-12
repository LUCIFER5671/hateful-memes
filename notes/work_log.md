# Work log — ARI 510 Lab 3

## 2026-11-07

**What I tried:**

**What broke:**

**What I figured out:**

## 2026-11-12

**What I tried:**
- Computed and cached all embeddings on my RTX 5060 laptop GPU: a 200-meme subset per split first, then everything. Full run: 317 s total (BERT 48 s, ResNet-18 110 s, CLIP ViT-B/32 144 s for train + dev + test). `embeddings/` is 94 MB.
- Did the blind "be the model" on 20 dev memes, marking for each whether I needed the image, the text, or both.

**What broke:**
- The first CLIP run printed `QuickGELU mismatch between final model config (quick_gelu=False) and pretrained tag 'openai' (quick_gelu=True)`. OpenAI's CLIP weights were trained with QuickGELU, but open_clip's plain "ViT-B-32" config uses standard GELU, so every CLIP vector would have been slightly wrong without any error. Fixed with `force_quick_gelu=True`, deleted the cached CLIP files and recomputed. Checked the fix: CLIP now matches each meme image to its own text out of 200 candidates 62–65% of the time (chance is 0.5%).

**What I figured out:**
- My human baseline is 65% accuracy (13/20): precision 0.444, recall 0.667, F1 0.533. 5 of my 7 mistakes were calling a meme hateful when it wasn't.
- The memes that needed both image and text were the hardest: 4/9 right, vs 4/5 for image-only and 5/6 for text-only judgments.
- Many memes share identical text: in dev, 78 of the 79 identical-text groups contain both a hateful and a non-hateful meme, so a text-only model has to miss some of them.
