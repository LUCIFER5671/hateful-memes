# Part 0 Notes

## Patterns in text
- Hateful memes usually end in a punchline that carries the attack; non-hateful memes mostly don't have a punchline structure at all (e.g. id 65871).
- The groups targeted in the hateful examples were mostly Muslims and Black people; one (id 80945) was anti-feminist.

## Patterns in images
- A mix of cartoons, illustrations, and stock-style images, plus real photos (e.g. id 06435).
- Some hateful memes used ordinary real photos (a child, a well-known film character, animals) that are harmless on their own; the attack comes from the text placed over them.
- The non-hateful ones were ordinary memes without a punchline.

## What hateful ones share
- Most needed both modalities: the image on its own was usually neutral, and it was the text that made the meaning hateful once paired with that image (e.g. id 58609).

## Messy/surprising
- In my 12-meme sample, I agreed with every label and none of the images were blurry or confusing.

## Why I got 7 wrong (human baseline, 13/20)

**Missed hateful memes (89425, 23810):** I didn't understand either one. The images made no sense to me, and I couldn't work out what the text was referring to. I believe both depend on cultural or regional references, likely American, that I'm not familiar with coming from India. Recognizing hate in these memes requires background knowledge, not just reading the image and text.

**False alarms (5 memes I labeled hateful that weren't):**
- **73021:** The meme was about Jewish people, so I assumed it was antisemitic. The topic alone isn't an attack.
- **37160:** The number 6 million appeared, which I linked to the Holocaust and read as a hateful reference. It wasn't hateful in context.
- **61085:** I read it as promoting cocaine and labeled it hateful. In hindsight, drug content isn't hate under the dataset's definition, since it doesn't attack a group of people.
- **41728:** Very deceptive. Even with both the image and text, it read to me as calling for Muslims to be exterminated, but it was labeled not hateful.
- **37420:** I read it as promoting terrorism against Lebanon and as hate against the country.

**Pattern:** Most of my false alarms came from reacting to a sensitive *topic* (Jewish people, the Holocaust, Muslims, Lebanon, drugs) rather than to an actual *attack* on a group. My two misses were the opposite problem: the hate depended on cultural context I didn't have. This suggests the task needs both world knowledge and careful reading of how the text and image combine, which a model trained on frozen BERT and ResNet vectors also won't have.
