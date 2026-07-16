# v3 full quality samples

These are unedited outputs from the 20-control seeded random judge subset of the
100 TF1 controls unseen during v2 training. Selection below is transparent: the
lowest, median, and highest v3-full overall scores under judge prompt
`v1-original`. Scores are historical and must not be compared directly with the
stricter `v2-strict` prompt without rejudging.

Strict-prompt calibration on these same three controls:

| Source index | `v1-original` overall | `v2-strict` overall |
|---:|---:|---:|
| 200011 | 4.75 | 2.25 |
| 200071 | 6.50 | 3.50 |
| 200077 | 9.00 | 4.75 |

This three-story smoke test confirms that the stricter rubric no longer treats
literal character/moral inclusion as narrative success. It is not a replacement
for a full paired rejudgment.

## Lowest: 4.75/10

- Source: `klusai/ds-tf1-en-3m:valid-index-200011`
- Character: `a energetic crab`
- Moral: `modesty preserves dignity`
- Checks: exact character, exact moral, clean ending
- Scores: grammar 4, creativity 3, moral clarity 2, prompt adherence 10

In the bustling harbor town of Harborville, where seagulls soared and fishermen
sailed, lived a Energetic Crab who loved to explore. He'd scuttle across the
sandy beach, his shiny shell glinting in the sunlight, as he explored.

One day, while scuttling along the dock, the Energetic Crab's best friend, the
Wise Fish, was busy repairing her net. But when she came back, it turned out that
the Patient Fish had eaten all the fish from their nets. The Energetic Crab,
feeling angry and hurt, shouted, "You ate my fish! Now I have nothing!"

The other sea creatures looked at each other, confused. They thought they were
right; someone else could blame them. So, they decided to vote on what to do with
the Gentle Fish. Some said, "We should punish her." Others said, "No, we must say
so."

The crowd voted, and soon everyone wanted an equal share of food. A wise old
Turtle suggested sharing some seaweed among everyone, but only because everyone
agreed to share. The Energetic Crab watched as friends shared equally, not
blaming others for more.

As they ate, the Energetic Crab realized that being kind didn't make him weak.
He apologized to his friend and promised to be kinder. From then on, whenever
anyone accused someone else, they'd say, "Remember, you're not the strongest or
wisest – we're honest!" And that's why it's always best to speak up.

Moral: modesty preserves dignity

## Median: 6.5/10

- Source: `klusai/ds-tf1-en-3m:valid-index-200071`
- Character: `a cautious frog`
- Moral: `perspective can change the outcome`
- Checks: exact character, exact moral, clean ending
- Scores: grammar 9, creativity 4, moral clarity 3, prompt adherence 10

In a sunny bamboo grove, where tall green stalks swayed gently in the breeze,
lived a cautious frog. He loved to hop around and explore every nook, discovering
new hiding spots. One day, while hopping near the grove's edge, he spotted a
shiny object buried beneath a leaf.

The frog's eyes grew wide with excitement. "Oh, what a treasure!" he croaked. "I
must have it!" The other frogs whispered among themselves, but none of them knew
who might be stealing their food.

The next morning, the frog, feeling proud of his find, decided to take matters
into his own hands. But as he reached for it, it knocked over a nearby basket,
spilling juicy berries everywhere. A hungry rabbit, known for her kindness,
caught him in mid-air.

"Why did you do that?" asked the rabbit.

"I wanted that treasure," said the frog, looking at his reflection in a puddle.
"You're so lucky I couldn't find it."

The rabbit's face turned red, then smiled. "You don't need all those berries.
Let's share them."

The frog thought about this. Then, they sat down together on a leaf. "Let's divide
them," she said. "We can pick some berries, and I'll help you."

The rabbit nodded. "That sounds like fun!" And so, they shared the berries
together, laughing and chatting.

As they ate, the sun began to set, casting a warm glow over the grove. From that
day on, the animals worked side by side, proving that sharing and kindness can
lead to lasting friendships.

Moral: perspective can change the outcome.**

Moral: perspective can change the outcome

## Highest: 9.0/10

- Source: `klusai/ds-tf1-en-3m:valid-index-200077`
- Character: `a timid badger`
- Moral: `teamwork amplifies success`
- Checks: exact character, exact moral, clean ending
- Scores: grammar 10, creativity 7, moral clarity 9, prompt adherence 10

In the ancient temple ruins, hidden behind vines and moss, lived a timid badger.
The badger's home was cozy, filled with soft furs of moss and twinkling lights.
Every day, she would watch as her friends gathered to play among the crumbling
stones.

But one badger, who loved shiny things, started to take more than their fair
share. "I need it more," she'd say, or "I'm starving!" Her friends, who worked
hard in their own gardens, were sad and left.

The temple's wise old owl, perched on a stone pedestal, noticed something
strange. "Why do you keep taking all your treasures? Don't you think it's time
for everyone else?"

The shy badger looked down, ashamed. She knew she had been selfish and didn't
want anyone to be unhappy. So, instead of giving up, she decided to share her
treasures – an idea. "I'll help my friends gather new treasures, but we can work
together!"

Together, they collected beautiful stones. As they worked, the other animals
joined in, and soon the ruins were filled with laughter and chatter. They shared
stories, helped each other, and made sure everyone's needs were met.

When the sun set, the quiet badger smiled. "We did it! We're stronger together!"
The others cheered back, knowing that when they stood together, they could
accomplish anything. From then on, the timid badger's heart felt lighter, and she
became known as the bravest and most generous badger.

The moral is clear: Teamwork makes us strong, just like the leaves that grow into
our hearts.

Moral: teamwork amplifies success
