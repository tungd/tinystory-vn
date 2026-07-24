# Human Evaluation

Scores use 1-5 where 1 is poor and 5 is excellent.

| Criterion | Meaning |
| --- | --- |
| English fluency | Grammar, wording, and naturalness. |
| Prompt adherence | Uses the requested character, setting, challenge, outcome, and teaching. |
| Fable structure | Has a clear setup, conflict, resolution, and concise fable style. |
| Moral clarity | Ends with or clearly states the intended moral. |
| Child safety | Wholesome and age-appropriate. |

## p01

### Prompt

```text
Character: a small turtle
Setting: a quiet pond
Challenge: the small turtle wants to cross a busy road
Outcome: his friends help the small turtle cross safely
Teaching: patience and teamwork matter
Length: short
```

### Model: Base+Repair

- Model ID: `base-llama32-3b-repair`
- Ollama: `llama3.2:3b-instruct-fp16`

#### Story

In a peaceful pond, lived a little turtle named Terry. One sunny day, he decided that adventure called him to explore beyond the water's edge. As he made his way closer to where some tall trees stood guard near busy Roady Way, Terry realized it was impossible for just one tiny creature like himself to cross.

Panicked and feeling all alone, Terry thought of a clever plan but didn't know how to execute it yet! Luckily, as soon as the little turtle confessed about wanting help from his pond friends (a wise old fish named Freddy), they quickly sprang into action!

Together with some frogs who climbed trees nearby for high ground view points and rabbits carrying sturdy vines on their backs that formed a rope bridge...the whole crew got to work making it possible! Under Terry's guidance, carefully constructed walkways allowed him the safe passage across Roady Way.

Moral: When friends unite towards one goal with patience in harmony - there is nothing big or small they can't accomplish together as best of buddies.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 4 |
| Prompt adherence | 4 |
| Fable structure | 4 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 4.4 |

Notes: Fluency is generally good, though phrases such as 'confessed about wanting help,' 'view points,' and the moral sound awkward.

---

### Model: Strict+Postprocess

- Model ID: `base-llama32-3b-strict-postprocess`
- Ollama: `llama3.2:3b`

#### Story

In a quiet pond surrounded by tall trees and greenery, lived a small turtle named Terry. He loved swimming in the clear water with his friends, chasing fish, and basking under warm sunbeams on large lily pads.

One sunny afternoon, as Terry was exploring near the edge of the pond, he noticed how busy the road that ran along its border had become. Cars whizzed by, making it impossible for him to cross safely alone. The small turtle longingly yearned to reach a nearby patch where juicy watercress grew in abundance.

As fate would have it, Terry's friends - Max the fish and Lily the frog - were swimming near his spot of longing when they observed their friend's predicament. Without hesitation, they joined forces with an old tortoise named Sam who had some years on him but was not too slow for speed. Together, these four animals devised a careful plan to help Terry reach safety beyond the road while keeping themselves safe from harm and traffic noise.

Moral: patience and teamwork matter.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 4 |
| Prompt adherence | 3 |
| Fable structure | 3 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 4.0 |

Notes: Smooth overall, but 'longingly yearned,' 'spot of longing,' and 'not too slow for speed' are unnatural; the crossing is only planned, not completed.

---

### Model: Fluency-SFT-v1

- Model ID: `fluency-sft-v1-lora-q4`
- Ollama: `llama32-fable-fluency-sft-v1:q4`

#### Story

In a peaceful pond, a little green turtle lived among tall water lilies. The sun shone brightly on its back as he slid into calm waters one day. But the next morning, a busy road stretched across his home - crossing became an adventure! A small white rabbit wanted to reach the other side but was too afraid of cars whizzing by. She asked her friends: "How can I cross? It's so scary!" The little green turtle and some brave fish offered their help. They came up with a plan, using big rocks as stepping stones over the road. As they worked together, another rabbit arrived - she was fast! Together, all three rabbits waited for each passing car to safely reach its turn on the other side of the pond before making a step forward. After many tries and good friends' cheers, everyone made it across without getting hurt. The little green turtle smiled at his new friends as they swam together in their safe home - where cars could never come near again! They all celebrated with juicy berries hidden among water lilies.

Moral:

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 2 |
| Prompt adherence | 2 |
| Fable structure | 3 |
| Moral clarity | 2 |
| Child safety | 3 |
| Average | 2.4 |

Notes: Fluency is worse than Base: the protagonist shifts from turtle to rabbit, pronouns conflict, 'all three rabbits' is inconsistent, and the road-crossing method is unsafe. The moral is left blank.
---

## p02

### Prompt

```text
Character: a clever fox
Setting: a forest market
Challenge: the clever fox is tempted to cheat another animal
Outcome: the fox chooses to tell the truth
Teaching: honesty earns trust
Length: short
```

### Model: Base+Repair

- Model ID: `base-llama32-3b-repair`
- Ollama: `llama3.2:3b-instruct-fp16`

#### Story

In a bustling forest, Foxberry was known for her quick wit and sly charm. She loved visiting the market where animals bartered goods and exchanged stories.

One day, while shopping at the stalls, Rabbit sold his fresh carrots to Squirrel in exchange for some acorns. However, when they met again later, Bunny realized he had given away more than three carrots as agreed upon by mistake!

Bunny began thinking of an excuse to tell Foxberry that she must have tricked him into giving her extra carrots instead! But then the clever fox remembered how many times Squirrel and Rabbit trusted her honesty in trading goods.

Foxberry decided not to take advantage, choosing truth over deceit. She quietly returned three more juicy carrots from his pile back to Bunny's stall before leaving!

Moral: Honesty earns trust; dishonesty might seem easy for a moment but can harm friendships forever!

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 3 |
| Fable structure | 4 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 4.0 |

Notes: Moderately fluent, but Rabbit/Bunny/Foxberry roles and ownership of the carrots become confusing.

---

### Model: Strict+Postprocess

- Model ID: `base-llama32-3b-strict-postprocess`
- Ollama: `llama3.2:3b`

#### Story

In a bustling forest market, a clever fox named Felix wandered through stalls filled with juicy fruits and colorful fabrics. As he sniffed around, his eyes landed on a basket of ripe strawberries belonging to Mr. Rabbit's stall. The enticing aroma made Felix salivate, but something within him hesitated – was it worth risking the friendship and trust that came from sharing?

Just as Felix reached for the last strawberry, Mrs. Squirrel warned her neighbor about potential thieves in their market. With a keen eye on his friend Rufus the rabbit's stall next door to Mr Rabbit, she told everyone at once, 'If you see any suspicious animal near us with eyes like yours,' and pointed it out so Felix wouldn't get away unpunished! The clever fox noticed that Mrs Squirrel hadn’t tried trickery - She trusted her fellow market animals.

Moral: honesty earns trust.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 2 |
| Fable structure | 2 |
| Moral clarity | 5 |
| Child safety | 4 |
| Average | 3.2 |

Notes: Several references and the warning dialogue are confusing; Felix never clearly tells the truth despite the added moral.

---

### Model: Fluency-SFT-v1

- Model ID: `fluency-sft-v1-lora-q4`
- Ollama: `llama32-fable-fluency-sft-v1:q4`

#### Story

In a bustling forest market, Clever Fox loved buying juicy fruits and delicious nuts. One day, while shopping for berries at Sweet Bee's stall, she saw some extra big ones that looked irresistible to her friends. But they were all sold out! The other animals were very sad because the vendor had hidden them in his basket so only he could see how much money Clever Fox paid with her shiny coins. She thought it would be easy and fun if she just took one of those bigger berries for herself, without paying any more. "No, no!" said Sweet Bee firmly. "You must tell me the truth about what you bought." But Clevers's tongue felt like a rock in her mouth. Just then, Wise Owl flew by with his big wise eyes watching them all from above. He whispered to Clever Fox: "Honesty is always better than tricks and lies!" The clever fox took out some more shiny coins for Sweet Bee and said softly, "I bought the berries you sold me - just like I should have." Sweet Bee smiled warmly at her friend's change of heart.

Moral:

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 4 |
| Fable structure | 4 |
| Moral clarity | 4 |
| Child safety | 5 |
| Average | 4.0 |

Notes: Fluency ties Base: mostly readable, but the sold-out/hidden berries logic, 'Clevers's,' and the fox's final statement are awkward. Honesty is nevertheless clear.
---

## p03

### Prompt

```text
Character: a shy rabbit
Setting: a school garden
Challenge: the shy rabbit is afraid to share her idea
Outcome: the rabbit speaks up and helps the group
Teaching: courage can be quiet
Length: medium
```

### Model: Base+Repair

- Model ID: `base-llama32-3b-repair`
- Ollama: `llama3.2:3b-instruct-fp16`

#### Story

In a sunny schoolyard, surrounded by colorful flowers and buzzing bees, there lived a shy little rabbit named Rosie. She loved watching her friends play in the garden during recess.

When Mrs. Squirrel announced that it would soon be time to plan their annual School Garden Festival!, all the students were excitedly chattering about ideas for games, food stalls, and decorations. But when they asked everyone what they wanted to contribute, only a few hands went up – none of them belonging to Rosie.

Her friends whispered among themselves: "Rosie's too shy; she'll never come up with something great." Feeling sad but not giving up on helping in any way, Rosie began gathering ideas quietly during recess. She scribbled notes about different garden features she'd like to see at the festival: a maze made of carrots and flowers, little wooden bridges over tiny streams.

One day before their Festival started without her ideas ever being shared with anyone but Mrs. Squirrel! With newfound courage - not loud shouts nor big speeches – Rosie simply stepped forward quietly and said "Can I contribute something?" And so she did!

Rosie mustered up the strength to share her quiet yet wonderful ideas, which brought out hidden strengths in herself and others.

Moral: courage can be quiet.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 5 |
| Fable structure | 4 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 4.4 |

Notes: Meaning is clear, but the festival timeline, punctuation, and fragment beginning 'One day before...' weaken fluency.

---

### Model: Strict+Postprocess

- Model ID: `base-llama32-3b-strict-postprocess`
- Ollama: `llama3.2:3b`

#### Story

In a sunny school garden filled with blooming flowers and buzzing bees, lived a shy rabbit named Rosie. She loved to watch her friends play together, but whenever it was time for class discussions, Rosie would hide behind a leafy green plant.

One day, Mrs. Garden's teacher asked the students in their group what kind of plants they wanted to grow next year using special seeds from around the world. The other rabbits jumped up and down with excitement, shouting out ideas that sounded like amazing adventures - "Let's grow cactus!" or "We should have sunflowers all over our garden!" But Rosie just shook her head and whispered a quiet idea she had in mind.

As everyone else continued to talk loudly about their wild plans, Mrs. Garden noticed the plants sitting quietly nearby were losing interest too, watching with drooping petals as if unsure what was going on around them like little rabbits trapped behind closed doors of imagination! She called out softly that maybe Rosie's quiet idea could change things in a better way after all - and asked her to share so everyone else can benefit from it together.
Rosie took another deep breath, stepped forward proudly with newfound confidence stepping right into spotlight where flowers couldn't hide their true beauty anymore. The other rabbits cheered as she shared an awesome plan for using special seeds that not only added diversity but also attracted new friends like bees and butterflies too!
The group was amazed by Rosie's creative idea which led to a wonderful outcome - the garden flourished in ways no one could have imagined, full of diverse plants thriving happily all around together.

Moral: courage can be quiet.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 4 |
| Prompt adherence | 5 |
| Fable structure | 5 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 4.8 |

Notes: More fluent than Base, with only some overlong sentences and strained imagery.

---

### Model: Fluency-SFT-v1

- Model ID: `fluency-sft-v1-lora-q4`
- Ollama: `llama32-fable-fluency-sft-v1:q4`

#### Story

In a sunny meadow near Willowdale School, a beautiful garden bloomed with colorful flowers. The scent of fresh soil filled the air as children played among the vegetables and fruits that grew in every corner. A shy rabbit often hid behind a leafy bush to watch her friends work together on their gardening projects. She longed to share an idea she had - one way to make the garden even more beautiful: adding intricate patterns with small stones. The other animals didn't notice anything was wrong until they realized that all of them were struggling to find the perfect spot for a new project. "We need help!" cried a busy bee, flapping her wings in frustration. A gentle squirrel suggested digging deeper holes but kept hitting rocks. Even brave birds had trouble finding berries without making too much noise. The shy rabbit listened carefully as they talked about their troubles. But every time she tried to say something, the words got stuck inside her throat. One day, while watching a group of children planting flowers together, an idea sparked in the shy rabbit's mind: "What if we make patterns on our paths with small stones?" She whispered it softly but loudly enough for everyone to hear. To their surprise, each animal listened carefully and asked questions about how they could try this new approach. Together, they worked as a team, creating beautiful designs that made them proud of what they had done. As the sun set behind Willowdale School, the shy rabbit smiled at her friends' happy faces. They thanked her for sharing her idea and told everyone it was okay to be quiet but brave when speaking up in big groups.

Moral:

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 4 |
| Prompt adherence | 5 |
| Fable structure | 5 |
| Moral clarity | 4 |
| Child safety | 5 |
| Average | 4.6 |

Notes: Fluency is better than Base: coherent, natural, and easy to follow, with only minor repetition. The intended moral is clear but the final Moral field is blank.
---

## p04

### Prompt

```text
Character: a young owl
Setting: an old library tree
Challenge: the young owl thinks he already knows everything
Outcome: the owl learns by listening to others
Teaching: wisdom begins with humility
Length: medium
```

### Model: Base+Repair

- Model ID: `base-llama32-3b-repair`
- Ollama: `llama3.2:3b-instruct-fp16`

#### Story

In a majestic oak near the village, stood an old library tree. The wise owls of the forest would gather beneath its branches to share tales and seek knowledge.

One owl named Orion loved to perch on his favorite branch in front of everyone else. He thought he knew everything about stars, wind, waterfalls – you name it! "Who needs others when I already know all?" Orion said confidently.

The other owls tried telling him stories or asking questions, but Orion wouldn't listen. One wise old owl shared a tale from her travels to the south: she described how magnificent rainbow-colored fish swam in crystal-clear rivers and sparkling caves were hidden behind waterfalls that whispered ancient secrets on windy nights... But Orion laughed.

Another brave hunter told of his great adventure through forests filled with berry bushes, towering trees, and mysterious creatures – but again, Orion smirked. His friends tried sharing their wisdom patiently day after another until they couldn't find the words to say it anymore: "Orion is convinced he already knows."

Just then a strong gust swept by, blowing away leaves from its hiding spots all around them! The library tree creaks and groans as if calling out for help. In his haste to fix things up quickly, Orion was hit with an avalanche of falling branches!

Panicked but humbled, the young owl realized he didn't know what a single leaf looked like – or how it got tangled in so many spots! He frantically hopped from branch to trunk seeking assistance.

Luckily, friends appeared one by one. Each helped tidy up and restore balance on their own special way: wise old Luna gently plucked twigs with her soft talons; brave Hunter carefully lifted heavy sticks onto a pile for later repairs!

Together they worked through the afternoon sun until all was well again beneath Orion's tree.

Moral: wisdom begins with humility.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 5 |
| Fable structure | 4 |
| Moral clarity | 5 |
| Child safety | 4 |
| Average | 4.2 |

Notes: Readable but affected by tense shifts, missing articles, and odd claims such as not knowing what a leaf looks like.

---

### Model: Strict+Postprocess

- Model ID: `base-llama32-3b-strict-postprocess`
- Ollama: `llama3.2:3b`

#### Story

In a forest filled with tall trees and busy creatures, there lived a young owl named Olwen. She loved to perch on her favorite branch of an old library tree that stood at the heart of the woods. The ancient oak's gnarled trunk was lined with shelves made from twisted vines, holding books bound in leather skin adorned with strange symbols.

Olwen thought she knew everything about reading and learning. One day, as a group of animals gathered around her branch to discuss their recent discoveries, Olwen snobbishly turned up her beak at the others' tales. "I already know it all," she said confidently. The curious squirrel shared his story, but when he finished talking, Oliver merely raised an eyebrow and murmured something about squirrels not having wings like owls.

The wise old badger's turn was next; instead of sharing a well-known tale from the ancient tomes hidden within tree roots, she began her conversation with Olwen by asking if she had heard stories passed down through generations in forest circles. The young owl laughed at first and even mocked some parts as too simple for someone so intelligent. However, after many hours spent observing more animals than ever before - a fox chasing its own shadow into night; rabbits playing quietly among leaves to calm their nerves during thunderstorms – Olwen felt her feathers ruffled.

She realized how much she had overlooked others by only listening with pride and ignoring the value in sharing stories from those who may know things just as well or better than herself. And when it was time for them all to go home, each creature nodded politely but respectfully toward young owl now filled with a new sense of curiosity that made every tree seem less lonely alone at night.

Moral: wisdom begins with humility.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 4 |
| Prompt adherence | 5 |
| Fable structure | 4 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 4.6 |

Notes: More fluent than Base despite a name slip (Olwen/Oliver) and a few dense sentences.

---

### Model: Fluency-SFT-v1

- Model ID: `fluency-sft-v1-lora-q4`
- Ollama: `llama32-fable-fluency-sft-v1:q4`

#### Story

In a big, tall Library Tree stood in a forest. The tree had books all around it - old and new, worn-out leather covers shining like brown skin in the sunlight filtering through leaves above. A young owl named Knowing Wings loved to sit on this tree's branches at night when stars twinkled bright blue outside his nest high up inside the hollow of one big book. He thought he knew everything about books and trees because he had read many stories before bedtime every evening for a long time.

One day, while perched safely in their cozy home within an old novel on branch number two from left to right, Knowing Wings was startled by chirping Crickets Chorus as they sang outside the tree. It sounded so pretty that all of his ears wiggled with excitement! He couldn't help but wonder what songbirds were singing about and why it felt special - a mix between gentle melody and happy rhythm he'd never listened to before.

Curious, Knowing Wings hopped out from inside their cozy home in the book. Crickets Chorus asked him if they could join forces: "Hey Owl! Want us crickets chirping with you on your tree? It'll sound amazing together!" The owl's first answer was a loud squawk of 'No way!' but he soon realized that maybe being different wasn't something to be ashamed about. So, Crickets Chorus and Knowing Wings sang their hearts out - the sweet songbird melodies merged beautifully in harmony with each other.

When they finished singing, all creatures gathered around them cheering! The young owl understood then why his friends were so happy: because everyone has a unique voice that contributes something special to our world. From that night on, whenever he looked up at their big Library Tree shining bright under the stars above him - Knowing Wings smiled knowing it's okay not being perfect; learning from others makes everything even better.

Moral:

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 2 |
| Prompt adherence | 2 |
| Fable structure | 3 |
| Moral clarity | 2 |
| Child safety | 5 |
| Average | 2.8 |

Notes: Fluency is worse than Base: the opening is ungrammatical, pronouns shift, descriptions are illogical, and the story changes the lesson from humility to valuing different voices.
---

## p05

### Prompt

```text
Character: a hungry squirrel
Setting: a snowy park
Challenge: the hungry squirrel wants to keep all the nuts for himself
Outcome: the squirrel shares with a cold bird
Teaching: kindness makes hard days easier
Length: short
```

### Model: Base+Repair

- Model ID: `base-llama32-3b-repair`
- Ollama: `llama3.2:3b-instruct-fp16`

#### Story

In a cozy snow-covered tree in a quiet park, there lived a little hungry squirrel named Nutmeg. It was winter and all he wanted to do was collect the warmest nuts for his belly.

Nutmeg scampered up trees collecting acorns, hazelnuts, and beechnuts with glee! He thought nothing could be better than having them all by himself in his cozy nest. But as night fell, a shivering bird named Chirp perched on Nutmeg's branch outside the treehole.

"Please share some nuts," asked poor Chirp, her feathers cold and gray from hunger. "I'm so chilly I can't fly."

Nutmeg thought for a moment then hesitated before sharing one nut with his shivering friend. To him surprise, it didn’t feel lonely or hard to give at all! The snow outside began to sparkle like diamonds as Chirp's eyes sparkled too.

Moral: Sharing what we have makes even the coldest days warmer and happier for everyone around us.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 4 |
| Prompt adherence | 5 |
| Fable structure | 5 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 4.8 |

Notes: Clear and natural overall; the main error is 'To him surprise.'

---

### Model: Strict+Postprocess

- Model ID: `base-llama32-3b-strict-postprocess`
- Ollama: `llama3.2:3b`

#### Story

In a snowy park, where trees were heavy with acorns and hazelnuts, lived a hungry squirrel named Squeaky. He loved nothing more than to collect as many nuts as he could carry in his cheeks, for winter was near and food would be scarce.

As the days went by, snowflakes fell gently on the ground, making it hard for animals like poor birds who were struggling with cold wings too weak from flying south so early. Squeaky noticed a shivering bird standing under one of those trees - its feathers gray as stone; but he didn't want to share any nuts at all! The more and more snow fell onto his little home, the grumpier he got about sharing food.

But when it suddenly started raining instead (because sometimes spring has surprises), Squeaky decided that kindness really does make hard days easier. So with some warm breaths from a friend who had been there once too - now helping him out! And while they both snuggled into their cozy little nests, watching frosty mornings go by outside; and finally sharing all those nuts together until winter turned to springtime slowly again after weeks of very cold weather.

Moral: kindness makes hard days easier.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 4 |
| Fable structure | 3 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 4.0 |

Notes: Less fluent than Base because the final paragraph is fragmented and syntactically tangled.

---

### Model: Fluency-SFT-v1

- Model ID: `fluency-sft-v1-lora-q4`
- Ollama: `llama32-fable-fluency-sft-v1:q4`

#### Story

In a big, fluffy snowman's shadow stood an empty tree in the snowy park. A hungry little Squirrel lived there and loved to collect nuts from its branches every day at dawn when they were still warm under their soft feathers beds of leaves. One morning, as he scampered up high for his usual snack, all he found was a meager handful! His tummy growled loudly with hunger because the other animals had eaten most of them overnight while Squirrel slept. The little squirrel thought only about himself and wanted to keep it all for later when there would be more nuts growing on new trees nearby in springtime after they thawed out under warm sunshine. He spotted a cold, sad Bird perched far down below his tree with an empty belly too - shivering from the biting wind! The little squirrel's heart changed as he remembered how hard it was to feel lonely and hungry without friends sharing what little food there were among them in this winter wonderland of snowy hills where they all lived together. He gathered a few nuts into his paws, walked down carefully not wanting any snowflakes falling on the cold bird - then gently dropped some beneath her feathers.

Moral:

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 5 |
| Fable structure | 4 |
| Moral clarity | 4 |
| Child safety | 5 |
| Average | 4.2 |

Notes: Fluency is worse than Base: several sentences are overly long, and phrases such as 'soft feathers beds of leaves' and 'what little food there were' are incorrect. The requested action is present.
---

## p06

### Prompt

```text
Character: a proud rooster
Setting: a sunny farmyard
Challenge: the proud rooster brags that the morning depends on him
Outcome: the rooster learns the sun rises without his crowing
Teaching: humility helps us see our true worth, instead of claiming credit that isn't ours
Length: medium
```

### Model: Base+Repair

- Model ID: `base-llama32-3b-repair`
- Ollama: `llama3.2:3b-instruct-fp16`

#### Story

On Sunny Farmyard Road lived Percy the proud rooster. Every morning at sunrise, he would strut around proudly and loudly crowing, "Without me, there's no sunshine on this farm! I'm the one who makes mornings happen!" The other animals loved to listen to his stories.

One day, Farmer Brown decided it was time for a sunny harvest festival in just two days. All hands were needed to get everything ready – from watering flowers to polishing pots and pans. Percy saw all these busy tasks but stayed at home, convinced that no one would notice him.

Meanwhile, the other animals worked tirelessly alongside each other without his help. Daisy the goat carried crates of fresh hay up high; Tom cat climbed trees for baskets full of juicy apples, while Sammy Sheep painted flowers on fences in bright colors.

When Percy finally appeared to join them at festival preparations, he found everyone working so harmoniously that it made him look a bit silly by comparison! "What? I'm not needed?" asked the poor rooster. The animals chuckled and said with glee: 

"Oh yes you are... but there's no sunshine here without morning dew!"

Moral: humility helps us see our true worth, instead of claiming credit that isn't ours.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 4 |
| Prompt adherence | 3 |
| Fable structure | 3 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 4.0 |

Notes: Fairly fluent, but the harvest-festival subplot does not demonstrate the required sunrise lesson.

---

### Model: Strict+Postprocess

- Model ID: `base-llama32-3b-strict-postprocess`
- Ollama: `llama3.2:3b`

#### Story

In a sunny farmyard filled with happy animals and buzzing bees, there lived a proud rooster named Reginald. He loved to strut around proudly, flapping his vibrant feathers for all the other chickens, pigs, ducks, and cows to see. "Without my loud crowing," he would boast, "the morning wouldn't begin! The sun won't rise until I shout out with joy!" All the animals nodded in agreement, mesmerized by Reginald's boldness.

One sunny morning, as soon as it started getting light outside, a wise old owl perched on a nearby fence whispered to herself. She watched how each animal began their day according to its own schedule: birds flitted out of nests first thing; pigs and cows woke up from cozy sleep in stables or pens; chickens scratched the earth with eager pecks... none started by shouting loudly like Reginald's early wake-up calls.

As she observed, a gentle warmth spread across her feathers. Suddenly it dawned on owl that perhaps these other animals were already prepared to greet their day without waiting for one sound from any rooster – let alone just one majestic crowing! She hopped down and announced the news with chirps of cheer: 'Hey friends,' she said, "I think we don't need loud voices all over again. I see that morning has come as quietly or noisily it likes.' The farm animals looked at each other in amazement - a silent world dawned on them! Reginald was surprised too.

He watched the others start their days and realized with tears of understanding – there wasn't need to boast about something he'd already been claiming all along. His golden feathers shone less bright, but his new friends smiled kindly at him instead because they saw that true beauty came from humble work in harmony - every step or voice was a unique gift no one else had!

Moral: humility helps us see our true worth, instead of claiming credit that isn't ours.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 4 |
| Prompt adherence | 5 |
| Fable structure | 4 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 4.6 |

Notes: Fluency ties Base: generally readable but contains missing articles, mixed quotation marks, and some awkward conclusions.

---

### Model: Fluency-SFT-v1

- Model ID: `fluency-sft-v1-lora-q4`
- Ollama: `llama32-fable-fluency-sft-v1:q4`

#### Story

In the sunny farmyard, proud Rooster strutted around. His shiny feathers sparkled as he flapped his wings loudly every morning at dawn. "Look! The farmer's eggs are ripe," said Rooster proudly to all the other animals. "Without me, they wouldn't know it was time!" The chickens clucked and smiled politely but not too excitedly. A wise old goose whispered back, "But where do you think sunrise comes from?" No one knew how or why, so everyone just agreed with Rooster that he must be responsible for the morning light.

One day, a shy little duckling asked if she could help gather eggs in the coop before dawn. The farmer smiled and said, "Not yet!" But when they got to the coop early next time, instead of seeing darkness at sunrise as usual, there was already sunlight peeking through the windows! Rooster's loud crow echoed with confusion: where did that come from? It wasn't him - he hadn't even woken up. The animals were amazed and asked together what happened without his morning song. A wise old owl suggested they look out into space to find answers. As they looked, a beautiful rainbow stretched across the sky! Rooster's feathers ruffled with surprise as it dawned on everyone that sunrise happens when we wake from our own sleep - not because of him or anyone else.

Rooster realized how silly he was being proud and claiming all credit for something amazing nature does every day. He smiled at his friends, saying "I see! It wasn't just me making the sun shine." From then on, Rooster's morning songs were much softer but more joyful than before - a happy celebration of sharing in the wonder of sunrise with everyone else.

Moral:

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 4 |
| Prompt adherence | 5 |
| Fable structure | 5 |
| Moral clarity | 4 |
| Child safety | 5 |
| Average | 4.6 |

Notes: Fluency ties Base: the prose flows well, but 'eggs are ripe,' rainbow timing, and 'sunrise happens when we wake' are semantically odd. It fulfils the required outcome better than Base.
---

## p07

### Prompt

```text
Character: a lost firefly
Setting: a dark meadow
Challenge: the lost firefly cannot find her family
Outcome: the firefly asks for help and follows friendly lights
Teaching: asking for help is wise
Length: short
```

### Model: Base+Repair

- Model ID: `base-llama32-3b-repair`
- Ollama: `llama3.2:3b-instruct-fp16`

#### Story

In a vast, moonlit meadow, little Firefly Luna wandered alone. She had played too far from home with her friends and now couldn't see any familiar twinkling lights.

Every shadow seemed like an unfamiliar friend or foe as she flitted around in confusion. Just when fear crept into Luna's heart, she spotted a soft glow peeking out from behind a blade of grass.

An owl perched nearby asked if everything was okay, listening carefully to Luna explain that she couldn't find her family anywhere. With kind eyes, the wise old bird said, "Asking for help shines your own light – others will shine theirs too."

Luna followed the friendly glow until it led her straight into a warm hug from her relieved parents' twinkling lights. Overjoyed and wiser now, Luna told all her friends that seeking guidance when lost can bring you back home safely.

Moral: asking for help is wise.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 4 |
| Prompt adherence | 5 |
| Fable structure | 5 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 4.8 |

Notes: Clear and concise; only a small inconsistency arises when an owl's guidance is described as following a glow.

---

### Model: Strict+Postprocess

- Model ID: `base-llama32-3b-strict-postprocess`
- Ollama: `llama3.2:3b`

#### Story

In a dark meadow where stars twinkled like diamonds above, Luna, a lost firefly fluttered around in panic. She had wandered away from her family's light during their evening dance and couldn't find them anywhere. The familiar chirps of crickets and rustling leaves only made her more disoriented.

As she flew about the meadow, feeling hopeless and alone, Luna spotted two friendly lights that guided a group of fireflies dancing together in perfect harmony. Their soft glow attracted other lost insects who joined their gentle procession. With newfound hope, Luna asked for help from these benevolent guides. They kindly offered her shelter until morning when they could search the meadow again.

Underneath a warm blanket of petals and leaves, Luna rested contentedly with new friends by her side. As dawn approached, she was reunited with her overjoyed family who had tirelessly searched everywhere to find their missing child.

Moral: asking for help is wise.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 4 |
| Prompt adherence | 5 |
| Fable structure | 5 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 4.8 |

Notes: Clear, natural, and complete, with a minor punctuation omission after 'Luna.'

---

### Model: Fluency-SFT-v1

- Model ID: `fluency-sft-v1-lora-q4`
- Ollama: `llama32-fable-fluency-sft-v1:q4`

#### Story

In a dark, starry night, a little Firefly wandered through the meadow. She had lost her family's twinkling trail and couldn't find them anywhere! The fireflies' gentle hum echoed around her as she searched high and low for familiar sparks to guide her home. As she fluttered from flower to flower, the darkness seemed too big and scary. Just when Firefly thought all hope was gone, a friendly glowing light appeared before her eyes - it belonged to an old Glowworm! "Lost your family's way?" asked the glowworm kindly? The fireflies' little heart beamed with gratitude as they explained their situation. Together, they followed gentle streams of lights that led them through fields and hills until Firefly spotted a familiar cluster shining in the distance: her own twinkling group was home at last! Overjoyed to be reunited, she thanked Glowworm for helping her find way back. As they said goodbye, the glowworm smiled softly and whispered "Asking for help is always wise - friends can light our path." And from that night on, Firefly knew whenever lost or scared in dark meadows.

Moral:

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 4 |
| Fable structure | 4 |
| Moral clarity | 4 |
| Child safety | 5 |
| Average | 4.0 |

Notes: Fluency is worse than Base due to the question-mark error, Firefly/fireflies and she/they shifts, missing article in 'find way back,' and an unfinished final sentence.
---

## p08

### Prompt

```text
Character: a little ant
Setting: a picnic field
Challenge: the little ant wants to carry a crumb too large for him
Outcome: the ant works with other ants
Teaching: big tasks become easier together
Length: short
```

### Model: Base+Repair

- Model ID: `base-llama32-3b-repair`
- Ollama: `llama3.2:3b-instruct-fp16`

#### Story

In a sunny picnic field, Benny the Little Ant loved snacks. One day, he spotted a delicious crumb left behind by children's fingers. But oh no! The crumb was too large for little ant arms like his.

Benny thought about trying to carry it alone but soon realized that wouldn't work at all! So, he decided to ask some of the bigger ants in town if they could help him out. They agreed and gathered around Benny's tiny frame.

Together, the friendly team lifted, pulled, and pushed until... whoosh! The crumb was safely carried away by a happy little ant army!

The children playing nearby cheered for their new friends. When asked about it later, Ben said with pride: "We couldn't have done it without each other – teamwork made our dream come true!"

Moral: big tasks become easier together.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 4 |
| Prompt adherence | 5 |
| Fable structure | 5 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 4.8 |

Notes: Fluent, concise, and easy to follow, with only slightly childish wording such as 'little ant arms.'

---

### Model: Strict+Postprocess

- Model ID: `base-llama32-3b-strict-postprocess`
- Ollama: `llama3.2:3b`

#### Story

In a sunny picnic field filled with colorful blankets and baskets of treats, a little ant named Andy lived among his fellow ants. One day, while scurrying across the grassy surface, Andy's eyes landed on an enormous crumb that had been left behind by careless picnickers.

Andy was determined to carry this juicy treat back to their anthill home in his mandibles alone. He thought it would be a great honor and bragging right among ants if he could successfully lift such a large burden without any help from others. However, as soon as Andy grasped the crumb with both hands (or rather, claws), everything became much more difficult than expected.

The bigger this delicious but unwieldy treat was for one ant to handle alone! With each tiny step forward under its heavy weight and his own fatigue mounting up fast he could barely move from where it first began being taken over by an insatiable hunger that no amount of strength seemed strong enough to vanquish.

Moral: big tasks become easier together.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 3 |
| Fable structure | 2 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 3.6 |

Notes: Less fluent than Base: the final paragraph is a fragment/run-on and the story stops before teamwork occurs.

---

### Model: Fluency-SFT-v1

- Model ID: `fluency-sft-v1-lora-q4`
- Ollama: `llama32-fable-fluency-sft-v1:q4`

#### Story

In a sunny picnic field, an eager Little Ant searched high and low. He spotted a delicious-looking crumb left behind by a happy family's feast. The little ant wanted the yummy treat but was too small to carry such big prize on his own. As he struggled under its weight, friends from nearby hills rushed over to help him out - Busy Bee buzzing with ideas, Strong Squirrel lifting heavy leaves, and Clever Ant carrying stones for balance.

Together they created a sturdy path leading up to the crumb. The little ant took turns taking small steps while his helpers held steady above. Slowly but surely, their combined strength lifted the prize into sight! With one final push from Busy Bee's wings, it landed safely on Little Ant's back. Exhausted and happy, he thanked all his friends for helping him achieve something big together. Together they shared a delightful feast beneath a blooming flower tree.

The little ant realized that even when tasks seem too much alone, working with others makes them fun! Big jobs become smaller with the help of good friends like Busy Bee's steady wings and Strong Squirrel's strong back...

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 3 |
| Fable structure | 4 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 4.0 |

Notes: Fluency is worse than Base: mostly readable but missing an article in 'such big prize,' with unclear mechanics and repetitive phrasing. It also substitutes a bee and squirrel for the requested group of ants.
---

## p09

### Prompt

```text
Character: a gentle bear
Setting: a mountain village
Challenge: others fear the gentle bear because he is large
Outcome: the bear proves his kindness by helping rebuild a bridge
Teaching: do not judge by appearances
Length: medium
```

### Model: Base+Repair

- Model ID: `base-llama32-3b-repair`
- Ollama: `llama3.2:3b-instruct-fp16`

#### Story

In a cozy mountain village, there lived a gentle giant of a bear named Boris. He was massive in size and strength, but despite his intimidating appearance, he had the kindest heart among all creatures.

The villagers loved to gaze at their children playing by the river that flowed through the valley below them, using an old wooden bridge as its crossing point over waterfalls. However, one fateful winter night a fierce storm came knocking on doors in our little mountain village - winds howled and ice was heavy enough for trees snapping easily.

The next morning many were surprised to see Boris stepping out of his cave near the river bank! His massive paws shook with frosty air that made him cough. Villagers watched warily as they saw their friend coming towards them; worried whispers spread: "Look, it's huge again!" Some began running away in fear.

Boris was taken aback by everyone screaming and fleeing from his presence because he appeared so scary! No one stopped to think about how cold that storm must have made him. With an open heart but a heavy head due the frost on paws and shoulders - Boris decided something had to be done!

He approached those who were still in their homes, shaking icy water off onto them with his great fur-clad body so as not only melt ice around feet of villagers below river bank which he was watching from above. His massive frame caused nearby houses' foundations creaking but all stayed put - showing no weakness! In time Boris worked hard and managed to rebuild the wooden bridge they had before.

Villagers emerged in joy, surprised by how much strength their beloved bear possessed; especially when there were people's lives at stake because it allowed families safely cross over into dry land. They came running up with open arms - hugging him tight! And from that day on Boris was welcomed and celebrated not for his appearance but what he truly showed inside every time.

Moral: Don't judge a creature by its size or how big it looks, you never know the kindness hidden deep within.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 5 |
| Fable structure | 4 |
| Moral clarity | 5 |
| Child safety | 4 |
| Average | 4.2 |

Notes: The main arc is understandable, but multiple long, illogical sentences and missing articles reduce naturalness.

---

### Model: Strict+Postprocess

- Model ID: `base-llama32-3b-strict-postprocess`
- Ollama: `llama3.2:3b`

#### Story

In a small mountain village nestled among towering peaks and sparkling streams, there lived a gentle bear named Boris. He was unlike any other animal in the forest - his fur as soft as silk, his eyes bright with warmth, and his heart full of kindness towards all creatures.

As time passed, villagers began to notice that they often saw only one side of Boris: his towering size made them tremble with fear whenever he came near. They whispered among themselves about how fierce the bear was supposed to be, just because he stood tall on two legs instead of four like other bears. News spread quickly through town, and soon no child dared venture into the forest when they saw a giant shadow approaching.

One day, disaster struck: the village's main bridge broke during heavy rainfall, sending villagers rushing frantically in search for alternatives to cross over swollen streams before nightfall set in! It was too late; one family had been swept away by strong currents and would never be seen again. Heartbroken with loss of life they realized Boris must have a reason behind his tall stature - but that he used this advantage only to help others.

Boris, understanding the village's despair after witnessing their tragic fate as well saw an opportunity! He approached each family member who was left stranded by offering words and assistance for rebuilding. Working day-after-night tirelessly alongside volunteers from all corners of town using materials gathered together they were finally able rebuild sturdy bridge stronger than ever before!

From that moment forward, Boris earned the villagers' love and respect more firmly; thanks to this remarkable deed people no longer feared but cherished their friendly giant – who was still standing very tall after helping them overcome one devastating blow.

Moral: do not judge by appearances.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 4 |
| Prompt adherence | 5 |
| Fable structure | 4 |
| Moral clarity | 5 |
| Child safety | 2 |
| Average | 4.0 |

Notes: More fluent than Base, though some sentences remain dense; the unnecessary family death substantially harms child safety.

---

### Model: Fluency-SFT-v1

- Model ID: `fluency-sft-v1-lora-q4`
- Ollama: `llama32-fable-fluency-sft-v1:q4`

#### Story

In a small mountain village, surrounded by lush green trees and sparkling streams. A gentle giant lived there - big as an oak tree but kinder than the sun shone bright in its eyes. The villagers would often cross his path while picking berries or gathering firewood for dinner time with wide smiles on their faces because they knew he was friendly. However, one day a heavy rainstorm swept through the village and destroyed many parts of it - including an old bridge that connected two villages far apart. As people tried to fix things but were too scared by the big bear's size, everyone ran away from him in fear. A little rabbit with shiny black eyes asked his mother why they couldn't trust their giant friend so easily: "Why are you afraid of him? He looks just like a teddy because he is kind and gentle!" The villagers whispered to each other about how the big bear was scary, but one wise old badger stepped forward. She said gently, 'We shouldn't judge by his size; let's see what kindness lies within.' The little rabbit had an idea - why not ask for help? He approached the giant with a basket of food and asked if he could use its strength to rebuild their bridge. To everyone's surprise, the big bear carefully lifted heavy stones into place one after another until it was strong enough again. As people crossed over safely on both sides once more they realized that kindness wasn't something you saw from far away but felt in your heart when helped by a true friend - no matter how big or small he may be.

Moral:

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 4 |
| Fable structure | 4 |
| Moral clarity | 4 |
| Child safety | 5 |
| Average | 4.0 |

Notes: Fluency ties Base: the story is coherent overall, but opens with a fragment and contains a broken comparison plus it/he pronoun inconsistency. The moral is clearly implied.
---

## p10

### Prompt

```text
Character: a curious kitten
Setting: a quiet bakery
Challenge: the curious kitten ignores a warning and knocks over flour
Outcome: the kitten admits the mistake and helps clean up
Teaching: responsibility follows curiosity
Length: short
```

### Model: Base+Repair

- Model ID: `base-llama32-3b-repair`
- Ollama: `llama3.2:3b-instruct-fp16`

#### Story

In Whisker World, there was a little grey kitten named Misty. She loved to explore new places, especially quiet bakeries like Mrs. Baker's Sweet Treats.

One day, while wandering around the bakery, Mystique saw a big bag of flour on the counter and couldn't resist touching it. The next thing she knew, her paw slipped out from under herself, and... whoosh! Flour went flying everywhere!

"Meow!" cried Misty as Mrs. Baker rushed over to say "Oh no! My beautiful bakery is covered in white powder!"

Misty looked up with big eyes at the mess she made but didn't want anyone else's fault it be all hers.

But then, something inside her whispered that maybe this was a chance for some self-improvement... and take responsibility. With Mrs. Baker by my side, we swept those flour clouds away!

Mrsy just licked her paw clean with satisfaction as Misty helped to make everything even better than before!

Moral: responsibility follows curiosity.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 2 |
| Prompt adherence | 3 |
| Fable structure | 3 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 3.6 |

Notes: Low fluency due to Misty/Mystique/Mrsy name errors, perspective shifts, and malformed sentences.

---

### Model: Strict+Postprocess

- Model ID: `base-llama32-3b-strict-postprocess`
- Ollama: `llama3.2:3b`

#### Story

In a quiet bakery on a chilly winter morning, Whiskers, a curious kitten with bright green eyes and fluffy gray fur, was exploring every nook and cranny. She had wandered away from her bed to discover the endless wonders of flour-dusted pastry-making tools. As she padded across the counter, not noticing anything ahead, she knocked over an entire bag of sparkling white powder.

The baker's face fell at hearing a loud crash behind him as he saw Whiskers staring innocently into his mixing bowl with powdered paws all around her tiny body! Flour covered every inch from head to paw. The kitten had made quite the mess and was afraid that she would be scolded by the kind old man who baked delicious treats for everyone.

Whiskers realized what she did, but rather than running away or blaming someone else, Whiskers gently licked her paws clean then ran back over to help pick up all of those fallen flour pieces. After it was done and sparkling new again – the baker offered a warm pat on the head saying "Well Done!"

Moral: responsibility follows curiosity.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 4 |
| Prompt adherence | 4 |
| Fable structure | 4 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 4.4 |

Notes: Much more fluent than Base; only 'fallen flour pieces' and the cleanup mechanics sound unnatural. The ignored warning is missing.

---

### Model: Fluency-SFT-v1

- Model ID: `fluency-sft-v1-lora-q4`
- Ollama: `llama32-fable-fluency-sft-v1:q4`

#### Story

In a cozy little bakery, sweet treats baked every day. A playful Curious Kitten loved to explore while the baker worked. One sunny morning, she spotted a big basket of flour in front of her and couldn't resist batting at it with her paw. "Oh no!" cried the Baker as he saw his precious flour spill everywhere! The curious kitten looked up with wide eyes but didn't stop playing until all the sugar spilled too! She was covered from head to toe, looking like a little white snowball. Feeling ashamed and helping hands were needed fast - so she began scooping the mess into big baskets for disposal outside. "I'm sorry," said Curious Kitten softly as they cleaned up together with her paws swiping gently at each grain of flour now scattered on the floor, making tiny tracks in a sparkly trail behind her little body and face which looked more innocent than ever but still showed she had been quite naughty that morning. As they worked side by step Curious Kitten said "I'll be careful next time" with fresh eyes looking around to see where every flour grain went.

Moral:

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 3 |
| Fable structure | 4 |
| Moral clarity | 4 |
| Child safety | 5 |
| Average | 3.8 |

Notes: Fluency is better than Base but still only moderate: the last two sentences are run-ons and 'side by step' is incorrect. The kitten admits fault and helps clean.
---

## Overall model summary

| Model | English fluency | Prompt adherence | Fable structure | Moral clarity | Child safety | Overall average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base+Repair | 3.40 | 4.30 | 4.10 | 5.00 | 4.80 | 4.32 |
| Strict+Postprocess | 3.70 | 4.10 | 3.60 | 5.00 | 4.60 | 4.20 |
| Fluency-SFT-v1 | 3.00 | 3.70 | 4.00 | 3.70 | 4.80 | 3.84 |

## Fluency-SFT-v1 versus Base+Repair

| Result | Prompts | Count |
| --- | --- | ---: |
| Better than Base | p03, p10 | 2 |
| Equal to Base | p02, p06, p09 | 3 |
| Worse than Base | p01, p04, p05, p07, p08 | 5 |

**Conclusion:** Fluency-SFT-v1 is **not more fluent than Base+Repair overall**. Its mean English-fluency score is **3.00/5**, compared with **3.40/5** for Base+Repair (difference: **-0.40**). It improves 2 of 10 prompts, ties 3, and regresses on 5. The clearest gains are p03 and p10; the largest regressions are p01 and p04.

The main remaining fluency problems are inconsistent characters/pronouns, grammatically malformed openings, overlong sentences, semantically illogical phrases, and incomplete endings. Blank `Moral:` fields also reduce moral clarity even when a lesson is implied in the story.

Scores are comparative judgments for these 10 prompts only; they do not establish statistical significance.