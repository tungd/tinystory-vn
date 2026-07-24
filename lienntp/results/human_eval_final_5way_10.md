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

### Model: Base FP16

- Model ID: `base-llama32-3b-instruct`
- Ollama: `llama3.2:3b-instruct-fp16`

#### Story

In a peaceful pond, lived a little turtle named Terry. One sunny day, he decided that adventure called him to explore beyond the water's edge. As he made his way closer to where some tall trees stood guard near busy Roady Way, Terry realized it was impossible for just one tiny creature like himself to cross.

Panicked and feeling all alone, Terry thought of a clever plan but didn't know how to execute it yet! Luckily, as soon as the little turtle confessed about wanting help from his pond friends (a wise old fish named Freddy), they quickly sprang into action!

Together with some frogs who climbed trees nearby for high ground view points and rabbits carrying sturdy vines on their backs that formed a rope bridge...the whole crew got to work making it possible! Under Terry's guidance, carefully constructed walkways allowed him the safe passage across Roady Way. 

Moral: When friends unite towards one goal with patience in harmony - there is nothing big or small they can't accomplish together as best of buddies

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 4 |
| Prompt adherence | 4 |
| Fable structure | 4 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 4.4 |

Notes: Complete and safe, with a clear moral; a few awkward phrases and an overcomplicated rope-bridge solution.

---

### Model: SFT Clean 3K

- Model ID: `sft-llama32-3b-clean3k`
- Ollama: `llama32-fable-clean3k:q4`

#### Story

In a peaceful pond, lived tiny Terry the Turtle. One sunny day, he decided to explore beyond its waters by crossing over to greener pastures on another side of the lake.

As soon as his feet touched dry land near our quiet home - little did anyone know but what happened next! A speeding car came driving right towards where they all played outside every single morning without fail until some brave souls stepped forward & saved everyone from bad luck!

Feeling grateful yet scared after such an adventure so close back at their pond again felt happy knowing now that whenever something seems too hard or scary we needn't go alone because our friends will always lend us helping hands when needed most! And last but not least... Moral:

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 2 |
| Prompt adherence | 2 |
| Fable structure | 2 |
| Moral clarity | 1 |
| Child safety | 4 |
| Average | 2.2 |

Notes: Confused event sequence, unclear helpers and outcome, and the story ends after an empty moral label.

---

### Model: Failure LoRA

- Model ID: `failure-lora-llama32-3b`
- Ollama: `llama32-fable-failure-lora:q4`

#### Story

In a peaceful pond, a little green turtle lived with its fish family. One sunny day, our brave turtle friend wanted to reach new waters but found itself standing near the edge of his home – right in front of an old and busy road where cars drove by fast.

As it looked around for help, some worried voices whispered "but you can't! The big noisy machines are too scary!" But one wise fish said softly, “If we work together as friends will make our little turtle safe.” So the brave ones gathered from all sides of pond. A family duck with sharp eyes spotted a long rope hanging above trees to cross over streams for ducks and decided that same rope should be used.

A tiny dragonfly who flew very fast came down quickly below, using her strong body lift up big leaf like bridge high on wall so little turtle can safely walk across road. And one wise old fish gave him last gentle nudge forward when he was almost there...

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 3 |
| Fable structure | 3 |
| Moral clarity | 2 |
| Child safety | 5 |
| Average | 3.2 |

Notes: Includes the main situation and attempted help, but the crossing and intended moral are not clearly completed.

---

### Model: Strict Prompt

- Model ID: `base-llama32-3b-strict-prompt`
- Ollama: `llama3.2:3b`

#### Story

In a quiet pond surrounded by tall trees and greenery, lived a small turtle named Terry. He loved swimming in the clear water with his friends, chasing fish, and basking under warm sunbeams on large lily pads.

One sunny afternoon, as Terry was exploring near the edge of the pond, he noticed how busy the road that ran along its border had become. Cars whizzed by, making it impossible for him to cross safely alone. The small turtle longingly yearned to reach a nearby patch where juicy watercress grew in abundance.

As fate would have it, Terry's friends - Max the fish and Lily the frog - were swimming near his spot of longing when they observed their friend's predicament. Without hesitation, they joined forces with an old tortoise named Sam who had some years on him but was not too slow for speed. Together, these four animals devised a careful plan to help Terry reach safety beyond the road while keeping themselves safe from harm and traffic noise.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 4 |
| Prompt adherence | 3 |
| Fable structure | 2 |
| Moral clarity | 1 |
| Child safety | 5 |
| Average | 3.0 |

Notes: Strong setup and conflict, but it stops at planning: no completed crossing and no moral.

---

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

Notes: Complete and safe with an explicit moral; wording is occasionally awkward.
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

### Model: Base FP16

- Model ID: `base-llama32-3b-instruct`
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

Notes: The honesty lesson is explicit, but character roles and carrot ownership shift confusingly.

---

### Model: SFT Clean 3K

- Model ID: `sft-llama32-3b-clean3k`
- Ollama: `llama32-fable-clean3k:q4`

#### Story

In Forest Market, creatures gathered daily. Clever Fox loved trading goods there.
One sunny day, while shopping for fresh berries,
Clever Fox saw Rabbit struggling to carry heavy baskets full of bright carrots.

Feeling sorry but also wanting some juicy treats all his own, the clever fox thought: "Why should I share my money when no one knows?" Just then, wise Owl perched nearby and watched closely. After seeing what happened next – a friendly squirrel helped rabbit by sharing their basket - Clever Fox realized everyone could use an extra helping hand or word.

Feeling ashamed yet grateful to owl for teaching him right from wrong,
Clever Fox went back home proud of choice made earlier that day.
When his family asked how was shopping done so wisely, the fox smiled genuinely. With honesty earned trust now within their little community too! 

Moral: Honesty earns trust

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 2 |
| Prompt adherence | 3 |
| Fable structure | 2 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 3.4 |

Notes: States the moral, yet never clearly depicts cheating, truth-telling, or a concrete resolution.

---

### Model: Failure LoRA

- Model ID: `failure-lora-llama32-3b`
- Ollama: `llama32-fable-failure-lora:q4`

#### Story

In a sunny forest, a hungry rabbit went shopping at the bustling market. There were stalls selling juicy carrots and plump berries for all animals. Among them was Benny the clever fox, with his bright eyes shining like stars.

As they browsed through the colorful fruits and vegetables, a sly raccoon whispered to Benny: "Pick that big red apple from Mrs.'s stall; it's hers!" The other shoppers saw how tempting it looked on the counter nearby. But Benny thought about what would happen if he took something without permission... or was found lying.

He remembered his mom teaching him an old saying, 'An honest heart shines brighter than any treasure.' So when Mrs., a wise badger with warm eyes like honey, caught her shiny apple being picked by someone's paw – and saw Benny looking very scared! She smiled warmly at the young fox: "Well done for not taking my prize. Would you help me find fresh carrots instead?" And from that day on...

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 4 |
| Fable structure | 3 |
| Moral clarity | 3 |
| Child safety | 5 |
| Average | 3.6 |

Notes: Shows resistance to temptation and supports honesty, though trust is only implied and the ending is unfinished.

---

### Model: Strict Prompt

- Model ID: `base-llama32-3b-strict-prompt`
- Ollama: `llama3.2:3b`

#### Story

In a bustling forest market, a clever fox named Felix wandered through stalls filled with juicy fruits and colorful fabrics. As he sniffed around, his eyes landed on a basket of ripe strawberries belonging to Mr. Rabbit's stall. The enticing aroma made Felix salivate, but something within him hesitated – was it worth risking the friendship and trust that came from sharing?

Just as Felix reached for the last strawberry, Mrs. Squirrel warned her neighbor about potential thieves in their market. With a keen eye on his friend Rufus the rabbit's stall next door to Mr Rabbit, she told everyone at once, 'If you see any suspicious animal near us with eyes like yours,' and pointed it out so Felix wouldn't get away unpunished! The clever fox noticed that Mrs Squirrel hadn’t tried trickery - She trusted her fellow market animals.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 2 |
| Fable structure | 2 |
| Moral clarity | 1 |
| Child safety | 4 |
| Average | 2.4 |

Notes: Ends before Felix chooses honesty or earns trust; several references are confusing.

---

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

Notes: Clear moral and resolution, but inconsistent animal roles weaken adherence and fluency.
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

### Model: Base FP16

- Model ID: `base-llama32-3b-instruct`
- Ollama: `llama3.2:3b-instruct-fp16`

#### Story

In a sunny schoolyard, surrounded by colorful flowers and buzzing bees, there was a shy little rabbit named Rosie. She loved watching her friends play in the garden during recess.

One day, Mrs. Squirrel announced that it would be time to plan their annual School Garden Festival! The students were excitedly chattering about ideas for games, food stalls, and decorations. But when they asked everyone what they wanted to contribute, only a few hands went up – none of them belonging to Rosie.

The other rabbits whispered among themselves: "Rosie's too shy; she'll never come up with something great." The bees buzzed around her head, saying "Bees do the best jobs!" and leaving their friends feeling left out. Even Mrs. Squirrel gently mentioned that maybe it was time for someone else to lead.

Feeling sad but not giving up on helping in any way, Rosie began gathering ideas quietly during recess – scribbling notes about different garden features she'd like to see at the festival: a maze made of carrots and flowers, little wooden bridges over tiny streams. While others were busy chatting or drawing more grand plans for food stalls.

As it turned out one day before their Festival was supposed to start without her ideas ever being shared with anyone but Mrs. Squirrel! Rosie mustered up courage - not loud shouts nor big speeches like those kids she'd heard of, no – simply stepped forward quietly and said "Can I contribute something?"

Mrs. Squirrel's eyes lit upon a fresh piece of paper in front of the rabbit who was trembling all over.

Here are my contributions for our School Garden Festival:

- Maze made from carrot sticks
-and little bridges

Her friends stared at her, but Mrs. squirrel just smiled! She said "Thank you Rosie – your small and simple ideas bring out hidden strengths!"

That day we had lots of fun playing together in the garden while everyone helped build something amazing thanks to all our different ideas.

Moral:

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 4 |
| Fable structure | 4 |
| Moral clarity | 2 |
| Child safety | 5 |
| Average | 3.6 |

Notes: Rosie quietly speaks up and helps the festival, but the explicit moral is missing and some sentences are awkward.

---

### Model: SFT Clean 3K

- Model ID: `sft-llama32-3b-clean3k`
- Ollama: `llama32-fable-clean3k:q4`

#### Story

In a sunny meadow near Mrs. Fluffers' class, there was a beautiful school garden filled with colorful flowers, buzzing bees, and happy children tending to them together.

Lily the rabbit lived among these lovely friends but she rarely spoke up unless someone asked her for help finding lost seeds or forgotten watering cans by accidently bumping into each other while working near their homes - which made everyone smile at how kind-hearted this shy little creature looked like every day after class when all students gathered around Lily saying thank you so much sweetie! 

One afternoon, as the group worked on planting new flowers, Emily asked for ideas to make them look even more amazing. Several classmates shared theirs – but then no one said anything because someone else had spoken up already before they could say what she thought too!

Feeling embarrassed and nervous about sharing her own brilliant idea (which involved creating tiny stone paths leading from bigger blooms), Lily decided not speak - hoping nobody noticed that wonderful suggestion hidden deep inside of heart beating fast under layers soft fluffy fur as if trying hide itself completely away forever after all those lovely mornings spent smiling shyly whenever class ended every time friends came running towards school gates calling good morning everyone including our very own little secret keeper – who knew hiding secrets could hurt more than speaking freely sometimes?

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 2 |
| Prompt adherence | 2 |
| Fable structure | 1 |
| Moral clarity | 1 |
| Child safety | 5 |
| Average | 2.2 |

Notes: Long run-on prose and no resolution: Lily never shares the idea or helps the group.

---

### Model: Failure LoRA

- Model ID: `failure-lora-llama32-3b`
- Ollama: `llama32-fable-failure-lora:q4`

#### Story

In a sunny meadow near a bustling school, there was an empty plot of land that had been waiting for children to grow beautiful flowers. Every day at lunchtime when the students took their break outside, they'd play together in this space, but one little rabbit felt left out.

This shy and quiet bunny liked watching her friends as much as playing with them; however, whenever she tried telling an idea of hers about how pretty sunflowers could look around a central pond to attract bees for our school's garden salad days. She got nervous and stayed hidden behind the big tree where they planted tomatoes once upon a time.

One beautiful day when teacher Ms Green decided that every student should draw their favorite type flower from this new empty space, all students were super excited except little shy bunny. As she looked around at other flowers drawing on paper being passed between friends while everyone laughed and admired each colorful picture her eyes met one friendly squirrel who sat quietly by the tree listening to our school’s bird song during recess yesterday.

The teacher asked every student what flower they drew next, but when it came time for little bunny's turn she stood there very still with an empty piece of paper. She took a deep breath and remembered how much fun she had watching bees flying around flowers near her home on sunny days last summer so that day while thinking quietly to herself I could show Ms Green what sunflowers look like in our garden space if the teacher says we can help plant them too!

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 3 |
| Fable structure | 2 |
| Moral clarity | 1 |
| Child safety | 5 |
| Average | 2.8 |

Notes: Builds toward courage but stops before the rabbit clearly speaks up, helps the group, or states the moral.

---

### Model: Strict Prompt

- Model ID: `base-llama32-3b-strict-prompt`
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

Notes: Fulfils all requested elements with a complete resolution and exact moral; only minor verbosity/awkward imagery.

---

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

Notes: Meets the requested outcome and moral, though grammar and timeline remain uneven.
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

### Model: Base FP16

- Model ID: `base-llama32-3b-instruct`
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

Moral:

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 4 |
| Fable structure | 4 |
| Moral clarity | 2 |
| Child safety | 4 |
| Average | 3.4 |

Notes: Good humility arc and resolution, but the blank moral label weakens moral clarity; falling branches are mildly intense.

---

### Model: SFT Clean 3K

- Model ID: `sft-llama32-3b-clean3k`
- Ollama: `llama32-fable-clean3k:q4`

#### Story

In a cozy nook of the ancient Library Tree, lived a curious little Owl named Olwen. She spent most days perched on her favorite branch, reading books and thinking she knew it all already.

Olwen loved to listen when wise Old Badgers shared their stories under the tree's gnarled limbs or whispered secrets through leaves that swayed softly in breeze like nature’s own lullabies; however sometimes arrogance crept into her heart making some creatures uncomfortable around olven which made them stop sharing what they had learned so far from long walks across forest floors!

One sunny afternoon while sitting on high shelf above floor level reading old leather-bound tales without noticing others nearby - a quick glance revealed no friend near except those big round eyes peeking quietly behind trunk nearby! Feeling puzzled she set aside worn-out book pages gently folding them up before taking flight down branches below where soft whispers followed closely until finding hidden clearing filled beautiful flowers blooming everywhere including rare orchids neither bird nor beast knew existed yet here!

Now feeling grateful towards fellow travelers sharing knowledge - Olwen realized learning never truly ends as long friend keeps teaching us something new every single day! With newfound understanding growing inside heart she gently whispered thank you back then returned home nestled comfortably beside Book Tree once again ready learn all over again today

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 2 |
| Prompt adherence | 4 |
| Fable structure | 3 |
| Moral clarity | 3 |
| Child safety | 5 |
| Average | 3.4 |

Notes: Olwen learns that knowledge continues, but the listening mechanism and humility lesson are mostly implicit; prose is run-on.

---

### Model: Failure LoRA

- Model ID: `failure-lora-llama32-3b`
- Ollama: `llama32-fable-failure-lora:q4`

#### Story

In a forest far away, under branches of an old library tree, lived a wise young owl named Olwen. She loved reading about magical adventures in the books that her mother collected from traveling birds.

One sunny day while fluttering around her nest made out of book pages and feathers, she thought to herself: 'I know it all already! I've read my way through every fairy tale.' As a result, whenever another bird asked for advice or needed help with something new Olwen would say proudly that there was nothing they didn't know. They listened carefully but tried not tell her because of the big smile on their faces showing how kind she could be.

But soon disaster struck! When autumn arrived and rain poured down all day long, making a small pond form under where Olwen lived in comfort, most birds were scared that it might freeze so they couldn't cross to find food. They called upon Olwen for help because of her claim to know everything about the world outside their nest.

At first she laughed but after hearing stories from others and listening carefully as well - not just pretending like always before! She discovered many things new which made sense only if listened hard enough or understood deeply too instead thinking 'this is how it's done!' From talking with other birds who'd been through those very same storms, Olwen learned that the rain helped wash away mud to create clean pathways and showed everyone they could be brave even in scary times.

When all was well again after days passed under sunny skies when warm breeze filled up their feathered hearts once more - many of her friends flew near saying: 'thank you ol' wise owl!' Olwen's big smile softened a bit because now she knew that not knowing meant being open to find out, grow and see the world in new ways every time.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 5 |
| Fable structure | 4 |
| Moral clarity | 4 |
| Child safety | 5 |
| Average | 4.2 |

Notes: Clearly shows arrogance changing through listening; the moral is strongly implied rather than stated verbatim.

---

### Model: Strict Prompt

- Model ID: `base-llama32-3b-strict-prompt`
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

Notes: Strong adherence and explicit moral, with a coherent learning arc despite some awkward phrasing.

---

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

Notes: Complete humility lesson with explicit moral; awkward grammar and the branch accident reduce polish/safety slightly.
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

### Model: Base FP16

- Model ID: `base-llama32-3b-instruct`
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

Notes: Complete, concise, wholesome, and closely aligned; only a minor grammar error ('To him surprise').

---

### Model: SFT Clean 3K

- Model ID: `sft-llama32-3b-clean3k`
- Ollama: `llama32-fable-clean3k:q4`

#### Story

In a snow-covered park, a little brown squirrel named Nutmeg scurried around searching for food. His belly growled loudly as he spotted many delicious-looking nuts on the ground.

Nutmeg wanted those all to himself because everyone knows how hard it gets when you're hungry! But then, his eyes fell upon a shivering bird perched under an icy tree branch nearby - its feathers were ruffled by strong wind gusts. Feeling sorry for her coldness after seeing what others do during harsh weather conditions like this made him decide not just grab everything but also share some love too!

Nutmeg picked up several juicy acorns and carefully carried them over towards the poor bird, offering warmth with every step closer until they shared their snack together - both happy now. Now whenever winter brings bitter chill outside each friend helps other find sheltered places where meals await safely inside warm cozy nests made just right size friendly helping hands can turn any cold night into happier days filled friendship love kindness makes hard times more bearable especially when sharing comes naturally like how two become best buddies after being kind towards one another 

Moral: Kindness makes hard days easier

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 2 |
| Prompt adherence | 5 |
| Fable structure | 4 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 4.2 |

Notes: All requested elements and moral are present, but the final sentence is a long, difficult run-on.

---

### Model: Failure LoRA

- Model ID: `failure-lora-llama32-3b`
- Ollama: `llama32-fable-failure-lora:q4`

#### Story

In a snow-covered park, little Squeaky the squirrel was searching for winter food. He spotted many delicious nuts on a bench and scampered up to grab them all! The sun beat down as he ran around with his treasure in mouth.

Meanwhile, high above him flew Wobbly wings Birdie who shivered from cold air brushing against her feathers. "I'm so tired," she cried out loudly while searching for food nearby the same park bench where Squeaky sat happily munching on nuts. Seeing how sad and weak he looked after eating almost all of his winter supply, a clever idea struck him: why not share some of those precious treasures? So with each delicate step down from that snow-covered perch to walk back up once again toward Birdie's direction so she could get warm food too – one shiny nut at time - until it became plain for everyone watching their tiny friend.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 4 |
| Fable structure | 3 |
| Moral clarity | 2 |
| Child safety | 5 |
| Average | 3.4 |

Notes: The squirrel shares, but pronouns are inconsistent and neither the resolution nor moral is clearly finished.

---

### Model: Strict Prompt

- Model ID: `base-llama32-3b-strict-prompt`
- Ollama: `llama3.2:3b`

#### Story

In a snowy park, where trees were heavy with acorns and hazelnuts, lived a hungry squirrel named Squeaky. He loved nothing more than to collect as many nuts as he could carry in his cheeks, for winter was near and food would be scarce.

As the days went by, snowflakes fell gently on the ground, making it hard for animals like poor birds who were struggling with cold wings too weak from flying south so early. Squeaky noticed a shivering bird standing under one of those trees - its feathers gray as stone; but he didn't want to share any nuts at all! The more and more snow fell onto his little home, the grumpier he got about sharing food.

But when it suddenly started raining instead (because sometimes spring has surprises), Squeaky decided that kindness really does make hard days easier. So with some warm breaths from a friend who had been there once too - now helping him out! And while they both snuggled into their cozy little nests, watching frosty mornings go by outside; and finally sharing all those nuts together until winter turned to springtime slowly again after weeks of very cold weather.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 4 |
| Fable structure | 3 |
| Moral clarity | 4 |
| Child safety | 5 |
| Average | 3.8 |

Notes: Contains the requested choice and lesson, though the rain/spring digression makes the resolution less direct.

---

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

Notes: Complete, concise, wholesome, and clearly moralized; one minor grammar error remains.
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

### Model: Base FP16

- Model ID: `base-llama32-3b-instruct`
- Ollama: `llama3.2:3b-instruct-fp16`

#### Story

On Sunny Farmyard Road lived Percy the proud rooster. Every morning at sunrise, he would strut around proudly and loudly crowing, "Without me, there's no sunshine on this farm! I'm the one who makes mornings happen!" The other animals loved to listen to his stories.

One day, Farmer Brown decided it was time for a sunny harvest festival in just two days. All hands were needed to get everything ready – from watering flowers to polishing pots and pans. Percy saw all these busy tasks but stayed at home, convinced that no one would notice him.

Meanwhile, the other animals worked tirelessly alongside each other without his help. Daisy the goat carried crates of fresh hay up high; Tom cat climbed trees for baskets full of juicy apples, while Sammy Sheep painted flowers on fences in bright colors.

When Percy finally appeared to join them at festival preparations, he found everyone working so harmoniously that it made him look a bit silly by comparison! "What? I'm not needed?" asked the poor rooster. The animals chuckled and said with glee: 

"Oh yes you are... but there's no sunshine here without morning dew!"

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 4 |
| Prompt adherence | 2 |
| Fable structure | 2 |
| Moral clarity | 1 |
| Child safety | 5 |
| Average | 2.8 |

Notes: The festival subplot does not prove that the sun rises without crowing, and there is no clear resolution or moral.

---

### Model: SFT Clean 3K

- Model ID: `sft-llama32-3b-clean3k`
- Ollama: `llama32-fable-clean3k:q4`

#### Story

On Sunshine Farm, Percy the proud Rooster ruled over morning duties. Every day at dawn, he'd strut around proudly saying, "I do all this work! Without my mighty crowing sounder thingy," people would wake up from their cozy beds and start fresh.

One sunny afternoon while taking a leisurely stroll near his favorite nesting box tree, Percy spotted some little chicks hiding behind its leafy branches. One of them asked why everyone was always so busy before breakfast time every morning? "That's because I'm doing my job," replied proud old Rooster!

But when he got home that night after helping Farmer Brown fix broken tools on his land and tending to animals too, no matter how big or tiny they were...and couldn't hear even a single peep out loud anywhere! Then only did Percy begin thinking maybe there was something else causing morning sounds besides just himself - perhaps those beautiful birds singing sweet songs high above ground?

Next day dawn arrived again & so did all other farm animals waking up feeling excited because now their voices heard beautifully together after night rest. As soon as everyone started making noise, Farmer Brown looked very happy too! And finally understood truth was told long ago by wise old owl somewhere... 

Moral: humility helps us see our true worth

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 2 |
| Prompt adherence | 2 |
| Fable structure | 2 |
| Moral clarity | 4 |
| Child safety | 5 |
| Average | 3.0 |

Notes: Attempts a humility lesson but never clearly shows the sun rising without Percy; grammar is frequently broken.

---

### Model: Failure LoRA

- Model ID: `failure-lora-llama32-3b`
- Ollama: `llama32-fable-failure-lora:q4`

#### Story

On a sunny farmyard full of happy animals, there lived a proud rooster named Rufus. Every morning at dawn, the other chickens would wake up and say to each other: "Let's see if our big brother has crowed yet!" If they heard that loud roar outside their coop door, it was like sunshine in their hearts – everything looked bright again.

One day when all friends were still asleep inside the cozy coops, Rufus woke them with his proud morning singing. The chickens cheered for him and said: "Today is going to be a great one because of our big brother!" They started getting excited about breakfast time but couldn't understand what was happening outside - why did their door open? That's when they saw Mr. Sun rise up through the blue sky!

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 2 |
| Fable structure | 1 |
| Moral clarity | 1 |
| Child safety | 5 |
| Average | 2.4 |

Notes: Stops just as the sun rises and never shows Rufus learning humility or states the moral.

---

### Model: Strict Prompt

- Model ID: `base-llama32-3b-strict-prompt`
- Ollama: `llama3.2:3b`

#### Story

In a sunny farmyard filled with happy animals and buzzing bees, there lived a proud rooster named Reginald. He loved to strut around proudly, flapping his vibrant feathers for all the other chickens, pigs, ducks, and cows to see. "Without my loud crowing," he would boast, "the morning wouldn't begin! The sun won't rise until I shout out with joy!" All the animals nodded in agreement, mesmerized by Reginald's boldness.

One sunny morning, as soon as it started getting light outside, a wise old owl perched on a nearby fence whispered to herself. She watched how each animal began their day according to its own schedule: birds flitted out of nests first thing; pigs and cows woke up from cozy sleep in stables or pens; chickens scratched the earth with eager pecks... none started by shouting loudly like Reginald's early wake-up calls.

As she observed, a gentle warmth spread across her feathers. Suddenly it dawned on owl that perhaps these other animals were already prepared to greet their day without waiting for one sound from any rooster – let alone just one majestic crowing! She hopped down and announced the news with chirps of cheer: 'Hey friends,' she said, "I think we don't need loud voices all over again. I see that morning has come as quietly or noisily it likes.' The farm animals looked at each other in amazement - a silent world dawned on them! Reginald was surprised too.

He watched the others start their days and realized with tears of understanding – there wasn't need to boast about something he'd already been claiming all along. His golden feathers shone less bright, but his new friends smiled kindly at him instead because they saw that true beauty came from humble work in harmony - every step or voice was a unique gift no one else had!

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 4 |
| Prompt adherence | 5 |
| Fable structure | 4 |
| Moral clarity | 4 |
| Child safety | 5 |
| Average | 4.4 |

Notes: Best match: the morning proceeds independently and Reginald understands his limited role; the exact moral is implicit.

---

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

Notes: Adds the requested moral, but the plot still fails to demonstrate that sunrise occurs without crowing.
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

### Model: Base FP16

- Model ID: `base-llama32-3b-instruct`
- Ollama: `llama3.2:3b-instruct-fp16`

#### Story

In a vast, moonlit darkness, Luna the little firefly wandered. She had strayed too far from her family's twinkling light while playing with friends and couldn't find them anywhere.

As she flitted about in confusion, every shadow looked like an unknown stranger, making it hard for her to recognize friendly faces or lights. Just when fear began to creep into Luna's tiny heart, a gentle glow caught sight of the lost firefly from behind a blade of grass.

A soft-spoken owl asked if everything was alright and listened as Luna explained how she couldn't find her family. The wise old bird offered words of encouragement: "Asking for help is like shining your light – it makes others shine theirs to guide you home."

Luna followed the friendly glow, which led her straight into a warm hug from her relieved parents' twinkling lights. Grateful and wiser, Luna told all her friends about seeking guidance when lost.

Moral: Asking for help can lead you right back where you need to be!

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 4 |
| Prompt adherence | 5 |
| Fable structure | 5 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 4.8 |

Notes: Complete, clear, safe, and strongly aligned; the owl is described as a glow once, creating a small inconsistency.

---

### Model: SFT Clean 3K

- Model ID: `sft-llama32-3b-clean3k`
- Ollama: `llama32-fable-clean3k:q4`

#### Story

In a dark meadow, Luna, a tiny lost firefly, fluttered around frantically. She had been separated from her family during their evening flight home.

Luna searched high and low but couldn't find any familiar faces among the tall flowers or buzzing insects. As night fell heavier, she began to worry that no one would ever see her again.

Just as tears sparkled on Luna's shiny wings, a gentle glow caught her eye - two friendly fireflies led by an older light named Nova! They welcomed Luna warmly and asked if they could assist their new friend in finding its family. Overjoyed at having someone to trust with such big problems like hers own small ones were very manageable after all!

Guiding themselves towards familiar landmarks, the group finally spotted a cozy patch of glowing flowers where Luna's mom waited anxiously under them together forever! 
Moral: asking for help is wise

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 5 |
| Fable structure | 4 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 4.4 |

Notes: All key elements are present and the exact moral is stated; several sentences are grammatically tangled.

---

### Model: Failure LoRA

- Model ID: `failure-lora-llama32-3b`
- Ollama: `llama32-fable-failure-lora:q4`

#### Story

In a vast, star-filled night sky, Luna found herself alone in an empty meadow. She had wandered too far from her family's twinkling lanterns and couldn't see them anywhere.

"Oh no!" Luna cried out softly to the darkness. "I don't know how I'll find my way home!"

Just then, she spotted a friendly firefly light hovering near a patch of tall flowers. It approached Luna with warmth in its glow.

"Don't worry little one," said her new friend. "We can help you shine brighter and guide your family to where they are."

Together, the two friends joined hands – or rather, sparkles on their lanterns – lighting up the meadow path ahead of them like a beacon for Luna's returning light from home.

They walked together through tall grasses until finally Luna saw her smiling face shining at the very edge of this beautiful clearing.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 4 |
| Fable structure | 3 |
| Moral clarity | 2 |
| Child safety | 5 |
| Average | 3.4 |

Notes: Friendly light helps Luna, but she does not clearly ask for help and the ending/moral is incomplete.

---

### Model: Strict Prompt

- Model ID: `base-llama32-3b-strict-prompt`
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

Notes: Complete and well aligned, with a clear request for help, reunion, and explicit moral.

---

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

Notes: Complete, concise, safe, and explicitly moralized; only minor wording inconsistencies.
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

### Model: Base FP16

- Model ID: `base-llama32-3b-instruct`
- Ollama: `llama3.2:3b-instruct-fp16`

#### Story

In a sunny picnic field, Benny the Little Ant loved snacks. One day, he spotted a delicious crumb left behind by children's fingers. But oh no! The crumb was too large for little ant arms like his.

Benny thought about trying to carry it alone but soon realized that wouldn't work at all! So, he decided to ask some of the bigger ants in town if they could help him out. They agreed and gathered around Benny's tiny frame.

Together, the friendly team lifted, pulled, and pushed until... whoosh! The crumb was safely carried away by a happy little ant army!

The children playing nearby cheered for their new friends. When asked about it later, Ben said with pride: "We couldn't have done it without each other – teamwork made our dream come true!"

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 4 |
| Prompt adherence | 5 |
| Fable structure | 5 |
| Moral clarity | 4 |
| Child safety | 5 |
| Average | 4.6 |

Notes: Clear teamwork and successful resolution; the moral is expressed naturally in dialogue rather than as a final label.

---

### Model: SFT Clean 3K

- Model ID: `sft-llama32-3b-clean3k`
- Ollama: `llama32-fable-clean3k:q4`

#### Story

In a sunny picnic field, a busy little ant named Annie loved helping her friends. One day, while exploring the grassy area, she found an enormous crumb dropped by humans.

Annie thought this was the best thing ever! "I can carry that big crumb back to our anthill!" But as soon as she tried picking it up alone... Oh no!

Her tiny legs struggled hard but couldn't budge even a small piece of ground under such weight. Feeling embarrassed and frustrated, Annie went searching for others' help.

Soon enough, many ants joined forces - some strong ones lifted from below while other quick workers helped balance above! Together they slowly moved towards their cozy home where everyone cheered upon successful delivery!

Together we can accomplish amazing things if only share efforts

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 5 |
| Fable structure | 5 |
| Moral clarity | 4 |
| Child safety | 5 |
| Average | 4.4 |

Notes: Complete teamwork arc with a clear implied lesson; a few grammar issues remain.

---

### Model: Failure LoRA

- Model ID: `failure-lora-llama32-3b`
- Ollama: `llama32-fable-failure-lora:q4`

#### Story

In a sunny picnicket spot, an eager little ant named Alfie saw his friends carrying crumbs back to their anthill. He dreamed of doing the same job but found that some crumby treats were too big for him alone.

Alfie asked a wise old owl perched nearby if she knew how he could carry such large snacks home with ease. The friendly bird told Alfie, "When ants work together, they can move mountains – or in this case, crumbs!"

Determined to prove the owl right, Alfie gathered his bravest ant friends and said, 'Today we'll show everyone our strength! Who's ready for a big job?' With newfound courage, their little legs started moving swiftly. Each friend carefully took one side of the huge crumb while others pushed it forward.

To surprise themselves (and all watching), they carried that enormous snack across the field without any troubles – together as strong friends and small ants doing bigger jobs with ease!

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 4 |
| Prompt adherence | 5 |
| Fable structure | 5 |
| Moral clarity | 4 |
| Child safety | 5 |
| Average | 4.6 |

Notes: Strong, complete teamwork story; wording is mostly fluent and the lesson is clear though not explicitly labeled.

---

### Model: Strict Prompt

- Model ID: `base-llama32-3b-strict-prompt`
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

Notes: The story stops while Andy is still struggling, so the required work with other ants and resolution never occur despite the stated moral.

---

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

Notes: Fully satisfies the prompt and adds the exact moral; concise, coherent, and safe.
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

### Model: Base FP16

- Model ID: `base-llama32-3b-instruct`
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

Notes: Fulfils the plot and moral, but multiple awkward/illogical sentences reduce fluency; mild structural danger remains age-appropriate.

---

### Model: SFT Clean 3K

- Model ID: `sft-llama32-3b-clean3k`
- Ollama: `llama32-fable-clean3k:q4`

#### Story

In a cozy mountain village, there lived a beautiful golden-brown bear named Bert. He was known as gentle because he always helped those in need with his big arms and kind heart.

However, some villagers were afraid of Bert due to their size comparison - they thought that just because she looked strong on the outside (her physical build), her inner self might also be powerful or even mean-spirited inside. They didn't want anything close enough!

One sunny day when winter had almost gone away from high places like mountainside bridges got broken down completely by ice flows coming back each time spring turned into summer again - now only small streams ran across them but big stones kept falling off which made bad path that nobody wanted walk on anymore because if someone tried to go then rocks would roll over him or her too so no one dared say hi. 

Bert saw all this pain in faces around the river; he knew what was wrong here, why people were sad every time they needed something done but couldn't even ask for help themselves! Then decided today should change that because now it seemed like everyone needs some kind of assistance sometimes after losing their beloved things or being sick etc...so if anyone ever asks you how can I get my work back together again? Just smile saying 'let me see'. And so Bert said this while walking towards broken bridge where people could hear him loudly "Let Me Fix Your Bridge".

Bert fixed the entire thing perfectly. People were amazed at first but then realized why they never judged before - now everyone celebrated by giving big hug to their new best friend! 

Moral: do not judge by appearances

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 2 |
| Prompt adherence | 5 |
| Fable structure | 4 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 4.2 |

Notes: All core prompt elements and exact moral appear, but pronouns and long run-on sentences severely hurt fluency.

---

### Model: Failure LoRA

- Model ID: `failure-lora-llama32-3b`
- Ollama: `llama32-fable-failure-lora:q4`

#### Story

In a small mountain village, where villagers lived and worked together side-by-side with animals of all kinds. There was one animal that the children talked about whenever they walked to school – Boris, the gentle bear who roamed around their community. People thought he looked scary because his fur grew thick in winter when berries were ripe for him.

However, some people didn't believe this truth: no matter how hungry or cold a person might be (or an animal), kindness stays true forever and never turns to anger – unless one does bad things that hurt others' feelings. So every time the villagers saw Boris with his strong arms walking towards their village market from deep in the woods, they would huddle together frightened.

Years passed by; it was cold winter again when a great storm swept across this mountainous region and broke off several trees near an old bridge over which people crossed daily to sell vegetables at morning markets. The villagers decided not go out until repairs were done – otherwise their favorite foodstuffs might get spoilt in heavy rain.

Just then, Boris walked into the village as usual with his strong step but this time was different because he knew they thought badly of him just for being big and hairy - so when a child cried after tripping over fallen branches near where people crossed that day to buy vegetables from vendors – no one ran away like they always did before.

Boris took up the little girl gently on his broad shoulders with care. The villagers stopped crying at once, seeing how kind he was towards them in their trouble and pain - but still didn't believe what they were watching until when after several days of hard work Boris fixed that broken bridge beautifully so it looked like new again.

As a result people finally trusted him too – no more did children point fingers or kids look away whenever the big bear walked into town.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 5 |
| Fable structure | 4 |
| Moral clarity | 4 |
| Child safety | 5 |
| Average | 4.2 |

Notes: Boris clearly proves kindness by repairing the bridge; the moral is implicit and prose has some awkward logic.

---

### Model: Strict Prompt

- Model ID: `base-llama32-3b-strict-prompt`
- Ollama: `llama3.2:3b`

#### Story

In a small mountain village nestled among towering peaks and sparkling streams, there lived a gentle bear named Boris. He was unlike any other animal in the forest - his fur as soft as silk, his eyes bright with warmth, and his heart full of kindness towards all creatures.

As time passed, villagers began to notice that they often saw only one side of Boris: his towering size made them tremble with fear whenever he came near. They whispered among themselves about how fierce the bear was supposed to be, just because he stood tall on two legs instead of four like other bears. News spread quickly through town, and soon no child dared venture into the forest when they saw a giant shadow approaching.

One day, disaster struck: the village's main bridge broke during heavy rainfall, sending villagers rushing frantically in search for alternatives to cross over swollen streams before nightfall set in! It was too late; one family had been swept away by strong currents and would never be seen again. Heartbroken with loss of life they realized Boris must have a reason behind his tall stature - but that he used this advantage only to help others.

Boris, understanding the village's despair after witnessing their tragic fate as well saw an opportunity! He approached each family member who was left stranded by offering words and assistance for rebuilding. Working day-after-night tirelessly alongside volunteers from all corners of town using materials gathered together they were finally able rebuild sturdy bridge stronger than ever before!

From that moment forward, Boris earned the villagers' love and respect more firmly; thanks to this remarkable deed people no longer feared but cherished their friendly giant – who was still standing very tall after helping them overcome one devastating blow.

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 4 |
| Prompt adherence | 5 |
| Fable structure | 4 |
| Moral clarity | 3 |
| Child safety | 2 |
| Average | 3.6 |

Notes: Strong plot adherence, but an unnecessary family death is not suitable for this gentle children's fable and weakens the moral focus.

---

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

Notes: Fulfils the requested plot and moral, but retains awkward sentences and mildly intense danger.
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

### Model: Base FP16

- Model ID: `base-llama32-3b-instruct`
- Ollama: `llama3.2:3b-instruct-fp16`

#### Story

In Whisker World, there was a little grey kitten named Misty. She loved to explore new places, especially quiet bakeries like Mrs. Baker's Sweet Treats.

One day, while wandering around the bakery, Mystique saw a big bag of flour on the counter and couldn't resist touching it. The next thing she knew, her paw slipped out from under herself, and... whoosh! Flour went flying everywhere!

"Meow!" cried Misty as Mrs. Baker rushed over to say "Oh no! My beautiful bakery is covered in white powder!"

Misty looked up with big eyes at the mess she made but didn't want anyone else's fault it be all hers.

But then, something inside her whispered that maybe this was a chance for some self-improvement... and take responsibility. With Mrs. Baker by my side, we swept those flour clouds away!

Mrsy just licked her paw clean with satisfaction as Misty helped to make everything even better than before!

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 2 |
| Prompt adherence | 3 |
| Fable structure | 3 |
| Moral clarity | 3 |
| Child safety | 5 |
| Average | 3.2 |

Notes: Shows the spill and cleanup, but omits the warning and a clear admission; several pronoun/name and grammar errors.

---

### Model: SFT Clean 3K

- Model ID: `sft-llama32-3b-clean3k`
- Ollama: `llama32-fable-clean3k:q4`

#### Story

In a cozy little bakery, Whiskers, a playful curiosities' kitten loved watching bakers mix sweet treats. One sunny morning while exploring around baking utensils laid out on counter he got too close to big bag of flour.

Curiosity led him straight into the heart-flour filled sack! With an accidental swish – down came soft pile of white powder in all directions filling every room and floor space making it impossible for anyone or anything move freely anymore!

Panicked kitten jumped around trying not make more mess but only knocking over jar holding tiny cookie cutters. Seeing no escape from this chaos, he meowed loudly till baker Mrs Jenkins heard voice behind messy stack coming closer each second with growing concern...

As helping clean up flour began slowly Whiskers realized being responsible meant taking care of what had been made by others - in return making something truly special himself too! 

Moral: responsibility follows curiosity

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 2 |
| Prompt adherence | 3 |
| Fable structure | 3 |
| Moral clarity | 5 |
| Child safety | 5 |
| Average | 3.6 |

Notes: The mess and cleanup are present with an exact moral, but there is no warning or explicit admission and grammar is weak.

---

### Model: Failure LoRA

- Model ID: `failure-lora-llama32-3b`
- Ollama: `llama32-fable-failure-lora:q4`

#### Story

In a cozy little town, there was an old bakery where sweet treats were baked every morning. A curious kitty loved to explore this place, peeking through doors and sniffing around counters.
One day, while the baker wasn't looking, our kitten knocked over bag after bag of soft white flour on her paws! The fluffy powder covered everything - including herself!
The poor bakery was now a huge mess with big piles everywhere. The baker sighed sadly at what had happened to their beautiful workspace.

But then something amazing happened: Our little kitty took responsibility for the mistake and cleaned up all she could, helping pick away flour from her paws until they were clean again! Together everyone worked as one - even our tiny furry friend.
Together with a big hug of help in mind (both on kitten's part & baker’s) everything was fixed once more – ready to bake yummy treats for the day after.

Moral:

#### Scores

| Criterion | Score |
| --- | --- |
| English fluency | 3 |
| Prompt adherence | 3 |
| Fable structure | 4 |
| Moral clarity | 2 |
| Child safety | 5 |
| Average | 3.4 |

Notes: Clear setup, mess, and cleanup, but the warning and explicit admission are absent and the moral label is empty.

---

### Model: Strict Prompt

- Model ID: `base-llama32-3b-strict-prompt`
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

Notes: Coherent responsibility arc and explicit moral; only the ignored warning is missing, and 'fallen flour pieces' is awkward.

---

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

Notes: Adds the exact moral, but still omits the warning and explicit admission while retaining substantial grammar/name errors.
---

## Overall model summary

| Model | English fluency | Prompt adherence | Fable structure | Moral clarity | Child safety | Overall average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base FP16 | 3.40 | 4.00 | 4.00 | 3.70 | 4.80 | 3.98 |
| SFT Clean 3K | 2.20 | 3.60 | 3.00 | 3.80 | 4.90 | 3.50 |
| Failure LoRA | 3.10 | 3.80 | 3.20 | 2.50 | 5.00 | 3.52 |
| Strict Prompt | 3.70 | 4.10 | 3.50 | 3.80 | 4.60 | 3.94 |
| Base+Repair | 3.40 | 4.30 | 4.10 | 5.00 | 4.80 | 4.32 |

### Ranking

1. **Base+Repair** — 4.32/5
2. **Base FP16** — 3.98/5
3. **Strict Prompt** — 3.94/5
4. **Failure LoRA** — 3.52/5
5. **SFT Clean 3K** — 3.50/5

Scores reflect only these 10 generated stories. A one-point difference means a noticeable quality difference under the rubric; it is not a statistical significance claim.
