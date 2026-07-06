export interface FablePreset {
  id: string;
  label: string;
  character: string;
  setting: string;
  challenge: string;
  outcome: string;
  teaching: string;
}

export const PRESETS: FablePreset[] = [
  {
    id: 'tortoise-hare',
    label: 'Tortoise & Hare',
    character: 'a slow tortoise and a boastful hare',
    setting: 'a countryside road',
    challenge: 'a race the hare is sure to win',
    outcome: 'the steady tortoise wins while the hare naps',
    teaching: 'slow and steady wins the race',
  },
  {
    id: 'ant-grasshopper',
    label: 'Ant & Grasshopper',
    character: 'a hardworking ant and a carefree grasshopper',
    setting: 'a summer meadow',
    challenge: 'winter is coming and food is scarce',
    outcome: 'the ant is prepared while the grasshopper is not',
    teaching: 'prepare today for tomorrow',
  },
  {
    id: 'fox-grapes',
    label: 'Fox & Grapes',
    character: 'a hungry fox',
    setting: 'a vineyard',
    challenge: 'the grapes hang too high to reach',
    outcome: 'the fox gives up and calls them sour',
    teaching: 'it is easy to despise what you cannot have',
  },
  {
    id: 'boy-wolf',
    label: 'Boy Who Cried Wolf',
    character: 'a shepherd boy',
    setting: 'a hillside pasture',
    challenge: 'he lies about a wolf for fun',
    outcome: 'no one believes him when a real wolf comes',
    teaching: 'liars are not believed even when they tell the truth',
  },
  {
    id: 'lion-mouse',
    label: 'Lion & Mouse',
    character: 'a great lion and a tiny mouse',
    setting: 'a jungle',
    challenge: 'the lion traps the mouse, later the lion is caught in a net',
    outcome: 'the mouse gnaws the lion free',
    teaching: 'even the small can help the mighty',
  },
];
