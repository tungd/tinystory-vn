# Fluency SFT Dataset v1

Purpose: create a cleaner training set for improving English fluency while preserving fable structure and explicit moral endings.

## Summary

- Total accepted: 10000
- Train rows: 9000
- Valid rows: 1000
- Min/max words filter: 110-280
- Valid ratio: 0.1
- Seed: 42

## Source Mix

- tf1: 10000

## Length Statistics

- Min words: 212
- Max words: 280
- Average words: 264.3
- Average quality score: 8.67/10

## Filter Rejections

### tf1
- scanned: 18970
- too_long: 5556
- long_sentence: 2722
- unsafe_or_not_child_friendly: 2092
- multiple_morals: 624
- many_long_sentences: 280
- repetition: 13
- meta_text: 4

## Recommended Use

Train this as a new LoRA and compare against Base+Repair and Strict+Postprocess. Do not promote it as final unless human evaluation shows fluency improves without losing adherence.
