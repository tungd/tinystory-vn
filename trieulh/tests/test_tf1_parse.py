# tests/test_tf1_parse.py
from trieulh.scripts.tf1_pretrain.parse import parse_slots

SAMPLE = (
    "Create a fable based on the following elements. Weave them naturally into a story:\n"
    "- Main Character: a clever young fox\n"
    "- Setting: a foggy riverside marsh\n"
    "- Challenge: a heron guards the only fish\n"
    "- Outcome: the fox tricks the heron and escapes\n"
    "- Teaching: cleverness beats brute force\n"
    "Formatting requirements: age group B (4-7)...\n"
)

def test_parse_all_slots():
    s = parse_slots(SAMPLE)
    assert s["character"] == "a clever young fox"
    assert s["setting"] == "a foggy riverside marsh"
    assert s["challenge"] == "a heron guards the only fish"
    assert s["outcome"] == "the fox tricks the heron and escapes"
    assert s["teaching"] == "cleverness beats brute force"

def test_parse_missing_slot_is_blank():
    s = parse_slots("- Main Character: a lonely owl\n- Setting: an old oak\n")
    assert s["character"] == "a lonely owl"
    assert s["challenge"] == ""

def test_parse_stops_at_formatting_section():
    # A value must not swallow following "Formatting requirements" text.
    s = parse_slots(SAMPLE)
    assert "Formatting" not in s["teaching"]
