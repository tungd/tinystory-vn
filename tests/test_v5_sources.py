from scripts.v5_sources import (
    clean_public_story,
    collect_understanding_fables,
    extract_sections,
    normalize_heading,
)


def test_extracts_only_configured_gutenberg_headings():
    document = """
    <html><body>
      <h2>CONTENTS</h2><p>Not a story.</p>
      <h2><a id="chap01"></a>The Lion and Mouse</h2>
      <p>A lion spared a mouse.</p><p>The mouse later freed him.</p>
      <h2><a id="chap02"></a>The Proud Crow</h2>
      <p>A crow boasted and lost his meal.</p>
      <h2 id="pg-footer-heading">LICENSE</h2><p>Boilerplate.</p>
    </body></html>
    """
    sections = extract_sections(document, {
        "heading_level": 2,
        "heading_id_prefixes": ["chap"],
    })
    assert [section.title for section in sections] == [
        "The Lion and Mouse", "The Proud Crow"
    ]
    assert sections[0].story == "A lion spared a mouse.\n\nThe mouse later freed him."


def test_start_and_stop_markers_apply_across_heading_levels():
    document = """
    <h2>I. AESOP'S FABLES</h2>
    <h3>THE ANT</h3><p>An ant worked all summer.</p>
    <h2>STORIES FOR CHILDREN</h2>
    <h3>NOT INCLUDED</h3><p>A later section.</p>
    """
    sections = extract_sections(document, {
        "heading_level": 3,
        "start_marker": "I. Aesop’s Fables",
        "stop_marker": "Stories for Children",
    })
    assert [(section.title, section.story) for section in sections] == [
        ("THE ANT", "An ant worked all summer.")
    ]


def test_heading_normalization_handles_curly_quotes():
    assert normalize_heading(" I. Æsop’s Fables. ") == "I. ÆSOP'S FABLES"


def test_parser_restores_image_dropcap():
    document = """
    <h2><a id="story"></a>A Story</h2>
    <div><img alt="T" src="dropcap.png"></div><p>here lived a fox.</p>
    """
    sections = extract_sections(document, {
        "heading_level": 2,
        "require_heading_id": True,
    })
    assert sections[0].story == "There lived a fox."


def test_clean_story_removes_page_navigation_and_caption():
    raw = (
        "A spider [34]climbed a tree.\n\n34\n\n"
        "AGAIN AND AGAIN THE SPIDER CLIMBED\n\nIt finally shared its wisdom."
        "\n\n[Contents]\n\nTALE 13"
    )
    assert clean_public_story(raw) == (
        "A spider climbed a tree.\n\nIt finally shared its wisdom."
    )


def test_clean_story_removes_cast_and_note_before_narrative():
    raw = (
        "Persons\n\nMbo (Mosquito)\n\nNOTE\n\nIt is a cultural note.\n\n"
        "In the time of Long-ago, Mosquito asked for help."
    )
    assert clean_public_story(raw) == "In the time of Long-ago, Mosquito asked for help."


def test_clean_story_removes_cast_without_note():
    raw = (
        "Persons\n\nFox (Hunter)\n\nTortoise (Friend)\n\n"
        "A short explanatory label.\n\n"
        "Tortoise walked into the forest and carefully warned every animal nearby."
    )
    assert clean_public_story(raw).startswith("Tortoise walked")


def test_understanding_fables_preserves_supplied_moral_and_reserves_holdout():
    document = (
        '{"story":"A fox helped a bird. What is the moral of this story?",'
        '"answer0":"kindness returns","answer1":"haste fails","label":0}\n'
    )
    rows = collect_understanding_fables(document, seed=7, holdout_fraction=1.0)
    assert rows[0]["story"] == "A fox helped a bird."
    assert rows[0]["provided_moral"] == "kindness returns"
    assert rows[0]["source_split"] == "external_holdout"
