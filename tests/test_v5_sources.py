from scripts.v5_sources import extract_sections, normalize_heading


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
