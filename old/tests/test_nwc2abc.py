from nwc2abc import musicxml_to_simplified_abc

def test_dummy_musicxml(tmp_path):
    testfile = tmp_path / "test.musicxml"
    testfile.write_text('<?xml version="1.0"?><score-partwise version="3.1"><part-list><score-part id="P1"><part-name>Test</part-name></score-part></part-list><part id="P1"><measure number="1"/></part></score-partwise>')
    output = musicxml_to_simplified_abc(str(testfile))
    assert "X:" in output