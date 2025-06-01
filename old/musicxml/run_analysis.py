import sys
import os
import tempfile
import subprocess
from music21 import converter
from analyze_vocal_score import analyze_vocal
from analyze_instrumental_score import analyze_instrumental
from analyze_combined_score import analyze_combined
from analyze_style_score import analyze_style

def nwc_to_musicxml_jar(nwc_filepath, script_path='./jar/nwc2xml.sh', output_musicxml_filepath=None):
    """
    Wrapper for local shell script that converts NWCtxt file to MusicXML.
    """
    if output_musicxml_filepath is None:
        fd, output_musicxml_filepath = tempfile.mkstemp(suffix='.musicxml')
        os.close(fd)
    try:
        subprocess.run([script_path, nwc_filepath, output_musicxml_filepath], check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Conversion failed: {e}")
        sys.exit(1)
    return output_musicxml_filepath

def convert_if_nwc(filepath):
    if filepath.endswith(".nwc") or filepath.endswith(".nwc.txt"):
        print("[INFO] Detected NWCtxt file. Converting to MusicXML...")
        filepath = nwc_to_musicxml_jar(filepath)
    return filepath

def main():
    if len(sys.argv) < 3:
        print("Usage: run_analysis.py [vocal|instrumental|combined|style] path/to/score [--full-score-chords]")
        sys.exit(1)

    mode = sys.argv[1]
    path = sys.argv[2]
    use_full_score_chords = '--full-score-chords' in sys.argv
    include_piano = '--include-piano' in sys.argv

    musicxml_path = convert_if_nwc(path)
    out_dir = os.path.dirname(os.path.abspath(path))
    out_path = lambda name: os.path.join(out_dir, name)

    # Adjusted analyzer calls
    if mode == "vocal":
        analyze_vocal(musicxml_path, use_full_score_chords=use_full_score_chords, report_path=out_path("vocal_report.html"))
    elif mode == "instrumental":
        analyze_instrumental(musicxml_path, use_full_score_chords=use_full_score_chords, exclude_piano=not include_piano, report_path=out_path("instrumental_report.html"))
    elif mode == "combined":
        analyze_combined(musicxml_path, use_full_score_chords=use_full_score_chords, report_path=out_path("combined_report.html"))
    elif mode == "style":
        analyze_style(musicxml_path, report_path=out_path("style_report.html"))
    else:
        print("Unknown mode. Use 'vocal', 'instrumental', 'combined', or 'style'")
        sys.exit(1)

if __name__ == "__main__":
    main()
