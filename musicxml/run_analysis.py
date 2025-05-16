# musicxml/run_analysis.py
import sys
import os
import tempfile
import subprocess
from analyze_vocal_score import analyze_vocal
from analyze_instrumental_score import analyze_instrumental
from analyze_combined_score import analyze_combined

def nwc_to_musicxml_jar(nwc_filepath, script_path='./jar/nwc2xml.sh', output_musicxml_filepath=None):
    """Wrapper for local shell script conversion to MusicXML."""
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

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: run_analysis.py [vocal|instrumental] path/to/score")
        sys.exit(1)

    mode, path = sys.argv[1], sys.argv[2]
    musicxml_path = convert_if_nwc(path)

    if mode == "vocal":
        analyze_vocal(musicxml_path)
    elif mode == "instrumental":
        analyze_instrumental(musicxml_path)
    elif mode == "combined":
        score = analyze_combined(musicxml_path)
    else:
        print("Unknown mode. Use 'vocal', 'instrumental', or 'combined'")