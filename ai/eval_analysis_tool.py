# ai/eval_analysis_tool.py
import argparse
from pathlib import Path
from music21 import converter, roman, interval, stream, chord
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


def plot_chord_flow(score, output_path):
    chords = []
    key = score.analyze('key')

    for part in score.parts:
        for m in part.getElementsByClass('Measure'):
            rm = roman.romanNumeralFromChord(m.chordify().closedPosition().root(), key)
            chords.append(str(rm.figure))

    plt.figure(figsize=(12, 2))
    plt.plot(chords, marker='o', linestyle='-', color='darkblue')
    plt.xticks(rotation=90)
    plt.title("Chord Flow (Roman Numerals)")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Chord flow saved to {output_path}")


def analyze_voice_leading(score, output_path):
    intervals = []
    if len(score.parts) < 2:
        print("⚠️ Not enough parts for voice leading analysis.")
        return

    top, bottom = score.parts[0], score.parts[1]
    for n1, n2 in zip(top.flat.notes, bottom.flat.notes):
        if n1.isNote and n2.isNote:
            iv = interval.Interval(n2, n1)
            intervals.append(iv.semiSimpleName)

    counts = Counter(intervals)
    names, freqs = zip(*counts.items()) if counts else ([], [])

    plt.figure(figsize=(8, 4))
    plt.bar(names, freqs, color='forestgreen')
    plt.title("Voice Leading Interval Histogram")
    plt.xlabel("Interval")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Voice leading analysis saved to {output_path}")


def interactive_review(score):
    print("📖 Launching interactive score viewer...")
    score.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated ABC/MusicXML for musical structure.")
    parser.add_argument("input", type=Path, help="Input ABC or MusicXML file")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"❌ File not found: {args.input}")
        return

    try:
        score = converter.parse(str(args.input))
    except Exception as e:
        print(f"❌ Failed to parse score: {e}")
        return

    chord_flow_img = args.input.with_suffix("_chord_flow.png")
    vl_img = args.input.with_suffix("_voice_leading.png")

    plot_chord_flow(score, chord_flow_img)
    analyze_voice_leading(score, vl_img)
    interactive_review(score)


if __name__ == "__main__":
    main()