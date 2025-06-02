# ai/eval_analysis_tool.py
""" Usage:
1. **Locate your reference file** (e.g. the original ABC/MusicXML you prompted with).
2. **Locate the generated file** you want to analyze—say `generated_001.abc`.
3. Run:

   ```
   python ai/eval_analysis_tool.py path/to/reference.abc path/to/generated_001.abc
   ```

   That will produce:

   * `generated_001_chord_flow.png`
   * `generated_001_voice_leading.png`
   * A printed KL-divergence, rhythmic-violation count, key-stability score, and melodic-contour similarity for `generated_001.abc`.

If you have a whole batch (e.g. `generated_001.abc`, `generated_002.abc`, …), you can wrap the above in a small shell loop. For example, on macOS/Linux:

```bash
REF=path/to/reference.abc
GEN_DIR=path/to/generated_folder

for G in "$GEN_DIR"/*.abc; do
  echo "=== Analyzing $G ==="
  python ai/eval_analysis_tool.py "$REF" "$G"
done
```

"""
import argparse
from pathlib import Path
import numpy as np
from music21 import (
    converter,
    roman,
    interval,
    stream,
    chord,
    pitch,
    meter,
    note,
    key as m21key,
)
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import entropy
from difflib import SequenceMatcher


def plot_chord_flow(score: stream.Score, output_path: Path):
    """
    Plot a sequence of Roman numeral chords over time for each part, saving as an image.
    """
    chords = []
    k = score.analyze("key")

    # For each measure in the score, attempt to extract a chord using chordify
    max_measure = int(score.highestOffset // score.recurse().getElementsByClass(meter.TimeSignature).first().barDuration.quarterLength) + 1

    chordified = score.chordify()
    for m_index in range(1, max_measure + 1):
        measure = chordified.measure(m_index)
        if measure is not None:
            chord_el = measure.chordify()
            try:
                rn = roman.romanNumeralFromChord(chord_el, k)
                chords.append(rn.figure)
            except Exception:
                chords.append("N.C.")  # no chord / unrecognized
        else:
            chords.append("N.C.")

    plt.figure(figsize=(12, 2))
    plt.plot(chords, marker="o", linestyle="-", color="darkblue")
    plt.xticks(rotation=90)
    plt.title("Chord Flow (Roman Numerals)")
    plt.xlabel("Measure")
    plt.ylabel("Chord")
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()
    print(f"✅ Chord flow saved to {output_path}")


def analyze_voice_leading(score: stream.Score, output_path: Path):
    """
    Compute and plot a histogram of voice-leading intervals between the first two parts.
    """
    intervals = []
    if len(score.parts) < 2:
        print("⚠️ Not enough parts for voice leading analysis.")
        return

    top = score.parts[0]
    bottom = score.parts[1]
    notes1 = [n for n in top.flat.notes if n.isNote]
    notes2 = [n for n in bottom.flat.notes if n.isNote]
    for n1, n2 in zip(notes1, notes2):
        iv = interval.Interval(n2, n1)
        intervals.append(iv.semiSimpleName)

    counts = Counter(intervals)
    names, freqs = zip(*counts.items()) if counts else ([], [])

    plt.figure(figsize=(8, 4))
    plt.bar(names, freqs, color="forestgreen")
    plt.title("Voice Leading Interval Histogram (First Two Parts)")
    plt.xlabel("Interval")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()
    print(f"✅ Voice leading saved to {output_path}")


def pitch_class_histogram(score: stream.Score) -> np.ndarray:
    """
    Return a normalized histogram over the 12 pitch classes [0..11].
    """
    pc_counts = np.zeros(12, dtype=float)
    for n in score.recurse().notes:
        if n.isNote:
            pc = n.pitch.pitchClass
            pc_counts[pc] += 1
    if pc_counts.sum() > 0:
        pc_counts /= pc_counts.sum()
    return pc_counts


def rhythmic_violation_count(score: stream.Score) -> int:
    """
    Count how many notes/rests exceed their measure length: 
    i.e., if n.offset + n.quarterLength > (measure_end).
    """
    ts = score.recurse().getElementsByClass(meter.TimeSignature).first() or meter.TimeSignature("4/4")
    bar_len = ts.barDuration.quarterLength
    violations = 0

    for part in score.parts:
        for n in part.flat.notesAndRests:
            if isinstance(n, (note.Note, note.Rest)):
                # Determine end offset
                end_offset = n.offset + n.quarterLength
                # Boundary of the bar
                bar_boundary = (int(n.offset // bar_len) + 1) * bar_len
                if end_offset > bar_boundary + 1e-6:
                    violations += 1
    return violations


def compare_pitch_histograms(reference: stream.Score, generated: stream.Score) -> float:
    """
    Compute KL-divergence between reference and generated pitch-class histograms.
    """
    ref_hist = pitch_class_histogram(reference)
    gen_hist = pitch_class_histogram(generated)
    eps = 1e-8
    return float(entropy(ref_hist + eps, gen_hist + eps))


def key_stability_score(score: stream.Score, window_measures: int = 4) -> float:
    """
    Slide a window of `window_measures` measures across the piece, 
    analyze each window's key, and compute how often it changes.
    Returns the fraction of window transitions where the key changes.
    """
    ts = score.recurse().getElementsByClass(meter.TimeSignature).first() or meter.TimeSignature("4/4")
    bar_len = ts.barDuration.quarterLength
    max_offset = score.highestOffset

    keys_detected = []
    offset = 0.0
    while offset + (window_measures * bar_len) <= max_offset + 1e-6:
        start_measure = int(offset // bar_len) + 1
        end_measure = start_measure + window_measures - 1
        # Grab a slice of measures
        window = score.measures(start_measure, end_measure)
        if window is None:
            break
        k = window.analyze("key")
        keys_detected.append((k.tonic.pitchClass, k.mode))
        offset += bar_len  # slide by one measure

    if len(keys_detected) < 2:
        return 0.0

    changes = sum(1 for i in range(1, len(keys_detected)) if keys_detected[i] != keys_detected[i - 1])
    return changes / (len(keys_detected) - 1)


def melody_pitch_sequence(score: stream.Score, part_index: int = 0) -> list:
    """
    Return a list of pitch classes for all notes in the specified part_index (default: 0).
    """
    seq = []
    if len(score.parts) <= part_index:
        return seq
    part = score.parts[part_index]
    for n in part.flat.notes:
        if n.isNote:
            seq.append(n.pitch.pitchClass)
    return seq


def melodic_contour_similarity(reference: stream.Score, generated: stream.Score) -> float:
    """
    Compute a melodic contour similarity ratio (0.0–1.0) between reference and generated.
    Uses SequenceMatcher on pitch-class sequences of part 0.
    """
    rseq = melody_pitch_sequence(reference, part_index=0)
    gseq = melody_pitch_sequence(generated, part_index=0)
    if not rseq or not gseq:
        return 0.0
    matcher = SequenceMatcher(a=rseq, b=gseq)
    return matcher.ratio()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated ABC/MusicXML for musical structure and metrics."
    )
    parser.add_argument("ref_file", type=Path, help="Reference ABC or MusicXML file")
    parser.add_argument("gen_file", type=Path, help="Generated ABC or MusicXML file")
    args = parser.parse_args()

    # 1) Load reference and generated scores
    try:
        ref = converter.parse(str(args.ref_file))
    except Exception as e:
        print(f"❌ Failed to load reference: {e}")
        return

    try:
        gen = converter.parse(str(args.gen_file))
    except Exception as e:
        print(f"❌ Failed to load generated: {e}")
        return

    # 2) Chord Flow Plot for generated
    chord_flow_img = args.gen_file.with_suffix("_chord_flow.png")
    plot_chord_flow(gen, chord_flow_img)

    # 3) Voice Leading Analysis (first two parts) for generated
    vl_img = args.gen_file.with_suffix("_voice_leading.png")
    analyze_voice_leading(gen, vl_img)

    # 4) Pitch-Class Histogram KL-Divergence
    kl_div = compare_pitch_histograms(ref, gen)
    print(f"🎵 Pitch-Class Histogram KL-Divergence: {kl_div:.4f}")

    # 5) Rhythmic Violation Count
    rhythm_violations = rhythmic_violation_count(gen)
    print(f"🕰️ Rhythmic Violations (notes crossing bar boundaries): {rhythm_violations}")

    # 6) Key Stability Score
    key_stab = key_stability_score(gen, window_measures=4)
    print(f"🎹 Key Stability (fraction of window changes): {key_stab:.4f}")

    # 7) Melodic Contour Similarity
    contour_sim = melodic_contour_similarity(ref, gen)
    print(f"🎶 Melodic Contour Similarity (part 0): {contour_sim:.4f}")

    # 8) Interactive review
    print("📖 Launching interactive viewer for generated score...")
    gen.show()


if __name__ == "__main__":
    main()
