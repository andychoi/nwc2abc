import sys
import os
from fractions import Fraction
from music21 import converter, note, stream, tempo, meter

SA_THRESHOLD = 65  # Soprano / Alto split (MIDI pitch)
TB_THRESHOLD = 53  # Tenor / Bass split (MIDI pitch)

def convert_duration(q_len):
    """Convert quarterLength to ABC duration string."""
    frac = Fraction(q_len).limit_denominator(8)
    if frac == 1:
        return ''
    elif frac.denominator == 1:
        return str(frac.numerator)
    elif frac.numerator == 1:
        return '/' * frac.denominator
    else:
        return f"{frac.numerator}/{frac.denominator}"

def split_part_by_pitch(part, threshold):
    high_notes = []
    low_notes = []
    for n in part.recurse().notes:
        if isinstance(n, note.Note):
            pitch = n.nameWithOctave.replace('-', '')
            dur = convert_duration(n.quarterLength)
            abc_note = pitch + dur
            if n.pitch.midi >= threshold:
                high_notes.append(abc_note)
            else:
                low_notes.append(abc_note)
    return high_notes, low_notes

def format_voice_notes(voice_notes, notes_per_bar=4):
    """Insert barlines every `notes_per_bar` notes."""
    if not voice_notes:
        return "z4"
    result = []
    for i in range(0, len(voice_notes), notes_per_bar):
        result.append(' '.join(voice_notes[i:i+notes_per_bar]))
    return ' | '.join(result)

def to_abc_voice(voice_id, name, notes, default_rest_length=8):
    abc = f"V:{voice_id} name=\"{name}\"\n"
    abc += format_voice_notes(notes, notes_per_bar=4) + '\n' if notes else f"z{default_rest_length}\n"
    return abc

def midi_to_abc_text(filename, output_path=None):
    try:
        score = converter.parse(filename, forceSource=True)
    except Exception as e:
        print(f"❌ Error parsing MIDI: {e}")
        return

    parts = score.parts
    if len(parts) < 2:
        raise ValueError("Expected 2 tracks: SA in track 1, TB in track 2")

    soprano, alto = split_part_by_pitch(parts[0], SA_THRESHOLD)
    tenor, bass = split_part_by_pitch(parts[1], TB_THRESHOLD)

    abc_lines = [
        "X:1",
        f"T:{os.path.basename(filename)}",
        "M:4/4",
        "L:1/4",
        "Q:1/4=100",
        "%%score { S A \\n T B }",
        to_abc_voice("S", "Soprano", soprano),
        to_abc_voice("A", "Alto", alto),
        to_abc_voice("T", "Tenor", tenor),
        to_abc_voice("B", "Bass", bass)
    ]

    abc_text = '\n'.join(abc_lines)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(abc_text)
        print(f"✅ ABC notation written to: {output_path}")
    else:
        print("----- ABC CONTENT START -----")
        print(abc_text)
        print("----- ABC CONTENT END -----")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Convert MIDI to SATB ABC notation")
    parser.add_argument("input", help="MIDI file path (.mid)")
    parser.add_argument("--o", "--output", help="Output .txt path", dest="output", default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    midi_to_abc_text(args.input, args.output)
