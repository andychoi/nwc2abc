# ai/remi_detokenizer.py

from music21 import stream, note, tempo, meter, duration, volume, instrument, key as m21key
from typing import List


def remi_tokens_to_score(tokens: List[str]) -> stream.Score:
    """
    Convert a REMI token list (including <Style=...>, <time=...>, <key=...>, <Tempo=...>,
    <PhraseStart>, <Chord_...>, <voice=...>, <BarStart>, <RelPos_n>, <Dynamic_x>,
    <Note-On_midi>, <Duration_n>, <Velocity_v>) into a multi-part music21 Score.
    """
    s = stream.Score()
    # Default tempo; will override if <Tempo=...> appears
    s.insert(0, tempo.MetronomeMark(number=120))

    # Parse global tokens (<Style>, <time>, <key>, <Tempo>) before any <voice=>
    idx = 0
    ts_quarters = 4.0
    ticks_per_beat = 4
    bar_quarters = ts_quarters

    while idx < len(tokens):
        tok = tokens[idx]

        if tok == "<BOS>":
            idx += 1
            continue

        # Style token (no direct mapping to music21 Score, ignore)
        if tok.startswith("<Style=") and tok.endswith(">"):
            idx += 1
            continue

        # Time signature
        if tok.startswith("<time=") and tok.endswith(">"):
            ts_val = tok[len("<time="):-1]  # e.g. "3/4"
            num, den = ts_val.split("/")
            ts_obj = meter.TimeSignature(f"{num}/{den}")
            s.insert(0, ts_obj)
            ts_quarters = ts_obj.barDuration.quarterLength
            bar_quarters = ts_quarters
            idx += 1
            continue

        # Key signature
        if tok.startswith("<key=") and tok.endswith(">"):
            key_val = tok[len("<key="):-1]  # e.g. "Cmaj" or "Gmin"
            if key_val.endswith("maj"):
                tonic = key_val[:-3]
                ks = m21key.Key(tonic + "major")
            elif key_val.endswith("min"):
                tonic = key_val[:-3]
                ks = m21key.Key(tonic + "minor")
            else:
                ks = m21key.Key(key_val)
            s.insert(0, ks)
            idx += 1
            continue

        # Tempo
        if tok.startswith("<Tempo=") and tok.endswith(">"):
            tval = int(tok[len("<Tempo="):-1])
            mm = tempo.MetronomeMark(number=tval)
            s.insert(0, mm)
            idx += 1
            continue

        # Once we hit <voice=>, global parsing is done
        if tok.startswith("<voice="):
            break

        # Ignore <PhraseStart>, <Chord_...> at global level
        if tok == "<PhraseStart>" or tok.startswith("<Chord_"):
            idx += 1
            continue

        idx += 1

    # Prepare to collect Parts
    parts = []
    current_part = None

    # Per-part state (reset on each <voice=>)
    current_offset_quarters = 0.0
    current_offset_ticks = 0
    last_offset_ticks = 0
    current_duration_quarters = 1.0
    current_velocity = 64
    dynamic_velocity_map = {"pp": 32, "p": 48, "mp": 56, "mf": 72, "f": 88, "ff": 112, "sfz": 100}

    def finalize_part(part_obj: stream.Part):
        if part_obj is not None:
            parts.append(part_obj)

    # Iterate through remaining tokens
    while idx < len(tokens):
        tok = tokens[idx]

        # New voice block
        if tok.startswith("<voice=") and tok.endswith(">"):
            finalize_part(current_part)
            voice_label = tok[len("<voice="):-1]
            new_part = stream.Part()
            new_part.id = voice_label
            new_part.partName = voice_label

            lbl = voice_label.lower()
            if lbl == "s" or "sop" in lbl:
                new_part.insert(0, instrument.Soprano())
            elif lbl == "a" or "alt" in lbl:
                new_part.insert(0, instrument.Alto())
            elif lbl == "t" or "ten" in lbl:
                new_part.insert(0, instrument.Tenor())
            elif lbl == "b" or "bas" in lbl:
                new_part.insert(0, instrument.Bass())
            elif "piano-rh" in lbl or "piano" in lbl:
                new_part.insert(0, instrument.Piano())
            else:
                new_part.insert(0, instrument.Vocalist())

            current_part = new_part
            current_offset_quarters = 0.0
            current_offset_ticks = 0
            last_offset_ticks = 0
            current_duration_quarters = 1.0
            current_velocity = 64
            idx += 1
            continue

        # Skip tokens until a voice is found
        if current_part is None:
            idx += 1
            continue

        # BarStart: advance to next measure boundary
        if tok == "<BarStart>":
            current_offset_quarters = (int(current_offset_quarters // bar_quarters) + 1) * bar_quarters
            current_offset_ticks = int(current_offset_quarters * ticks_per_beat)
            last_offset_ticks = current_offset_ticks
            idx += 1
            continue

        # RelPos: advance relative ticks
        if tok.startswith("<RelPos_") and tok.endswith(">"):
            try:
                rel = int(tok[len("<RelPos_"):-1])
            except ValueError:
                rel = 0
            last_offset_ticks += rel
            current_offset_ticks = last_offset_ticks
            current_offset_quarters = current_offset_ticks / ticks_per_beat
            idx += 1
            continue

        # Dynamic: set velocity mapping
        if tok.startswith("<Dynamic_") and tok.endswith(">"):
            dyn_val = tok[len("<Dynamic_"):-1]
            current_velocity = dynamic_velocity_map.get(dyn_val, current_velocity)
            idx += 1
            continue

        # Velocity override
        if tok.startswith("<Velocity_") and tok.endswith(">"):
            try:
                vel_val = int(tok[len("<Velocity_"):-1])
            except ValueError:
                vel_val = current_velocity
            current_velocity = vel_val
            idx += 1
            continue

        # Duration: set current duration
        if tok.startswith("<Duration_") and tok.endswith(">"):
            try:
                dur_ticks = int(tok[len("<Duration_"):-1])
            except ValueError:
                dur_ticks = ticks_per_beat
            current_duration_quarters = dur_ticks / ticks_per_beat
            idx += 1
            continue

        # Note-On or Rest: create note/rest event
        if tok.startswith("<Note-On_") and tok.endswith(">"):
            try:
                midi_val = int(tok[len("<Note-On_"):-1])
            except ValueError:
                midi_val = 0

            if midi_val == 0:
                n = note.Rest()
            else:
                n = note.Note()
                n.pitch.midi = midi_val

            n.duration = duration.Duration(current_duration_quarters)
            n.offset = current_offset_quarters
            n.volume = volume.Volume(velocity=current_velocity)
            current_part.insert(n.offset, n)
            idx += 1
            continue

        # Ignore chord annotations
        if tok.startswith("<Chord_"):
            idx += 1
            continue

        # Ignore <EOS>, <PhraseStart>
        if tok == "<EOS>" or tok == "<PhraseStart>":
            idx += 1
            continue

        # Anything else: skip
        idx += 1

    # Finalize last part
    finalize_part(current_part)
    # Insert all parts into the Score
    for i, part in enumerate(parts):
        s.insert(i, part)
    return s
