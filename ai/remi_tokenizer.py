# ai/remi_tokenizer.py

from music21 import stream, note, chord, meter, tempo, key as m21key, roman, harmony
from music21 import dynamics as m21dynamics
from typing import List, Tuple
import numpy as np


class REMIABCTokenizer:
    """
    Enhanced REMI-style tokenizer supporting:
      - <Style=...> token (if score.metadata.style is set)
      - <time=num/den> for flexible time signatures
      - <key=<tonic><mode>> for key signature
      - <Tempo=nnn> for tempo changes (first MetronomeMark encountered)
      - <PhraseStart> at beginning of piece
      - Per-bar chord tokens (multiple per bar, Roman with inversions/7ths or ChordSymbol fallback)
      - Per-bar <BarStart> tokens
      - Per-part <voice=...> tags (S, A, T, B, Piano-RH, Piano-LH, UNK)
      - Per-note <RelPos_n> (relative ticks since last event in that part)
      - <Dynamic_x> tokens if dynamics expressions are present
      - <Note-On_midi>, <Duration_n>, <Velocity_v> tokens
      - <EOS> at end
      - A built vocabulary that dynamically grows as new tokens appear
    """

    def __init__(self, ticks_per_beat: int = 4):
        self.ticks_per_beat = ticks_per_beat
        self.max_duration = 32  # up to 8 quarter notes = 8 * ticks_per_beat

        # Initialize vocabulary with fixed tokens
        self.vocab = {
            "<PAD>": 0,
            "<BOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
            "<BarStart>": 4,    # explicit bar-start marker
            "<PhraseStart>": 5  # phrase boundary marker
        }
        self.rev_vocab = {v: k for k, v in self.vocab.items()}

    def _add_token(self, tok: str):
        """Add a token to vocab if not already present."""
        if tok not in self.vocab:
            idx = len(self.vocab)
            self.vocab[tok] = idx
            self.rev_vocab[idx] = tok

    def build_base_vocab(self):
        """
        Predefine all non-voice/time/key/chord/dynamic tokens:
          - <RelPos_0> ... <RelPos_(ticks_per_beat*4 - 1)> for relative positions within a bar
          - <Note-On_21> ... <Note-On_108> for MIDI pitches A0 to C8
          - <Duration_1> ... <Duration_max_duration> in ticks
          - <Velocity_0>, <Velocity_8>, ..., <Velocity_120>
          - <Dynamic_pp>, <Dynamic_p>, <Dynamic_mp>, <Dynamic_mf>, <Dynamic_f>, <Dynamic_ff>, <Dynamic_sfz>
        """
        # Relative positions (0 to ticks_per_beat*4 - 1)
        for i in range(self.ticks_per_beat * 4):
            self._add_token(f"<RelPos_{i}>")

        # MIDI pitches 21 (A0) to 108 (C8)
        for pitch in range(21, 109):
            self._add_token(f"<Note-On_{pitch}>")
        # Rest encoded as <Note-On_0>
        self._add_token("<Note-On_0>")

        # Durations in ticks (1 to max_duration)
        for d in range(1, self.max_duration + 1):
            self._add_token(f"<Duration_{d}>")

        # Velocities in multiples of 8 from 0 to 120
        for v in range(0, 128, 8):
            self._add_token(f"<Velocity_{v}>")

        # Dynamics tokens
        for dyn in ["pp", "p", "mp", "mf", "f", "ff", "sfz"]:
            self._add_token(f"<Dynamic_{dyn}>")

        # <BarStart>, <PhraseStart> already in __init__
        # Voice, time, key, tempo, chord, style tokens added dynamically during tokenization

    def _collect_key_events(self, original_score: stream.Score) -> List[Tuple[float, str]]:
        """
        Scan the original (un‐transposed) score for KeySignature changes.
        Return a list of (offset_in_quarterLength, "TONICmode") sorted by offset.
        """
        events: List[Tuple[float, str]] = []
        # music21 often represents each key change as a KeySignature object at some offset.
        for ks in original_score.recurse().getElementsByClass(KeySignature):
            offset_q = ks.measureNumber * ks.barDuration.quarterLength - ks.barDuration.quarterLength
            # (measureNumber is 1‐based; measureNumber*barLength - barLength == offset of that measure start)
            # But if measureNumber isn't set, fallback to ks.offset:
            if ks.offset is not None:
                offset_q = ks.offset

            try:
                new_key = ks.asKey()  # yields a Key object (tonic+mode)
                tonic = new_key.tonic.name  # e.g., "G"
                mode = "maj" if new_key.mode == "major" else "min"
                events.append((offset_q, f"{tonic}{mode}"))
            except Exception:
                # fallback: if we can't convert to a Key, skip
                continue

        # Sort by offset
        events.sort(key=lambda x: x[0])
        return events


    def extract_chords_by_beat(self, score: stream.Score, tokens: List[str]):
        """
        Extract multiple chord tokens per bar from the score using:
          1. Roman numeral (with global key, including inversion/7th)
          2. harmony.ChordSymbol fallback
          3. <Chord_unk> if both fail

        Appends tokens directly to `tokens` and adds them to vocab via self._add_token.
        """
        # Step 1: Chordify the score (collapse all parts vertically)
        chordified = score.chordify()
        chordified.removeByNotOfClass(chord.Chord)

        # Step 2: Estimate global key for Roman numeral analysis
        try:
            global_key = score.analyze("key")
        except Exception:
            global_key = None

        # Step 3: Estimate time signature for subdivision
        ts = score.recurse().getElementsByClass(meter.TimeSignature).first()
        if ts:
            beat_count = ts.numerator                # e.g., 4 for 4/4
            beat_length = ts.beatDuration.quarterLength
        else:
            beat_count = 4
            beat_length = 1.0

        # Step 4: Iterate over measures and beats
        measures = chordified.getElementsByClass(stream.Measure)
        for m in measures:
            bar_offset = m.offset
            for i in range(beat_count):  # Subdivide each measure into beats
                beat_offset = bar_offset + (i * beat_length)
                # Gather chords sounding in this beat window
                chords_in_beat = m.flat.getElementsByClass(chord.Chord).getElementsByOffset(
                    beat_offset,
                    beat_offset + beat_length,
                    includeEndBoundary=False
                )

                if chords_in_beat:
                    chord_obj = chords_in_beat[0]  # take the first chord at this beat
                    try:
                        if global_key:
                            # Roman numeral with inversion and seventh info
                            rn = roman.romanNumeralFromChord(chord_obj, global_key)
                            chord_tok = f"<Chord_{rn.figure}>"
                        else:
                            raise ValueError("No global key")
                    except Exception:
                        try:
                            cs = harmony.chordSymbolFromChord(chord_obj)
                            chord_tok = f"<Chord_{cs.figure}>"
                        except Exception:
                            chord_tok = "<Chord_unk>"
                else:
                    chord_tok = "<Chord_unk>"

                self._add_token(chord_tok)
                tokens.append(chord_tok)

    def tokenize(self, s: stream.Score) -> List[str]:
        """
        Convert a music21 Score into a list of REMI tokens, including:
        1.  Record all <key_change=…> events from s_orig.
        2.  Build a copy of s_orig, transpose each measure‐block to C (maj/min).
        3.  Emit:
            <BOS>
            <Style=…>
            <time=num/den>
            <key=Cmaj/min>         ← now always C since we've normalized
            <Tempo=…>
            <PhraseStart>
            <key_change=…> …       ← one token per mid‐piece key event
            per‐bar chord tokens via extract_chords_by_beat(on the normalized score)
            per‐part: <voice=…>, <BarStart>, <RelPos_…>, <Dynamic_…>, <Note-On_…>, <Duration_…>, <Velocity_…>
            <EOS>
        """
        # 1) Gather all key‐change events from the *original* score:
        key_events = self._collect_key_events(s_orig)  # list of (offset_q, "Gmaj"), etc.

        # 2) Now, create a *fully transposed copy* of s_orig → s_norm in C
        #    We do a per‐section transposition so that each key‐section is transposed appropriately.
        #    Simplest: assume key signature changes only at measure boundaries. We iterate measure by measure.
        s_norm = stream.Score()
        s_norm.metadata = s_orig.metadata  # carry over metadata (style, etc.)

        # Extract the time signature and bar length once
        ts_obj = s_orig.recurse().getElementsByClass(meter.TimeSignature).first() or meter.TimeSignature("4/4")
        bar_length = ts_obj.barDuration.quarterLength

        # Build a list of (measure_index, KeySignature) sorted by measure start
        ks_list = [(ks.measureNumber, ks) for ks in s_orig.recurse().getElementsByClass(KeySignature)]
        ks_list.sort(key=lambda x: x[0])  # measureNumber ascending

        # If no explicit KeySignature, infer global key from analysis:
        if not ks_list:
            try:
                global_k = s_orig.analyze("key")
                ks0 = KeySignature(global_k.sharps)
                ks_list = [(1, ks0)]
            except Exception:
                ks_list = [(1, KeySignature(0))]  # default C

        # Now iterate measure by measure:
        all_parts = s_orig.parts
        for part in all_parts:
            # Create a matching part in s_norm
            p_norm = part.clone()  # deep copy notes/rests, but we'll re‐transpose
            p_norm.flat.notesAndRests.stream()  # ensure flat structure
            s_norm.insert(0, p_norm)  # timing will be overwritten by transposition below

        # We build a lookup table: for each measure number, find the KeySignature in effect.
        # Then transpose all notes in that measure to C/(A) depending on major/minor.
        key_by_measure = {}
        for meas_num, ks in ks_list:
            k = ks.asKey()
            key_by_measure[meas_num] = k

        # If no explicit KeySignature for a given measure, assume carry‐over previous:
        # Build a list of measureNumbers in ascending order:
        all_measure_numbers = sorted(key_by_measure.keys())
        # For measures not in key_by_measure, fill with last seen
        max_meas = int(s_orig.highestOffset // bar_length) + 1
        last_key = None
        for m_idx in range(1, max_meas + 1):
            if m_idx in key_by_measure:
                last_key = key_by_measure[m_idx]
                key_by_measure[m_idx] = last_key
            else:
                key_by_measure[m_idx] = last_key or m21key.Key("C")  # default C

        # Now copy part by part, measure by measure:
        for part_idx, part in enumerate(all_parts):
            new_part = stream.Part()
            new_part.id = part.id
            new_part.partName = part.partName

            # For each measure, collect notes/rests, transpose to C/A relative to that measure's key
            for m_idx in range(1, max_meas + 1):
                this_key = key_by_measure[m_idx]
                # Determine target: if this_key.mode == "major", target = C; else A minor
                if this_key.mode == "major":
                    tgt_key = m21key.Key("C")
                else:
                    tgt_key = m21key.Key("A", "minor")

                iv = m21key.interval.Interval(this_key.tonic, tgt_key.tonic)

                # Extract the measure from original
                m_orig = part.measure(m_idx)
                if m_orig is None:
                    continue

                # Clone measure, transpose its contents
                m_copy = m_orig.clone()
                m_copy.transpose(iv, inPlace=True)
                # Append to new_part at offset = (m_idx - 1) * bar_length
                for el in m_copy.flat.notesAndRests:
                    new_el = el.clone()
                    new_el.offset = (m_idx - 1) * bar_length + el.offset
                    new_part.insert(new_el.offset, new_el)

            s_norm.insert(0, new_part)

        # From here on, work exclusively with s_norm (everything is now in C)
        # but we still keep key_events (from the original) as metadata.

        # 3) Begin tokenization
        tokens: List[str] = []
        tokens.append("<BOS>")

        # 3a) Style
        style = getattr(s_orig.metadata, "style", None) or "Unknown"
        style_tok = f"<Style={style}>"
        self._add_token(style_tok)
        tokens.append(style_tok)

        # 3b) Build base vocab
        self.build_base_vocab()

        # 3c) Time signature (first)
        ts = s_norm.recurse().getElementsByClass(meter.TimeSignature).first()
        if ts:
            ts_tok = f"<time={ts.numerator}/{ts.denominator}>"
            ts_quarters = ts.barDuration.quarterLength
        else:
            ts_tok = "<time=4/4>"
            ts_quarters = 4.0
        self._add_token(ts_tok)
        tokens.append(ts_tok)

        # 3d) Key signature (normalized—should always be C!)
        #     We only emit the *initial* <key=> here.  Mid‐piece changes get <key_change=…>.
        #     So compute the *first* global key of s_orig, but then show it as “Cmaj” or “Amin”.
        try:
            init_orig_key = s_orig.analyze("key")
            init_mode = "maj" if init_orig_key.mode == "major" else "min"
            # After normalization, the first key is always C (or A) depending on init_mode,
            # but we want the model to learn that “initial key = Cmaj” if it was major, or Cmin if minor.
            if init_orig_key.mode == "major":
                key_tok = "<key=Cmaj>"
            else:
                key_tok = "<key=Amin>"
        except Exception:
            key_tok = "<key=Cmaj>"
        self._add_token(key_tok)
        tokens.append(key_tok)

        # 3e) Tempo (first or default)
        mm = s_norm.recurse().getElementsByClass(tempo.MetronomeMark).first()
        if mm:
            tempo_tok = f"<Tempo={int(mm.number)}>"
        else:
            tempo_tok = "<Tempo=120>"
        self._add_token(tempo_tok)
        tokens.append(tempo_tok)

        # 3f) PhraseStart
        tokens.append("<PhraseStart>")

        # 3g) Emit *all* mid‐piece key‐change tokens (in chronological order)
        #     Note: key_events is a list of (offset_q, "Gmaj"), … from the original
        for _, new_key_str in key_events:
            # Skip the very first if it matches the initial key
            if new_key_str.lower().startswith("cmaj") or new_key_str.lower().startswith("amin"):
                continue
            change_tok = f"<key_change={new_key_str}>"
            self._add_token(change_tok)
            tokens.append(change_tok)

        # 4) Chord tokens (per‐bar, on the normalized score s_norm)
        self.extract_chords_by_beat(s_norm, tokens)

        # 5) For each part in s_norm, emit <voice=…> + events
        ticks_per_measure = int(ts_quarters * self.ticks_per_beat)
        for part in s_norm.parts:
            raw_label = (part.partName or part.id or "UNK").strip()
            lu = raw_label.upper()
            if "S2" in lu or "S1" in lu or "SOP" in lu:
                voice_label = "S"
            elif "ALT" in lu or lu == "A":
                voice_label = "A"
            elif "TEN" in lu or lu == "T":
                voice_label = "T"
            elif "BAS" in lu or lu == "B":
                voice_label = "B"
            elif "PIANO-RH" in lu or "RH" in lu:
                voice_label = "Piano-RH"
            elif "PIANO-LH" in lu or "LH" in lu:
                voice_label = "Piano-LH"
            elif "PIANO" in lu:
                voice_label = "Piano"
            else:
                fc = lu[0]
                voice_label = fc if fc in ["S", "A", "T", "B"] else "UNK"

            voice_tok = f"<voice={voice_label}>"
            self._add_token(voice_tok)
            tokens.append(voice_tok)

            last_ticks = 0
            current_bar_index = 0

            for n in part.flat.notesAndRests:
                # 5a) Bar boundary?
                bar_idx = int((n.offset * self.ticks_per_beat) // ticks_per_measure)
                if bar_idx > current_bar_index:
                    current_bar_index = bar_idx
                    tokens.append("<BarStart>")

                # 5b) Relative position (in ticks since last event)
                curr_ticks = int(n.offset * self.ticks_per_beat)
                rel = curr_ticks - last_ticks
                rel = max(0, min(rel, ticks_per_measure - 1))
                rel_tok = f"<RelPos_{rel}>"
                self._add_token(rel_tok)
                tokens.append(rel_tok)
                last_ticks = curr_ticks

                # 5c) Dynamics if present
                for expr in n.expressions:
                    if isinstance(expr, m21dynamics.Dynamic):
                        dyn_val = expr.value  # e.g. "p", "mf"
                        dyn_tok = f"<Dynamic_{dyn_val}>"
                        self._add_token(dyn_tok)
                        tokens.append(dyn_tok)
                        break

                # 5d) Note‐On / Rest
                if isinstance(n, note.Note):
                    note_tok = f"<Note-On_{n.pitch.midi}>"
                    tokens.append(note_tok)
                elif isinstance(n, chord.Chord):
                    for p in n.pitches:
                        tokens.append(f"<Note-On_{p.midi}>")
                else:  # Rest
                    tokens.append("<Note-On_0>")

                # 5e) Duration (clamped)
                dur_ticks = int(min(n.quarterLength * self.ticks_per_beat, self.max_duration))
                dur_tok = f"<Duration_{dur_ticks}>"
                tokens.append(dur_tok)

                # 5f) Velocity
                vel = getattr(n.volume, "velocity", 64)
                vel_bin = int(np.clip(vel, 0, 127) // 8 * 8)
                vel_tok = f"<Velocity_{vel_bin}>"
                tokens.append(vel_tok)

        # 6) EOS
        tokens.append("<EOS>")
        return tokens


    def encode(self, tokens: List[str]) -> List[int]:
        """
        Convert token strings to integer IDs, using <UNK> (3) if token not in vocab.
        """
        return [self.vocab.get(tok, self.vocab["<UNK>"]) for tok in tokens]

    def decode(self, token_ids: List[int]) -> List[str]:
        """
        Convert integer IDs back to token strings, using <UNK> if ID not in rev_vocab.
        """
        return [self.rev_vocab.get(tid, "<UNK>") for tid in token_ids]

    def to_text(self, token_ids: List[int]) -> str:
        """
        Convert a list of token IDs to a single space-separated string.
        """
        return " ".join(self.decode(token_ids))
