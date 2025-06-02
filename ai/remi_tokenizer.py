# ai/remi_tokenizer.py

from music21 import stream, note, chord, meter, tempo, key as m21key, roman, harmony
from music21 import dynamics as m21dynamics
from typing import List
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
          1. <BOS>
          2. <Style=...>
          3. <time=num/den>
          4. <key=<tonic><mode>>
          5. <Tempo=nnn>
          6. <PhraseStart>
          7. Multiple chord tokens per bar via extract_chords_by_beat
          8. For each part:
               a. <voice=LABEL>
               b. Events: <BarStart>, <RelPos_n>, <Dynamic_x>, <Note-On>, <Duration_n>, <Velocity_v>
          9. <EOS>
        """
        tokens: List[str] = []
        tokens.append("<BOS>")

        # 1) Style token
        style = getattr(s.metadata, "style", None) or "Unknown"
        style_tok = f"<Style={style}>"
        self._add_token(style_tok)
        tokens.append(style_tok)

        # 2) Build base vocab before adding new tokens
        self.build_base_vocab()

        # 3) Time signature
        ts = s.recurse().getElementsByClass(meter.TimeSignature).first()
        if ts:
            ts_tok = f"<time={ts.numerator}/{ts.denominator}>"
            ts_quarters = ts.barDuration.quarterLength
        else:
            ts_tok = "<time=4/4>"
            ts_quarters = 4.0
        self._add_token(ts_tok)
        tokens.append(ts_tok)

        # 4) Key signature (analyzed)
        try:
            k = s.analyze("key")
            tonic = k.tonic.name    # e.g., "C"
            mode = "maj" if k.mode == "major" else "min"
            key_tok = f"<key={tonic}{mode}>"
        except Exception:
            key_tok = "<key=Cmaj>"
        self._add_token(key_tok)
        tokens.append(key_tok)

        # 5) Tempo token (first MetronomeMark or default 120)
        mm = s.recurse().getElementsByClass(tempo.MetronomeMark).first()
        if mm:
            tempo_tok = f"<Tempo={int(mm.number)}>"
        else:
            tempo_tok = "<Tempo=120>"
        self._add_token(tempo_tok)
        tokens.append(tempo_tok)

        # 6) Phrase start
        tokens.append("<PhraseStart>")

        # 7) Extract multiple chord tokens per bar
        self.extract_chords_by_beat(s, tokens)

        # 8) For each part, add voice tag and encode events
        ticks_per_measure = int(ts_quarters * self.ticks_per_beat)

        for part in s.parts:
            # Determine a clean voice label (S1/S2 → S)
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

            # Iterate through notes/rests in offset order
            last_ticks = 0
            current_bar_index = 0

            for n in part.flat.notesAndRests:
                # 8a) Check if we've crossed into a new bar
                bar_idx = int((n.offset * self.ticks_per_beat) // ticks_per_measure)
                if bar_idx > current_bar_index:
                    current_bar_index = bar_idx
                    tokens.append("<BarStart>")

                # 8b) Compute relative position in ticks
                curr_ticks = int(n.offset * self.ticks_per_beat)
                rel = curr_ticks - last_ticks
                # Clamp rel to [0, ticks_per_measure - 1]
                rel = max(0, min(rel, ticks_per_measure - 1))
                rel_tok = f"<RelPos_{rel}>"
                self._add_token(rel_tok)
                tokens.append(rel_tok)
                last_ticks = curr_ticks

                # 8c) Dynamic token if present
                for expr in n.expressions:
                    if isinstance(expr, m21dynamics.Dynamic):
                        dyn_val = expr.value  # e.g., "p", "mf"
                        dyn_tok = f"<Dynamic_{dyn_val}>"
                        self._add_token(dyn_tok)
                        tokens.append(dyn_tok)
                        break

                # 8d) Note-On or Rest
                if isinstance(n, note.Note):
                    note_tok = f"<Note-On_{n.pitch.midi}>"
                    tokens.append(note_tok)
                elif isinstance(n, chord.Chord):
                    for p in n.pitches:
                        tokens.append(f"<Note-On_{p.midi}>")
                else:  # Rest
                    tokens.append("<Note-On_0>")

                # 8e) Duration
                dur_ticks = int(min(n.quarterLength * self.ticks_per_beat, self.max_duration))
                dur_tok = f"<Duration_{dur_ticks}>"
                tokens.append(dur_tok)

                # 8f) Velocity
                vel = getattr(n.volume, "velocity", 64)
                vel_bin = int(np.clip(vel, 0, 127) // 8 * 8)
                vel_tok = f"<Velocity_{vel_bin}>"
                tokens.append(vel_tok)

        # 9) End-of-sequence
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
