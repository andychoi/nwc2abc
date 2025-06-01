
# Install environment in Windows
Install Prebuilt Binary Wheels Instead
Avoid building from source by upgrading pip, setuptools, and wheel first:

```
python -m pip install --upgrade pip setuptools wheel
Then install music21 and avoid broken builds:

pip install music21 --prefer-binary
If that fails, try:

pip install music21 --no-build-isolation
```

Here is the **raw Markdown** for your `README.md` documentation:

````markdown
# 🎼 MIDI to SATB ABC Notation Converter

This Python script converts a **MIDI file** into **ABC notation** for SATB (Soprano, Alto, Tenor, Bass) choral format. It uses pitch-based voice splitting and supports basic formatting suitable for rendering or playback using ABC-compatible tools.

---

## 🧰 Features

- Converts MIDI into four-part ABC notation: Soprano, Alto, Tenor, Bass.
- Splits two MIDI tracks into SA and TB based on pitch thresholds.
- Outputs ABC format compatible with abcjs, EasyABC, and abcmidi.
- Customizable output to file or console.

---

## 📦 Requirements

- Python 3.7+
- [music21](https://web.mit.edu/music21/) library

Install dependencies:
```bash
pip install music21
````

---

## 🚀 Usage

```bash
python midi_to_abc.py input_file.mid --o output_file.abc
```

### Arguments

* `input_file.mid` — Path to the input MIDI file.
* `--o` or `--output` — (Optional) Path to save the output `.abc` text. If not specified, outputs to console.

---

## 🧠 How It Works

1. **Parse MIDI file** using `music21.converter`.
2. **Split notes by pitch**:

   * Notes with MIDI pitch ≥ 65 are treated as Soprano, < 65 as Alto (first track).
   * Notes with MIDI pitch ≥ 53 are Tenor, < 53 as Bass (second track).
3. **Convert durations** using fractional quarter lengths to ABC duration syntax.
4. **Format bars** with `|` every 4 notes.
5. **Generate ABC notation** with `%%score { S A \n T B }` layout directive.

---

## 🔧 Functions Overview

* `convert_duration(q_len)` – Converts `quarterLength` to ABC duration string.
* `split_part_by_pitch(part, threshold)` – Splits a part into high/low based on MIDI pitch.
* `format_voice_notes(notes)` – Joins notes into bars using `|`.
* `to_abc_voice(voice_id, name, notes)` – Outputs ABC string for one voice.
* `midi_to_abc_text(filename, output_path)` – Main conversion function.
* `parse_args()` – Argument parser.

---

## 📄 Example Output

```abc
X:1
T:example.mid
M:4/4
L:1/4
Q:1/4=100
%%score { S A \n T B }
V:S name="Soprano"
C5 D5 E5 F5 | ...
V:A name="Alto"
A4 B4 C5 D5 | ...
V:T name="Tenor"
F3 G3 A3 B3 | ...
V:B name="Bass"
C3 D3 E3 F3 | ...
```

---

## ⚠️ Notes

* The script assumes **2 MIDI tracks**: one for Soprano/Alto, and one for Tenor/Bass.
* Voice assignment is based on pitch, not channel or instrument.

---

## 📜 License

MIT License

```


### Additional Tips:

* **Voice Separation:** If the Soprano and Alto (or Tenor and Bass) parts are on the same track, you might need to manually separate them based on MIDI channel or pitch range before conversion.

* **Alternative Tools:** If you prefer a web-based solution, you can use the [Melobytes MIDI to ABC converter](https://melobytes.com/en/app/midi2abc) for simple conversions.([Melobytes][1])

* **Advanced Parsing:** For more complex MIDI files, consider using the [music21](https://web.mit.edu/music21/) Python library, which offers advanced tools for parsing and analyzing MIDI and ABC notation.

If you encounter any issues during the conversion process or need assistance with specific parts of the ABC notation, feel free to ask!

[1]: https://melobytes.com/en/app/midi2abc?utm_source=chatgpt.com "Convert MIDI to abc [Melobytes.com]"
