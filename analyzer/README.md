# nwc2abc

Convert Noteworthy Composer (NWCtxt) files to ABC notation (optimized for GenAI experiments).

- Using AI, change styles and fix harmony issue

- small melodic improvements acceptable in Soprano to improve harmony
- generate it into a downloadable .abc file

Please reply “yes, keep Soprano as is, full text here” or
“yes, allow small improvements, downloadable file preferred”,


- In case V:T, V:B is broken, generate full Gospel SATB with aligned bars

finish V:T and V:B part 
- align with S, A as much as possible
- Keep the bass line functional and harmonic (roots, 5ths, passing tones only)
- Ensure it feels gospel but works for standard choirs
- Avoid unnecessary melodic wandering

- Tenor: supportive, simple, occasional rhythmic variety but never competing with Soprano
- Bass: true gospel feel: roots, 5ths, occasional passing tones, always functional harmonic support

- ABC Transcription Tools: https://michaeleskin.com/abctools/abctools.html
- Using tool, export as MIDI
- Import to NoteWorthy Composer

## Styles
The original improvement and revised Alto lean toward Baroque chorale style, inspired by J.S. Bach’s SATB writing:
- Functional harmony: clear tonic–dominant–subdominant motion
- Inner voices provide harmonic support, not melodic independence
- Stepwise motion dominates, with very few large leaps
- Soprano has melodic leadership, bass provides harmonic foundation
- Alto and tenor mostly fill chords and create smooth voice leading
suitable for chorale or hymn-like settings.

| Style                              | Characteristics                                                              | Recommended for                        |
| ---------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------- |
| **Classical (Mozart, Haydn)**      | More homophonic textures, wider phrase lengths, simpler inner voice movement | Sacred anthems, masses                 |
| **Romantic (Brahms, Mendelssohn)** | Richer chromaticism, more dramatic dynamics, wider ranges                    | Concert choir repertoire               |
| **Modern (20th-century)**          | Parallel motion, cluster chords, dissonance resolving into consonance        | Contemporary classical or experimental |
| **Gospel / Spirituals**            | Rhythmic energy, call-and-response, blues inflections                        | Church and community choirs            |
| **Barbershop**                     | Dominant 7th chords, very tight voicing, focus on "ringing" chords           | Small vocal ensembles                  |
| **Jazz Choral**                    | Complex chords (9ths, 11ths, 13ths), swing rhythm, syncopation               | Jazz or pop choirs                     |
| **Renaissance (Palestrina)**       | Pure polyphony, very controlled dissonance, all voices equal                 | Early music ensembles                  |

### Gospel and Spirituals
 A Gospel / Spirituals SATB arrangement will have a very different character:
🎵 Features to adapt for your music:
More rhythmic vitality: syncopation, offbeat accents
Richer harmonies: frequent use of dominant 7th and added 9th chords
More bass movement: walking bass lines are common
Call and response feel: sometimes soprano leads, others respond
"Blue notes" and flattened 3rd / 7th in some passing notes

Tenor (V:T): smooth harmonic filler line, more syncopation, occasional blue-note passing tone, close harmony with Alto
Bass (V:B): active walking lines, octave jumps, strong dominant 7th resolutions, a signature of gospel bass

### Jazz Style
To adapt your SATB arrangement into a Jazz Choral style, we'll incorporate elements characteristic of jazz harmony and rhythm. This includes:
- Extended harmonies: Incorporating 7ths, 9ths, and 13ths.
- Swing rhythms: Utilizing syncopation and swung eighth notes.
- Chromaticism: Adding color through chromatic passing tones.
- Voice leading: Smooth, often stepwise motion with occasional expressive leaps.
- Rhythmic variation: Introducing syncopation and varied note durations.

### Jazz inspired
- Keep choral singability (reasonable ranges, avoid extreme chromatic leaps).
- Add tasteful jazz harmony (7ths, 9ths, occasional 6ths), but no heavy improvisation.
- Introduce syncopation & suspensions, while retaining block-chord choral usability.
- Maintain your original rhythmic and phrase structure as much as possible.

### Modern style
 20th–21st century modern choral techniques:
- More independent inner voices, but singable
- Non-functional harmony moments, parallel chords or added tone textures
- Occasional quartal/quintal harmony
- Suspensions, mild dissonances resolving smoothly
- Slight rhythmic independence between voices
- No extreme ranges (suitable for real choirs)
This will still respect your original structure but reimagine the voicings in a more fluid, modern way, while maintaining readability in ABC.

## Features
- Converts NWC via musescore/nwc2musicxml
- Outputs ABC at 3 levels (`raw`, `medium`, `simple`)
- Ready for AI arrangement tasks

## Install

```bash
git clone ...
cd nwc2abc
pip install .
```

## Usage

```python
from nwc2abc import nwc_to_simplified_abc
abc = nwc_to_simplified_abc("your_score.nwc.txt", simplicity_level="medium")
print(abc)
```

or CLI:

```bash
nwc2abc-cli your_score.nwc.txt --level simple
```

## Requirements
- Python 3.8+
- Java (for `nwc2musicxml.jar` if using local mode)

## License
MIT