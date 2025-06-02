
# Symbolic Music Transformer with REMI Tokenization and Chord-Based Planning

This project implements a symbolic music generation system using a Transformer-based architecture. It uses a REMI-style tokenizer, incorporating global chord information and supports a two-stage training process with a high-level bar planner followed by a detail generator.

## Features

- **REMI-style Tokenization**: Custom tokenizer with support for `<Bar>`, `<Position>`, `<Note>`, `<Velocity>`, `<Duration>`, `<Tempo>`, `<TimeSig>`, and `<Key>` tokens.
- **Global Chord Planning**: Chord tokens per bar are extracted using Roman numeral analysis with fallback to chord symbols.
- **Voice Tags**: Support for voice-level control (e.g., `<voice=S1>`).
- **Two-Stage Model**:
  - **Planner Model**: Learns the global structure using bar-level chord sequences.
  - **Detail Model**: Learns note-level generation conditioned on the planner output.
- **Training/Evaluation Scripts**: Easy-to-use scripts for training and evaluating both stages.
- **Support for ABC and MusicXML formats**: Easily convert and tokenize symbolic music datasets.

---

## Project Structure

```text
.
├── remi_tokenizer.py       # Tokenizes MusicXML/ABC into REMI tokens
├── remi_detokenizer.py     # Converts REMI tokens back into MusicXML
├── train.py                # Main training script for planner and detail model
├── eval.py                 # Evaluation script for generating and analyzing output
├── dataset.py              # Dataset loader and sequence encoder/decoder
├── model.py                # Transformer model definitions
├── config.yaml             # Model and training configuration
├── data/
│   └── raw/                # Original music files
│   └── processed/          # Tokenized sequences
├── checkpoints/            # Saved model weights
└── README.md
````

---

## Installation

```bash
git clone https://github.com/your-user/music-transformer-remi.git
cd music-transformer-remi
pip install -r requirements.txt
```

Requirements include:

* `music21`
* `PyTorch`
* `PyYAML`
* `numpy`, `tqdm`, etc.

---

## Data Preparation

1. Put `.musicxml` or `.abc` files in the `data/raw/` directory.
2. Tokenize the files:

```bash
python remi_tokenizer.py --input_dir data/raw --output_dir data/processed
```

3. (Optional) Organize files by composer or source:

```bash
python scripts/organize_by_composer.py
```

---

## Training

### Stage 1: Bar-Level Chord Planner

```bash
python train.py --stage planner --config config.yaml
```

### Stage 2: Note-Level Detail Generator

```bash
python train.py --stage detail --config config.yaml --planner_checkpoint checkpoints/planner_best.pt
```

---

## Evaluation

Generate music from the trained model:

```bash
python eval.py --config config.yaml --output_dir outputs/
```

---

## Configuration

All model and training settings can be modified in `config.yaml`, including:

```yaml
model:
  type: transformer
  hidden_size: 512
  num_layers: 8

train:
  batch_size: 16
  max_seq_len: 2048
  lr: 3e-4
  epochs: 100
```

---

## Token Types

* `<Bar>`: Start of a new measure
* `<Position_XXX>`: Beat/offset within the bar
* `<Note_XXX>`: MIDI note pitch
* `<Duration_XXX>`: Note duration in ticks
* `<Velocity_XXX>`: Note dynamic
* `<Chord_XXX>`: Global bar-level chord (Roman numeral)
* `<Tempo_XXX>`: Tempo changes
* `<TimeSig_XXX>`: Time signature changes
* `<Key_XXX>`: Key signature
* `<voice=XXX>`: Voice designation (S1, S2, A, T, B, etc.)

---

## Notes

* Roman numeral analysis uses `music21.roman.romanNumeralFromChord` with a global key.
* Chord fallback uses `ChordSymbol` if Roman numeral analysis fails.
* `<Chord_unk>` is used when no valid chord can be extracted.
* Tokenization includes optional beat subdivision for finer chord resolution.

---

## Future Enhancements

* Data augmentation (transpose, rhythm variation)
* Polyphonic voice disentanglement
* Conditional generation based on melody, lyrics, or style
* GUI-based generation preview and editing

---

## License

MIT License © 2025 Your Name

---

## Acknowledgements

* [music21](https://web.mit.edu/music21/)
* [REMI](https://arxiv.org/abs/2002.00212)
* [PopMAG](https://arxiv.org/abs/2106.05630)

```

Let me know if you want the README tailored to a specific dataset (JSB, Bach Chorales, POP909, etc.) or want links to actual examples, weights, or demo notebooks.
```
