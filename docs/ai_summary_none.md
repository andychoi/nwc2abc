# Python Project Summary

## `bar_planner.py`
```python
# ai/bar_planner.py

import torch.nn as nn
import torch
from relative_transformer import PositionalEncoding

class BarPlannerModel(nn.Module):
    """
    Given a prefix of <time=...>, <key=...>, <BarStart>, <Chord_...>, predict next <Chord_...>.
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional = PositionalEncoding(d_model, max_len=1024)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        # We'll only decode positions that correspond to <BarStart> events.

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.positional(emb)
        tgt_mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), 1).bool()
        memory = torch.zeros_like(emb)
        out = self.decoder(emb, memory, tgt_mask=tgt_mask)
        logits = self.fc_out(out)
        return logits  # (batch, seq_len, vocab_size)

```

## `detail_generator.py`
```python
# ai/detail_generator.py

import torch.nn as nn
import torch
from relative_transformer import PositionalEncoding, PositionalEncoding


class DetailGeneratorModel(nn.Module):
    """
    Given a chord plan (e.g. <Chord_I>, <Chord_IV>, ...), we generate a sequence of
    `<voice=...>`, `<BarStart>`, `<RelPos_...>`, `<Note-On_...>`, `<Duration_...>`, `<Velocity_...>`, `<BarStart>`, ...
    for each bar, conditioned on the chord plan tokens.
    """
    def __init__(
        self,
        vocab_size,
        d_model=1024,
        nhead=16,
        num_layers=12,
        dim_feedforward=4096,
        dropout=0.1,
        max_rel_dist=2048
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional = PositionalEncoding(d_model, max_len=8192)
        self.decoder = RelativeTransformerDecoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_rel_dist=max_rel_dist
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.positional(emb)
        tgt_mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), 1).bool()
        out = self.decoder(emb, tgt_mask=tgt_mask)
        return self.fc_out(out)

```

## `eval.py`
```python
# ai/eval.py

"""
Usage:

python eval.py examples/melody.abc \
  --device cuda \
  --max_len 256 \
  --temperature 1.0 \
  --top_k 50 \
  --top_p 0.9 \
  --out generated_tokens.txt \
  --export_abc prompt_vs_gen.abc \
  --export_musicxml prompt_vs_gen.xml
"""

import argparse
from pathlib import Path
import torch
from music21 import converter, stream, metadata, instrument, key as m21key, interval, meter

from remi_tokenizer import REMIABCTokenizer
from remi_detokenizer import remi_tokens_to_score
from train import DecoderOnlyMusicGenModel


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    From: https://github.com/huggingface/transformers/blob/main/src/transformers/generation_logits_process.py
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove tokens with a probability less than the top-k largest probabilities
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits


def generate_sequence(
    model, tokenizer, prime_tokens, max_len=200, temperature=1.0, top_k=0, top_p=0.0, device="cpu"
):
    """
    Autoregressive generation with optional top-k and/or nucleus (top-p) sampling.
    """
    model.eval()
    input_ids = tokenizer.encode(prime_tokens)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_len):
        with torch.no_grad():
            outputs = model(input_tensor)  # (1, seq_len, vocab_size)
            next_logits = outputs[0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_logits.clone(), top_k=top_k, top_p=top_p)
            next_probs = torch.softmax(filtered_logits, dim=-1)
            next_token_id = torch.multinomial(next_probs, num_samples=1).item()

        input_ids.append(next_token_id)
        if tokenizer.rev_vocab.get(next_token_id) == "<EOS>":
            break
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    return tokenizer.decode(input_ids)


def merge_prompt_and_generated(prompt_score: stream.Score, generated_score: stream.Score) -> stream.Score:
    """
    Side-by-side layout: keep all original staves, then append 'Generated Continuation' as a piano part.
    """
    merged = stream.Score()
    merged.insert(0, metadata.Metadata())
    merged.metadata.title = "Prompt + AI Generated"
    merged.metadata.composer = "AI Composer"

    # Keep each original part (SATB, etc.)
    for i, part in enumerate(prompt_score.parts):
        part_copy = part.flat.notesAndRests.stream()
        part_copy.id = part.id or f"Voice_{i}"
        part_copy.partName = part.partName or part.id or f"Voice_{i}"

        label = part_copy.partName.lower()
        if "sop" in label or label == "s":
            part_copy.insert(0, instrument.Soprano())
        elif "alt" in label or label == "a":
            part_copy.insert(0, instrument.Alto())
        elif "ten" in label or label == "t":
            part_copy.insert(0, instrument.Tenor())
        elif "bass" in label or label == "b":
            part_copy.insert(0, instrument.Bass())
        else:
            part_copy.insert(0, instrument.Vocalist())

        merged.insert(i, part_copy)

    # Add generated part (Piano)
    gen_part = stream.Part()
    gen_part.id = "Generated"
    gen_part.partName = "Generated Continuation"
    gen_part.insert(0, instrument.Piano())
    gen_part.append(generated_score.flat.notesAndRests)
    merged.insert(len(prompt_score.parts), gen_part)

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Generate music with REMI model: transpose input→C, generate→C, then retranspose→original key/time"
    )
    parser.add_argument("prompt_file", type=Path, help="ABC file as prompt")
    parser.add_argument("--model", type=Path, default="musicgen_remi_model.pt", help="Path to trained model")
    parser.add_argument("--device", default="cpu", help="Device (cuda/mps/cpu)")
    parser.add_argument("--max_len", type=int, default=200, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling (0 → disabled)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling (0 → disabled)")
    parser.add_argument("--out_tokens", type=Path, help="Save generated tokens to file")
    parser.add_argument("--export_abc", type=Path, help="Save side-by-side as ABC")
    parser.add_argument("--export_musicxml", type=Path, help="Save side-by-side as MusicXML")
    args = parser.parse_args()

    # 1) Load original prompt
    prompt_score = converter.parse(str(args.prompt_file))
    orig_ts = prompt_score.recurse().getElementsByClass(meter.TimeSignature).first()
    orig_key_obj = prompt_score.analyze("key")

    # 2) Transpose prompt to C major/minor
    if orig_key_obj.mode == "major":
        target_key = m21key.Key("C")
    else:
        target_key = m21key.Key("C", "minor")
    iv_to_C = interval.Interval(orig_key_obj.tonic, target_key.tonic)
    prompt_in_C = prompt_score.transpose(iv_to_C)

    # 3) Tokenize in-C
    tokenizer = REMIABCTokenizer()
    tokens = tokenizer.tokenize(prompt_in_C)
    print(f"🎼 Prompt tokens (in C):\n{' '.join(tokens)}")

    # 4) Load model & vocab
    checkpoint = torch.load(str(args.model), map_location=args.device)
    model = DecoderOnlyMusicGenModel(vocab_size=len(checkpoint["vocab"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    # Replace tokenizer vocab with training-time vocab
    tokenizer.vocab = checkpoint["vocab"]
    tokenizer.rev_vocab = {v: k for k, v in tokenizer.vocab.items()}

    # 5) Generate continuation (in C) with top-k/top-p
    print("🎹 Generating continuation (in C)...")
    generated_tokens = generate_sequence(
        model=model,
        tokenizer=tokenizer,
        prime_tokens=tokens,
        max_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
    )

    print("\n🎵 Generated Tokens:\n")
    print(" ".join(generated_tokens))

    if args.out_tokens:
        args.out_tokens.write_text(" ".join(generated_tokens), encoding="utf-8")
        print(f"✅ Tokens saved to {args.out_tokens}")

    # 6) Detokenize → Score in C
    score_in_C = remi_tokens_to_score(generated_tokens)

    # 7) Reverse-transpose back to original key
    iv_to_orig = interval.Interval(target_key.tonic, orig_key_obj.tonic)
    score_in_orig = score_in_C.transpose(iv_to_orig)

    # 8) Re-insert original time signature & key at offset 0
    if orig_ts:
        score_in_orig.insert(0, orig_ts)
    score_in_orig.insert(
        0,
        m21key.Key(
            orig_key_obj.tonic.name
            + ("major" if orig_key_obj.mode == "major" else "minor")
        ),
    )

    # 9) Side-by-side merge & show
    combined_score = merge_prompt_and_generated(prompt_score, score_in_orig)
    combined_score.show()

    # 10) Exports
    if args.export_abc:
        combined_score.write("abc", fp=str(args.export_abc))
        print(f"✅ ABC exported to {args.export_abc}")
    if args.export_musicxml:
        combined_score.write("musicxml", fp=str(args.export_musicxml))
        print(f"✅ MusicXML exported to {args.export_musicxml}")


if __name__ == "__main__":
    main()

```

## `eval_analysis_tool.py`
```python
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

```

## `relative_transformer.py`
```python
# ai/relative_transformer.py

import torch
import torch.nn as nn


class RelativeMultiheadAttention(nn.Module):
    """
    A drop‐in replacement for nn.MultiheadAttention that adds
    relative positional biases. This follows the Music Transformer approach:
    we compute a learned R^{(length, length)} bias for each head, indexing
    by (i-j) relative distance.
    """
    def __init__(self, d_model, nhead, max_rel_dist=1024, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # relative bias table: for distances in [-max_rel_dist, +max_rel_dist]
        self.max_rel_dist = max_rel_dist
        self.rel_bias = nn.Parameter(
            torch.zeros((2 * max_rel_dist + 1, nhead))
        )

    def forward(self, x, attn_mask=None):
        # x: (batch, seq_len, d_model)
        bsz, seq_len, d_model = x.shape
        # 1) Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*d_model)
        qkv = qkv.view(bsz, seq_len, 3, self.nhead, self.d_k)
        q, k, v = qkv.unbind(dim=2)  # each is (batch, seq_len, nhead, d_k)

        # 2) reshape for multihead: (batch*nhead, seq_len, d_k)
        q = q.transpose(1, 2).reshape(bsz * self.nhead, seq_len, self.d_k)
        k = k.transpose(1, 2).reshape(bsz * self.nhead, seq_len, self.d_k)
        v = v.transpose(1, 2).reshape(bsz * self.nhead, seq_len, self.d_k)

        # 3) Scaled dot‐product attention
        scores = torch.bmm(q, k.transpose(1, 2))  # (bsz*nhead, seq_len, seq_len)
        scores = scores / (self.d_k ** 0.5)

        # 4) Add relative position bias
        #    Compute a matrix R where R_{i,j} = rel_bias[i-j + max_rel_dist]
        #    Clip distances outside [-max_rel_dist, max_rel_dist].
        device = x.device
        idxs = torch.arange(seq_len, device=device)
        rel_pos = idxs.unsqueeze(1) - idxs.unsqueeze(0)  # (seq_len, seq_len)
        rel_pos_clamped = torch.clamp(
            rel_pos + self.max_rel_dist, 0, 2 * self.max_rel_dist
        )  # shift into [0, 2*max_rel_dist]
        # rel_bias_table: (2*max_rel_dist+1, nhead)
        bias = self.rel_bias[rel_pos_clamped]  # (seq_len, seq_len, nhead)
        bias = bias.permute(2, 0, 1).contiguous()  # (nhead, seq_len, seq_len)
        bias = bias.view(self.nhead * 1, seq_len, seq_len)  # repeat for each batch
        # Since scores is (bsz*nhead, seq_len, seq_len), tile bias
        bias = bias.repeat(bsz, 1, 1)  # (bsz*nhead, seq_len, seq_len)
        scores = scores + bias

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == True, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.bmm(attn, v)  # (bsz*nhead, seq_len, d_k)
        out = out.view(bsz, self.nhead, seq_len, self.d_k)
        out = out.transpose(1, 2).reshape(bsz, seq_len, d_model)
        out = self.out_proj(out)
        return out


class RelativeTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, max_rel_dist=1024):
        super().__init__()
        self.self_attn = RelativeMultiheadAttention(d_model, nhead, max_rel_dist, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, tgt_mask=None):
        # Self‐attention with relative bias
        sa = self.self_attn(x, attn_mask=tgt_mask)
        x = x + self.dropout1(sa)
        x = self.norm1(x)

        # Feed‐forward
        ff = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = x + self.dropout2(ff)
        x = self.norm2(x)
        return x


class RelativeTransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout, max_rel_dist=1024):
        super().__init__()
        self.layers = nn.ModuleList([
            RelativeTransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_rel_dist=max_rel_dist
            )
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x, tgt_mask):
        out = x
        for layer in self.layers:
            out = layer(out, tgt_mask=tgt_mask)
        return out

```

## `remi_detokenizer.py`
```python
from music21 import stream, note, tempo, meter, duration, volume, instrument, key as m21key, key
from typing import List


def remi_tokens_to_score(tokens: List[str]) -> stream.Score:
    """
    Convert REMI tokens into a multi‐part music21.Score, now including:
      • <key_change=TONICmode> → insert a new KeySignature at the current offset
    All other behaviors remain as before.
    """
    s = stream.Score()
    s.insert(0, tempo.MetronomeMark(number=120))

    idx = 0
    ts_quarters = 4.0
    ticks_per_beat = 4
    bar_quarters = ts_quarters

    # 1) First parse global tokens until the first <voice=…>
    while idx < len(tokens):
        tok = tokens[idx]
        if tok == "<BOS>":
            idx += 1
            continue

        # Style: ignore
        if tok.startswith("<Style=") and tok.endswith(">"):
            idx += 1
            continue

        # Time signature
        if tok.startswith("<time=") and tok.endswith(">"):
            ts_val = tok[len("<time="):-1]
            num, den = ts_val.split("/")
            ts_obj = meter.TimeSignature(f"{num}/{den}")
            s.insert(0, ts_obj)
            ts_quarters = ts_obj.barDuration.quarterLength
            bar_quarters = ts_quarters
            idx += 1
            continue

        # Key signature (initial)
        if tok.startswith("<key=") and tok.endswith(">"):
            key_val = tok[len("<key="):-1]
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

        # PhraseStart: ignore at global level
        if tok == "<PhraseStart>":
            idx += 1
            continue

        # If we hit a <key_change=> before any <voice>, insert a new KeySignature at offset 0
        if tok.startswith("<key_change=") and tok.endswith(">"):
            new_key = tok[len("<key_change="):-1]
            if new_key.endswith("maj"):
                tonic = new_key[:-3]
                ks2 = key.Key(tonic + "major")
            else:
                tonic = new_key[:-3]
                ks2 = key.Key(tonic + "minor")
            s.insert(0, ks2)
            idx += 1
            continue

        # Once we see a <voice=>, global parsing ends
        if tok.startswith("<voice="):
            break

        # Ignore any mid‐piece <Chord_…> here
        if tok.startswith("<Chord_"):
            idx += 1
            continue

        # Anything else at global: skip
        idx += 1

    # 2) Now parse per‐voice blocks
    parts = []
    current_part = None

    current_offset_quarters = 0.0
    current_offset_ticks = 0
    last_offset_ticks = 0
    current_duration_quarters = 1.0
    current_velocity = 64
    dynamic_velocity_map = {"pp": 32, "p": 48, "mp": 56, "mf": 72, "f": 88, "ff": 112, "sfz": 100}

    def finalize_part(p):
        if p is not None:
            parts.append(p)

    while idx < len(tokens):
        tok = tokens[idx]

        # New voice
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

        # If no part yet, skip
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

        # key_change mid‐piece within a part: insert new KeySignature at current_offset
        if tok.startswith("<key_change=") and tok.endswith(">"):
            new_key = tok[len("<key_change="):-1]
            if new_key.endswith("maj"):
                tonic = new_key[:-3]
                ks2 = key.Key(tonic + "major")
            else:
                tonic = new_key[:-3]
                ks2 = key.Key(tonic + "minor")
            s.insert(current_offset_quarters, ks2)
            idx += 1
            continue

        # RelPos
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

        # Dynamic
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

        # Duration
        if tok.startswith("<Duration_") and tok.endswith(">"):
            try:
                dur_ticks = int(tok[len("<Duration_"):-1])
            except ValueError:
                dur_ticks = ticks_per_beat
            current_duration_quarters = dur_ticks / ticks_per_beat
            idx += 1
            continue

        # Note-On or Rest
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

        # Ignore chord annotations (already used in tokenization)
        if tok.startswith("<Chord_"):
            idx += 1
            continue

        # Ignore <EOS>, <PhraseStart>
        if tok == "<EOS>" or tok == "<PhraseStart>":
            idx += 1
            continue

        # Anything else: skip
        idx += 1

    # Finalize final part
    finalize_part(current_part)
    for i, part in enumerate(parts):
        s.insert(i, part)

    return s

```

## `remi_tokenizer.py`
```python
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

```

## `run_generate.py`
```python
#!/usr/bin/env python3
"""
run_generate.py

Given:
  • A folder of ABC training data (already tokenized by REMIABCTokenizer during train) 
  • A trained REMI‐Transformer checkpoint
  • A single ABC prompt file

Produce:
  • A generated continuation (in ABC and MusicXML)
  • Writes everything into a user‐specified output directory

Usage:
    python run_generate.py \
      --abc-input-dir /path/to/abc_corpus \
      --model-checkpoint /path/to/musicgen_remi_model.pt \
      --prompt-file /path/to/prompt.abc \
      --output-dir /path/to/output_folder \
      --device cuda \
      --max-len 256 \
      --temperature 1.0 \
      --top-k 50 \
      --top-p 0.9
"""

import argparse
import shutil
import sys
from pathlib import Path
import torch
from music21 import converter, stream, metadata, instrument, key as m21key, interval, meter

from remi_tokenizer import REMIABCTokenizer
from remi_detokenizer import remi_tokens_to_score
from train import DecoderOnlyMusicGenModel


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """
    Apply top-k and/or nucleus (top-p) filtering to logits before sampling.
    """
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Zero out everything not in top_k
        threshold = torch.topk(logits, top_k)[0][..., -1, None]
        logits[logits < threshold] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Tokens to remove: cumulative_probs > top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Always keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter back to original indices
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits


def generate_sequence(
    model,
    tokenizer: REMIABCTokenizer,
    prime_tokens: list,
    max_len: int = 200,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    device: str = "cpu",
) -> list:
    """
    Autoregressively generate a token stream given:
      • model: a trained DecoderOnlyMusicGenModel
      • tokenizer: REMIABCTokenizer (vocab already replaced by train-vocab)
      • prime_tokens: list[str] from tokenizer.tokenize(prompt_in_C)
      • max_len: maximum number of new tokens (break on <EOS>)
      • temperature, top_k, top_p: sampling hyperparameters
      • device: 'cpu' or 'cuda' or 'mps'
    """
    model.eval()
    input_ids = tokenizer.encode(prime_tokens)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_len):
        with torch.no_grad():
            out = model(input_tensor)  # shape: (1, seq_len, vocab_size)
            logits = out[0, -1] / temperature
            filtered = top_k_top_p_filtering(logits.clone(), top_k=top_k, top_p=top_p)
            probs = torch.softmax(filtered, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

        input_ids.append(next_id)
        if tokenizer.rev_vocab.get(next_id) == "<EOS>":
            break
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    return tokenizer.decode(input_ids)


def merge_prompt_and_generated(prompt_score: stream.Score, generated_score: stream.Score) -> stream.Score:
    """
    Build a side-by-side Score:
      - Keep every part in prompt_score as separate staves (SATB, etc.).
      - Append a new “Generated Continuation” part at the end (Piano).
    """
    merged = stream.Score()
    merged.insert(0, metadata.Metadata())
    merged.metadata.title = "Prompt + AI Generation"
    merged.metadata.composer = "AI Composer"

    # Preserve original parts
    for i, part in enumerate(prompt_score.parts):
        clone = part.flat.notesAndRests.stream()
        clone.id = part.id or f"Voice_{i}"
        clone.partName = part.partName or part.id or f"Voice_{i}"

        lname = clone.partName.lower()
        if "sop" in lname or lname == "s":
            clone.insert(0, instrument.Soprano())
        elif "alt" in lname or lname == "a":
            clone.insert(0, instrument.Alto())
        elif "ten" in lname or lname == "t":
            clone.insert(0, instrument.Tenor())
        elif "bass" in lname or lname == "b":
            clone.insert(0, instrument.Bass())
        else:
            clone.insert(0, instrument.Vocalist())

        merged.insert(i, clone)

    # Append generated continuation as a Piano part
    gen_part = stream.Part()
    gen_part.id = "Generated"
    gen_part.partName = "Generated Continuation"
    gen_part.insert(0, instrument.Piano())
    gen_part.append(generated_score.flat.notesAndRests)
    merged.insert(len(prompt_score.parts), gen_part)

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Load ABC corpus + trained REMI model, then generate new score from a prompt."
    )
    parser.add_argument(
        "--abc-input-dir",
        type=Path,
        required=True,
        help="Root directory of ABC files (training not run here)",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        required=True,
        help="Path to trained checkpoint (musicgen_remi_model.pt)",
    )
    parser.add_argument(
        "--prompt-file", type=Path, required=True, help="Single ABC file to use as prompt"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where generated ABC & MusicXML will be saved",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for inference",
    )
    parser.add_argument(
        "--max-len", type=int, default=200, help="Max number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k", type=int, default=50, help="Top-k sampling (0 to disable)"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling (0.0 to disable)"
    )
    args = parser.parse_args()

    abc_dir = args.abc_input_dir.resolve()
    model_ckpt = args.model_checkpoint.resolve()
    prompt_path = args.prompt_file.resolve()
    out_dir = args.output_dir.resolve()
    device = args.device
    max_len = args.max_len
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p

    if not abc_dir.is_dir():
        print(f"Error: ABC input directory not found: {abc_dir}")
        sys.exit(1)
    if not model_ckpt.is_file():
        print(f"Error: Model checkpoint not found: {model_ckpt}")
        sys.exit(1)
    if not prompt_path.is_file():
        print(f"Error: Prompt file not found: {prompt_path}")
        sys.exit(1)

    # 1) Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) Load & parse prompt
    prompt_score = converter.parse(str(prompt_path))
    orig_ts = prompt_score.recurse().getElementsByClass(meter.TimeSignature).first()
    orig_key_obj = prompt_score.analyze("key")

    # 3) Transpose prompt → C major/minor
    if orig_key_obj.mode == "major":
        target_key = m21key.Key("C")
    else:
        target_key = m21key.Key("C", "minor")
    iv_to_C = interval.Interval(orig_key_obj.tonic, target_key.tonic)
    prompt_in_C = prompt_score.transpose(iv_to_C)

    # 4) Tokenize prompt_in_C
    tokenizer = REMIABCTokenizer()
    tokens = tokenizer.tokenize(prompt_in_C)
    print("Prompt tokens (in C):")
    print(" ".join(tokens))

    # 5) Load trained model & checkpoint
    checkpoint = torch.load(str(model_ckpt), map_location=device)
    model = DecoderOnlyMusicGenModel(vocab_size=len(checkpoint["vocab"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # 6) Restore tokenizer vocab from checkpoint
    tokenizer.vocab = checkpoint["vocab"]
    tokenizer.rev_vocab = {v: k for k, v in tokenizer.vocab.items()}

    # 7) Generate continuation (in C)
    print("\nGenerating continuation (in C)...")
    generated_tokens = generate_sequence(
        model=model,
        tokenizer=tokenizer,
        prime_tokens=tokens,
        max_len=max_len,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
    )

    # Save generated tokens to a file
    gen_tokens_file = out_dir / "generated_tokens.txt"
    gen_tokens_file.write_text(" ".join(generated_tokens), encoding="utf-8")
    print(f"Saved generated tokens → {gen_tokens_file}\n")

    # 8) Detokenize → Score in C
    score_in_C = remi_tokens_to_score(generated_tokens)

    # 9) Reverse-transpose back to original key
    iv_to_orig = interval.Interval(target_key.tonic, orig_key_obj.tonic)
    score_in_orig = score_in_C.transpose(iv_to_orig)

    # 10) Re-insert original time signature & key
    if orig_ts:
        score_in_orig.insert(0, orig_ts)
    score_in_orig.insert(
        0,
        m21key.Key(
            orig_key_obj.tonic.name
            + ("major" if orig_key_obj.mode == "major" else "minor")
        ),
    )

    # 11) Merge prompt and generated side-by-side
    combined = merge_prompt_and_generated(prompt_score, score_in_orig)

    # 12) Save combined as ABC & MusicXML
    out_abc = out_dir / "prompt_vs_gen.abc"
    out_mxl = out_dir / "prompt_vs_gen.xml"
    combined.write("abc", fp=str(out_abc))
    combined.write("musicxml", fp=str(out_mxl))
    print(f"Saved side-by-side ABC      → {out_abc}")
    print(f"Saved side-by-side MusicXML → {out_mxl}")

    print("\nGeneration complete. All files are in:", out_dir)


if __name__ == "__main__":
    main()

```

## `train.py`
```python
# ai/train.py

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np
from music21 import converter, stream, key as m21key, interval, meter

import matplotlib.pyplot as plt

from remi_tokenizer import REMIABCTokenizer  # updated tokenizer
from relative_transformer import RelativeTransformerDecoder

# === Model definition ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

class DecoderOnlyMusicGenModel(nn.Module):
    """
    Even if you keep a single‐stage model, you can add a small “chord head” on top of the transformer’s 
    hidden states at bar boundaries. During training, you force it to predict the next bar’s chord.
    In your training loop:

    1. Collect chord_positions:
        - While tokenizing each C‐transposed score, record the indices of every <BarStart> that’s immediately followed by <Chord_...> in the token stream.
        - Create a tensor of shape (batch, n_bars) containing those indices.

    2. Compute two losses:
        - token_loss: cross‐entropy over next‐token at each position.
        - chord_loss: cross‐entropy between chord_logits[i, j, :] and the ground‐truth chord ID at bar j + 1.

    3. Combine them: loss = token_loss + λ * chord_loss (e.g. λ = 0.5).

    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 12,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_rel_dist: int = 1024,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional = PositionalEncoding(d_model, max_len=4096)

        self.decoder = RelativeTransformerDecoder( ... )
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Chord vocabulary: a separate small embedding & linear
        self.chord_vocab_size = len([tok for tok in tokenizer.vocab if tok.startswith("<Chord_")])
        self.chord_indices = {
            tok: idx for idx, tok in enumerate(tokenizer.vocab) if tok.startswith("<Chord_")
        }
        self.chord_head = nn.Linear(d_model, self.chord_vocab_size)

    def forward(self, x, chord_positions=None):
        """
        chord_positions: a list of token‐indices at which <BarStart> occurs.
        We'll extract the hidden state at those positions to predict the next chord.
        """
        emb = self.embedding(x)
        emb = self.positional(emb)
        tgt_mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), 1).bool()
        hidden = self.decoder(emb, tgt_mask=tgt_mask)  # (batch, seq_len, d_model)
        logits = self.fc_out(hidden)                  # (batch, seq_len, vocab_size)

        # If chord_positions is provided, we compute chord_logits:
        chord_logits = None
        if chord_positions is not None:
            # hidden_at_bar = hidden[:, chord_positions, :]  # gather at each bar boundary
            gathered = []
            for i, positions in enumerate(chord_positions):
                # positions: list of indices (e.g., [3, 10, 17, ...])
                # We want hidden[i, pos, :]
                gathered.append(hidden[i, positions, :])        # tensor (n_bars, d_model)
            # Pad to max bars, or stack if uniform length
            # For simplicity, assume each batch sample has the same number of bars:
            hidden_at_bar = torch.stack(gathered, dim=0)       # (batch, n_bars, d_model)
            chord_logits = self.chord_head(hidden_at_bar)      # (batch, n_bars, chord_vocab_size)

        return logits, chord_logits

# === Dataset, with transposition to C ===
class MusicREMI_Dataset(Dataset):
    def __init__(self, tokenizer: REMIABCTokenizer, scores: List[stream.Score]):
        self.tokenizer = tokenizer
        self.samples = []

        for score in scores:
            try:
                # 1) Analyze key of the original
                orig_key: m21key.Key = score.analyze("key")
                if orig_key.mode == "major":
                    target_key = m21key.Key("C")
                else:
                    target_key = m21key.Key("C", "minor")
                iv = interval.Interval(orig_key.tonic, target_key.tonic)
                score_C = score.transpose(iv)

                # 2) Tokenize the C‐transposed score
                tokens = tokenizer.tokenize(score_C)
                encoded = tokenizer.encode(tokens)
                self.samples.append(encoded)
            except Exception as e:
                print(f"❌ Failed to tokenize/transposition: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        x = seq[:-1]
        y = seq[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    ys = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0)
    return xs, ys


def extract_scores(folder_path: str) -> List[stream.Score]:
    scores = []
    for file in Path(folder_path).rglob("*.abc"):
        try:
            score = converter.parse(file)
            scores.append(score)
        except Exception as e:
            print(f"❌ Failed to parse {file}: {e}")
    return scores


# === Training & Evaluation Functions ===
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    losses = []
    for x, y in tqdm(dataloader, desc="Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)  # (batch, seq_len, vocab_size)
        out_flat = out.view(-1, out.size(-1))
        y_flat = y.view(-1)
        loss = criterion(out_flat, y_flat)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            out_flat = out.view(-1, out.size(-1))
            targets = y.view(-1)
            loss = criterion(out_flat, targets)
            total_loss += loss.item() * targets.size(0)
            total_tokens += targets.size(0)

            preds = out_flat.argmax(dim=-1)
            mask = targets != 0
            correct += (preds[mask] == targets[mask]).sum().item()

    perplexity = np.exp(total_loss / total_tokens)
    accuracy = correct / total_tokens
    return perplexity, accuracy


def plot_losses(losses):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss.png")
    print("📈 Saved training_loss.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder containing ABC files")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    default_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    args = parser.parse_args()

    print(f"📁 Loading scores from: {args.input}")
    scores = extract_scores(args.input)
    print(f"✅ Loaded {len(scores)} scores.")

    # Split into train / validation (80% / 20%)
    torch.manual_seed(0)
    n_total = len(scores)
    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val
    train_scores, val_scores = torch.utils.data.random_split(scores, [n_train, n_val])

    tokenizer = REMIABCTokenizer()
    train_dataset = MusicREMI_Dataset(tokenizer, train_scores)
    val_dataset = MusicREMI_Dataset(tokenizer, val_scores)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # At this point, tokenizer.vocab has been populated
    model = DecoderOnlyMusicGenModel(vocab_size=len(tokenizer.vocab)).to(args.device)

    # Use CrossEntropyLoss with label smoothing = 0.1
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    all_losses = []
    for epoch in range(args.epochs):
        print(f"\n🌀 Epoch {epoch+1}/{args.epochs}")
        train_losses = train_epoch(model, train_loader, optimizer, criterion, args.device)
        all_losses.extend(train_losses)

        train_ppl, train_acc = evaluate(model, train_loader, criterion, args.device)
        val_ppl, val_acc = evaluate(model, val_loader, criterion, args.device)
        print(f"   ▶️ Train  — Perplexity: {train_ppl:.2f}, Accuracy: {train_acc:.2%}")
        print(f"   ▶️ Valid  — Perplexity: {val_ppl:.2f}, Accuracy: {val_acc:.2%}")

    plot_losses(all_losses)

    # Save model + vocabulary
    torch.save(
        {"model_state_dict": model.state_dict(), "vocab": tokenizer.vocab},
        "musicgen_remi_model.pt",
    )
    print("✅ Model and vocab saved to musicgen_remi_model.pt")


if __name__ == "__main__":
    main()

```

## `train_light.py`
```python
# ai/train_light.py
"""
    4 layers, d_model=256
"""
import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np
from music21 import converter, stream, key as m21key, interval, meter

import matplotlib.pyplot as plt

from remi_tokenizer import REMIABCTokenizer  # updated tokenizer


# === Model definition ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class DecoderOnlyMusicGenModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)  # (batch, seq_len, d_model)
        emb = self.positional(emb)
        # causal mask: each position can attend only to itself and prior positions
        tgt_mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), 1).bool()
        memory = torch.zeros_like(emb)  # dummy memory
        out = self.decoder(emb, memory, tgt_mask=tgt_mask)
        return self.fc_out(out)  # (batch, seq_len, vocab_size)


# === Dataset, with transposition to C ===
class MusicREMI_Dataset(Dataset):
    def __init__(self, tokenizer: REMIABCTokenizer, scores: List[stream.Score]):
        self.tokenizer = tokenizer
        self.samples = []

        for score in scores:
            try:
                # 1) Analyze key of the original
                orig_key: m21key.Key = score.analyze("key")
                if orig_key.mode == "major":
                    target_key = m21key.Key("C")
                else:
                    target_key = m21key.Key("C", "minor")
                iv = interval.Interval(orig_key.tonic, target_key.tonic)
                score_C = score.transpose(iv)

                # 2) Tokenize the C‐transposed score
                tokens = tokenizer.tokenize(score_C)
                encoded = tokenizer.encode(tokens)
                self.samples.append(encoded)
            except Exception as e:
                print(f"❌ Failed to tokenize/transposition: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        x = seq[:-1]
        y = seq[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    ys = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0)
    return xs, ys


def extract_scores(folder_path: str) -> List[stream.Score]:
    scores = []
    for file in Path(folder_path).rglob("*.abc"):
        try:
            score = converter.parse(file)
            scores.append(score)
        except Exception as e:
            print(f"❌ Failed to parse {file}: {e}")
    return scores


# === Training & Evaluation Functions ===
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    losses = []
    for x, y in tqdm(dataloader, desc="Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)  # (batch, seq_len, vocab_size)
        out_flat = out.view(-1, out.size(-1))
        y_flat = y.view(-1)
        loss = criterion(out_flat, y_flat)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            out_flat = out.view(-1, out.size(-1))
            targets = y.view(-1)
            loss = criterion(out_flat, targets)
            total_loss += loss.item() * targets.size(0)
            total_tokens += targets.size(0)

            preds = out_flat.argmax(dim=-1)
            mask = targets != 0
            correct += (preds[mask] == targets[mask]).sum().item()

    perplexity = np.exp(total_loss / total_tokens)
    accuracy = correct / total_tokens
    return perplexity, accuracy


def plot_losses(losses):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss.png")
    print("📈 Saved training_loss.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder containing ABC files")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    default_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    args = parser.parse_args()

    print(f"📁 Loading scores from: {args.input}")
    scores = extract_scores(args.input)
    print(f"✅ Loaded {len(scores)} scores.")

    # Split into train / validation (80% / 20%)
    torch.manual_seed(0)
    n_total = len(scores)
    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val
    train_scores, val_scores = torch.utils.data.random_split(scores, [n_train, n_val])

    tokenizer = REMIABCTokenizer()
    train_dataset = MusicREMI_Dataset(tokenizer, train_scores)
    val_dataset = MusicREMI_Dataset(tokenizer, val_scores)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # At this point, tokenizer.vocab has been populated
    model = DecoderOnlyMusicGenModel(vocab_size=len(tokenizer.vocab)).to(args.device)

    # Use CrossEntropyLoss with label smoothing = 0.1
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    all_losses = []
    for epoch in range(args.epochs):
        print(f"\n🌀 Epoch {epoch+1}/{args.epochs}")
        train_losses = train_epoch(model, train_loader, optimizer, criterion, args.device)
        all_losses.extend(train_losses)

        train_ppl, train_acc = evaluate(model, train_loader, criterion, args.device)
        val_ppl, val_acc = evaluate(model, val_loader, criterion, args.device)
        print(f"   ▶️ Train  — Perplexity: {train_ppl:.2f}, Accuracy: {train_acc:.2%}")
        print(f"   ▶️ Valid  — Perplexity: {val_ppl:.2f}, Accuracy: {val_acc:.2%}")

    plot_losses(all_losses)

    # Save model + vocabulary
    torch.save(
        {"model_state_dict": model.state_dict(), "vocab": tokenizer.vocab},
        "musicgen_remi_model.pt",
    )
    print("✅ Model and vocab saved to musicgen_remi_model.pt")


if __name__ == "__main__":
    main()

```

## `train_two_stage.py`
```python
# ai/train_two_stage.py

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np
import random

from music21 import converter, stream, key as m21key, interval, meter

import matplotlib.pyplot as plt

from remi_tokenizer import REMIABCTokenizer
from ai.relative_transformer import RelativeTransformerDecoder  # your relative‐attention decoder
from remi_detokenizer import remi_tokens_to_score  # not used here, but for completeness


# ----------------------------------
# 1) Model Definitions
# ----------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class BarPlannerModel(nn.Module):
    """
    Stage 1: BarPlanner predicts a sequence of <Chord_...> tokens (plus global tokens like <time>, <key>, <Tempo>, <PhraseStart>).
    We feed it only the chord‐sequence portion of each score. It is a 6-layer, d_model=512 Transformer‐Decoder.
    """
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional = PositionalEncoding(d_model, max_len=1024)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                                                   dropout=dropout, activation="gelu", batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len_chord)
        returns: logits over full vocab, but only chord‐positions will be evaluated.
        """
        emb = self.embedding(x)            # (batch, seq_len_chord, d_model)
        emb = self.positional(emb)         # add positional encodings
        seq_len = x.size(1)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), 1).bool()
        memory = torch.zeros_like(emb)     # dummy memory
        out = self.decoder(emb, memory, tgt_mask=tgt_mask)
        return self.fc_out(out)            # (batch, seq_len_chord, vocab_size)


class DetailGeneratorModel(nn.Module):
    """
    Stage 2: DetailGenerator sees the entire full-token stream, including chord tokens at the start of each bar,
    and must predict the next token (notes, durations, velocities, barstarts, etc.). We build a large
    12-layer, d_model=1024 Transformer‐Decoder with relative attention.
    """
    def __init__(self, vocab_size: int, d_model: int = 1024, nhead: int = 16, num_layers: int = 12,
                 dim_feedforward: int = 4096, dropout: float = 0.1, max_rel_dist: int = 2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional = PositionalEncoding(d_model, max_len=8192)
        self.decoder = RelativeTransformerDecoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_rel_dist=max_rel_dist
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len_full)
        returns: (batch, seq_len_full, vocab_size)
        """
        emb = self.embedding(x)            # (batch, seq_len_full, d_model)
        emb = self.positional(emb)
        seq_len = x.size(1)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), 1).bool()
        out = self.decoder(emb, tgt_mask=tgt_mask)
        return self.fc_out(out)            # (batch, seq_len_full, vocab_size)


# ----------------------------------
# 2) Data Processing / Dataset Classes
# ----------------------------------

def extract_scores(folder_path: str) -> List[stream.Score]:
    """
    Recursively read all .abc files in folder_path and parse them with music21.
    Returns a list of music21.Score objects.
    """
    scores = []
    for file in Path(folder_path).rglob("*.abc"):
        try:
            score = converter.parse(str(file))
            scores.append(score)
        except Exception as e:
            print(f"❌ Failed to parse {file}: {e}")
    return scores


def extract_chord_sequence(full_tokens: List[str]) -> List[str]:
    """
    Given the full REMI token sequence for a score, filter out only:
      - <time=...>, <key=...>, <Tempo=...>, <PhraseStart>
      - <BarStart> <Chord_xxx> pairs
      - <EOS> at the end
    Ensures that every <BarStart> is immediately followed by its <Chord_...> token.
    """
    chord_seq = []
    saw_bos = False
    for i, tok in enumerate(full_tokens):
        if tok == "<BOS>":
            saw_bos = True
            chord_seq.append(tok)
            continue
        if not saw_bos:
            continue

        # Global tokens (only once at beginning)
        if tok.startswith("<time=") or tok.startswith("<key=") or tok.startswith("<Tempo=") or tok == "<PhraseStart>":
            chord_seq.append(tok)
            continue

        # BarStart → expect next token to be a Chord_...
        if tok == "<BarStart>":
            chord_seq.append(tok)
            # look ahead one token
            if i + 1 < len(full_tokens) and full_tokens[i + 1].startswith("<Chord_"):
                chord_seq.append(full_tokens[i + 1])
            else:
                chord_seq.append("<Chord_unk>")  # fallback if missing
            continue

        # <Chord_...> if it doesn’t directly follow <BarStart> is ignored
        # <EOS> at end
        if tok == "<EOS>":
            chord_seq.append(tok)
            break

    # Ensure <EOS> is present
    if chord_seq[-1] != "<EOS>":
        chord_seq.append("<EOS>")
    return chord_seq


def extract_full_sequence(full_tokens: List[str]) -> List[str]:
    """
    Just returns the full token list again (we assume remi_tokenizer.tokenize already produced
    a well‐formed, single <BOS> … <EOS> sequence).
    """
    return full_tokens.copy()


class BarChordDataset(Dataset):
    """
    Dataset for Stage 1: each item is a sequence of chord‐only tokens.
    We convert them to integer IDs here.
    """
    def __init__(self, tokenizer: REMIABCTokenizer, chord_token_lists: List[List[str]]):
        self.tokenizer = tokenizer
        self.encoded = []
        for chord_seq in chord_token_lists:
            ids = tokenizer.encode(chord_seq)
            self.encoded.append(ids)

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int):
        seq = self.encoded[idx]
        x = seq[:-1]
        y = seq[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class FullDetailDataset(Dataset):
    """
    Dataset for Stage 2: each item is the full REMI token sequence (chords + notes + everything).
    We convert them to integer IDs here.
    """
    def __init__(self, tokenizer: REMIABCTokenizer, full_token_lists: List[List[str]]):
        self.tokenizer = tokenizer
        self.encoded = []
        for full_seq in full_token_lists:
            ids = tokenizer.encode(full_seq)
            self.encoded.append(ids)

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int):
        seq = self.encoded[idx]
        x = seq[:-1]
        y = seq[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    xs, ys = zip(*batch)
    xs_padded = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    ys_padded = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0)
    return xs_padded, ys_padded


# ----------------------------------
# 3) Training Loops
# ----------------------------------

def train_bar_planner(
    model: BarPlannerModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: str,
    num_epochs: int = 5,
):
    """
    Standard training loop for the BarPlannerModel. Only uses chord sequences.
    """
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for x, y in tqdm(dataloader, desc=f"BarPlanner Epoch {epoch+1}/{num_epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)             # (batch, seq_len_chord, vocab_size)
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            loss = criterion(logits_flat, y_flat)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"    → BarPlanner Epoch {epoch+1} Loss: {avg_loss:.4f}")


def train_detail_generator(
    model: DetailGeneratorModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: str,
    num_epochs: int = 5,
):
    """
    Standard training loop for the DetailGeneratorModel. Uses full token sequences.
    """
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for x, y in tqdm(dataloader, desc=f"DetailGenerator Epoch {epoch+1}/{num_epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)             # (batch, seq_len_full, vocab_size)
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            loss = criterion(logits_flat, y_flat)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"    → DetailGenerator Epoch {epoch+1} Loss: {avg_loss:.4f}")


# ----------------------------------
# 4) Main: Putting It All Together
# ----------------------------------

def main():
    parser = argparse.ArgumentParser(description="Two-Stage Training: BarPlanner → DetailGenerator")
    parser.add_argument("--input", required=True, help="Path to folder containing ABC files")
    parser.add_argument("--bar_epochs", type=int, default=5, help="Epochs for BarPlanner")
    parser.add_argument("--detail_epochs", type=int, default=5, help="Epochs for DetailGenerator")
    parser.add_argument("--batch_size", type=int, default=8)
    default_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    args = parser.parse_args()

    print(f"📁 Loading scores from: {args.input}")
    all_scores = extract_scores(args.input)
    print(f"✅ Loaded {len(all_scores)} scores.")

    # 1) Build a unified vocabulary by tokenizing each score once
    tokenizer = REMIABCTokenizer()
    all_full_tokens: List[List[str]] = []

    print("🔨 Building full token lists (to grow vocab)…")
    for score in tqdm(all_scores, desc="Tokenizing Scores"):
        # 1a) Transpose to C major/minor
        orig_key = score.analyze("key")
        if orig_key.mode == "major":
            target_key = m21key.Key("C")
        else:
            target_key = m21key.Key("C", "minor")
        iv = interval.Interval(orig_key.tonic, target_key.tonic)
        score_C = score.transpose(iv)

        # 1b) Fully tokenize into REMI tokens
        full_tokens = tokenizer.tokenize(score_C)  # this also grows tokenizer.vocab
        all_full_tokens.append(full_tokens)

    print(f"🔑 Vocabulary size after scanning: {len(tokenizer.vocab)} tokens.")

    # 2) Extract chord-only sequences
    all_chord_sequences = [extract_chord_sequence(full) for full in all_full_tokens]

    # 3) Extract full detail sequences (just copy)
    all_detail_sequences = [extract_full_sequence(full) for full in all_full_tokens]

    # 4) Split into train/validation (80/20) by index
    random.seed(0)
    n = len(all_scores)
    idxs = list(range(n))
    random.shuffle(idxs)
    n_val = max(1, int(0.2 * n))
    train_idxs, val_idxs = idxs[n_val:], idxs[:n_val]

    # 5) Build BarChordDataset (train + val)
    train_chords = [all_chord_sequences[i] for i in train_idxs]
    val_chords = [all_chord_sequences[i] for i in val_idxs]
    train_chord_dataset = BarChordDataset(tokenizer, train_chords)
    val_chord_dataset   = BarChordDataset(tokenizer, val_chords)

    # 6) Dataloaders for BarPlanner
    bar_train_loader = DataLoader(train_chord_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    bar_val_loader   = DataLoader(val_chord_dataset,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 7) Instantiate BarPlannerModel
    bar_model = BarPlannerModel(vocab_size=len(tokenizer.vocab), d_model=512, nhead=8, num_layers=6, dropout=0.1)
    bar_optimizer = torch.optim.AdamW(bar_model.parameters(), lr=1e-4, weight_decay=1e-2)
    bar_criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 8) Pretrain BarPlannerModel
    print("\n🎬 Stage 1: Pretraining BarPlannerModel (chord sequences)")
    train_bar_planner(bar_model, bar_train_loader, bar_optimizer, bar_criterion, args.device, num_epochs=args.bar_epochs)

    # Optionally: evaluate on validation chord sequences
    bar_model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_tokens = 0
        for x, y in bar_val_loader:
            x, y = x.to(args.device), y.to(args.device)
            logits = bar_model(x)
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            loss = bar_criterion(logits_flat, y_flat)
            total_loss += loss.item() * y_flat.size(0)
            total_tokens += y_flat.size(0)
        val_perplexity = np.exp(total_loss / total_tokens)
        print(f"    ▶️ BarPlanner Validation Perplexity: {val_perplexity:.2f}")

    # 9) Build FullDetailDataset (train + val)
    train_details = [all_detail_sequences[i] for i in train_idxs]
    val_details   = [all_detail_sequences[i] for i in val_idxs]
    train_detail_dataset = FullDetailDataset(tokenizer, train_details)
    val_detail_dataset   = FullDetailDataset(tokenizer, val_details)

    detail_train_loader = DataLoader(train_detail_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    detail_val_loader   = DataLoader(val_detail_dataset,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 10) Instantiate DetailGeneratorModel
    detail_model = DetailGeneratorModel(
        vocab_size=len(tokenizer.vocab),
        d_model=1024,
        nhead=16,
        num_layers=12,
        dim_feedforward=4096,
        dropout=0.1,
        max_rel_dist=2048
    )
    detail_optimizer = torch.optim.AdamW(detail_model.parameters(), lr=1e-4, weight_decay=1e-2)
    detail_criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 11) Train DetailGeneratorModel
    print("\n🎬 Stage 2: Training DetailGeneratorModel (full detail sequences)")
    train_detail_generator(detail_model, detail_train_loader, detail_optimizer, detail_criterion, args.device, num_epochs=args.detail_epochs)

    # Optionally: evaluate on validation detail sequences
    detail_model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_tokens = 0
        correct = 0
        for x, y in detail_val_loader:
            x, y = x.to(args.device), y.to(args.device)
            logits = detail_model(x)
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            loss = detail_criterion(logits_flat, y_flat)
            total_loss += loss.item() * y_flat.size(0)
            total_tokens += y_flat.size(0)
            preds = logits_flat.argmax(dim=-1)
            mask = y_flat != 0
            correct += (preds[mask] == y_flat[mask]).sum().item()

        val_perplexity = np.exp(total_loss / total_tokens)
        val_accuracy = correct / total_tokens
        print(f"    ▶️ DetailGenerator Validation Perplexity: {val_perplexity:.2f}, Accuracy: {val_accuracy:.2%}")

    # 12) Save both models and the shared vocabulary
    torch.save({
        "bar_model_state_dict": bar_model.state_dict(),
        "detail_model_state_dict": detail_model.state_dict(),
        "vocab": tokenizer.vocab
    }, "two_stage_musicgen.pt")
    print("✅ Saved two-stage checkpoint to two_stage_musicgen.pt")


if __name__ == "__main__":
    main()

```

