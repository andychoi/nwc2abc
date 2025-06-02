The two modules (`ai/bar_planner.py` and `ai/detail_generator.py`) were provided as conceptual sketches for a two-stage “planner → detail” architecture, but they weren’t wired into the single-stage training/evaluation scripts as-is. In other words:

1. **They’re purely illustrative**

   * We showed how you *could* split the workflow into a Bar-Planner (predicting a sequence of `<Chord_…>` tokens) and a Detail-Generator (filling in notes given that chord plan).
   * However, the actual `train.py` and `eval.py` we shipped remain a *single-stage* model that directly generates all tokens (including chords and notes) in one pass.

2. **No data pipeline to extract “chord-only” sequences**

   * To train `BarPlannerModel`, you’d first need to extract, for each transposed-C score, a pure “BarStart→Chord” token sequence.
   * Likewise, to train `DetailGeneratorModel`, you’d need to feed it the ground-truth chord tokens (from Bar-Planner) plus any `<BarStart>` markers, then supervise it on the full note-level stream.
   * Those extraction steps and new Dataset classes were never implemented in the single-stage code.

3. **No unified training loop for two-stage**

   * Integrating `BarPlannerModel` and `DetailGeneratorModel` requires:

     1. A Dataset that yields `(chord_sequence, note_sequence)` pairs for each score.
     2. A training loop that first fits the Bar-Planner on chord sequences, then either uses its forecasts or the gold chord tokens to train the Detail-Generator.
   * Because we stayed with a one-stage approach in `train.py`, those two new classes simply never get called.

4. **Runtime imports missing**

   * Neither `train.py` nor `eval.py` ever `import bar_planner` or `detail_generator`. As a result, Python never instantiates or uses them.

---

### How to actually use those modules

If you want to adopt the two-stage approach, you would need to:

1. **Extend your Dataset:**

   * Loop through each (transposed-C) score.
   * Build a list of “bar-planner tokens,” i.e.

     ```
     <BOS> <time=…> <key=…> <BarStart> <Chord_X> <BarStart> <Chord_Y> … <EOS>
     ```
   * Build a corresponding list of full note-level tokens:

     ```
     <BOS> <Style=…> <time=…> <key=…> <Tempo=…> <PhraseStart> <Chord_X> … <voice=S> <BarStart> <RelPos…> <Note-On…> … <EOS>
     ```
   * Return both in `__getitem__`: `(planner_tokens, detail_tokens)`.

2. **Train the Bar-Planner:**

   ```python
   bar_model = BarPlannerModel(vocab_size=…, d_model=512, nhead=8, num_layers=6)
   optimizer = torch.optim.AdamW(bar_model.parameters(), lr=…)
   for epoch in range(…):
       for planner_tokens, _ in dataloader:
           # teacher-force next-token prediction on planner_tokens
           logits = bar_model(planner_tokens_input)
           loss = cross_entropy(logits, planner_tokens_target)
           …
   ```

3. **Train the Detail-Generator:**

   ```python
   detail_model = DetailGeneratorModel(vocab_size=…, d_model=1024, nhead=16, num_layers=12)
   optimizer = torch.optim.AdamW(detail_model.parameters(), lr=…)
   for epoch in range(…):
       for planner_tokens, detail_tokens in dataloader:
           # Option A: use *ground-truth* planner_tokens as context
           prefix = planner_tokens  
           # Option B: use bar_model to generate its own chord sequence: 
           #         prefix = bar_model.generate(planner_prompt)  
           x = torch.cat([prefix, detail_tokens[:-1]], dim=1)
           logits = detail_model(x)
           loss = cross_entropy(logits, detail_tokens_target)
           …
   ```

4. **Inference:**

   * First run the Bar-Planner (possibly conditioned on `<Style=...> <time=...> <key=...>`) to produce N chord tokens.
   * Concatenate its output with a `<BarStart>` marker and feed that entire token sequence into the Detail-Generator to sample the note events.

---

### Why they were omitted in the single-stage code

* We decided to keep the initial pipeline simpler (one Transformer that handles everything).
* Implementing a full two-stage “planner + detail” requires extra data processing, two full training loops, and careful scheduling (e.g. do you pretrain the Bar-Planner first, or train both jointly?).
* As a result, those two classes remain templates—you can copy/paste and integrate them as you wish, but the out-of-the-box `train.py` and `eval.py` do *not* use them.

---

Below is a complete, self-contained Python script that implements a *two-stage* “Planner → Detail” training pipeline. It assumes you already have:

1. **`ai/remi_tokenizer.py`** (the enhanced REMI tokenizer from before), which builds a shared vocabulary and can tokenize each Score into a full token stream (including `<Chord_…>` tokens).
2. **`ai/relative_transformer.py`** (which defines `RelativeTransformerDecoder`).
3. **`ai/remi_detokenizer.py`** (for later inference).

Save the following as, for example, **`ai/train_two_stage.py`**. This script will:

1. **Build a unified vocabulary** by scanning every score with `REMIABCTokenizer.tokenize(...)`.
2. **Extract “Chord-Only” sequences** (for the Bar-Planner) and **Full “Detail” sequences** (for the Detail-Generator).
3. **Pretrain a BarPlannerModel** on chord sequences alone.
4. **Freeze (or leave unfrozen) the BarPlannerModel**, then train a DetailGeneratorModel on full detail sequences, always providing the ground-truth chord tokens as context.

Feel free to adapt the hyperparameters (model sizes, learning rates, number of epochs, etc.) to your hardware and dataset size.

---

### Explanation of Key Steps

1. **Vocabulary Construction**

   * We instantiate a single `REMIABCTokenizer()` and loop over every score (transposed to C major/minor) to call `tokenizer.tokenize(score_C)`. This populates `tokenizer.vocab` (and `rev_vocab`) with *all* possible tokens found across the dataset: chord tokens, voice tokens, bar markers, note events, dynamics, etc.

2. **Extracting Chord‐Only Sequences**

   * Once you have each full token list (`full_tokens = tokenizer.tokenize(...)`), we call `extract_chord_sequence(full_tokens)` which:

     * Keeps `<BOS>`, global tokens `<time=…>`, `<key=…>`, `<Tempo=…>`, `<PhraseStart>`,
     * Whenever it sees `<BarStart>`, it appends `<BarStart>` plus the very next `<Chord_…>` token (or `<Chord_unk>` if missing),
     * Finally appends `<EOS>`.
   * The resulting `chord_seq` might look like:

     ```
     ["<BOS>",
      "<time=4/4>",
      "<key=Cmaj>",
      "<Tempo=120>",
      "<PhraseStart>",
      "<BarStart>", "<Chord_I>",
      "<BarStart>", "<Chord_IV>",
      "<BarStart>", "<Chord_V>",
      "<EOS>"]
     ```

3. **Extracting Full Detail Sequences**

   * We simply keep each full REMI token list in `all_detail_sequences`. This includes chord tokens, voice tags, bar starts, relative positions, note‐on, durations, velocities, and `<EOS>`.

4. **Stage 1: BarPlannerModel**

   * **Dataset:** `BarChordDataset`, which takes `List[List[str]]` of chord‐only token sequences, calls `tokenizer.encode(...)` to map to integer IDs, and prepares `(x=seq[:-1], y=seq[1:])`.
   * **Model:** `BarPlannerModel(vocab_size, d_model=512, nhead=8, num_layers=6)`.
   * **Training Loop:** we train it for `--bar_epochs` epochs. At each step we compute cross‐entropy over the next‐token in the chord vocabulary (which is a subset of the full vocabulary—but we still index into the full embedding matrix).
   * **Validation:** we compute perplexity on the held‐out chord sequences.

5. **Stage 2: DetailGeneratorModel**

   * **Dataset:** `FullDetailDataset`, which takes the *full* token lists, calls `tokenizer.encode(...)`, and again returns `(x=full_seq[:-1], y=full_seq[1:])`.
   * **Model:** `DetailGeneratorModel(vocab_size, d_model=1024, nhead=16, num_layers=12, max_rel_dist=2048)`. This uses `RelativeTransformerDecoder` so that each layer has relative positional biases.
   * **Training Loop:** for `--detail_epochs` epochs, we train on the full sequences. Because each full sequence *already* contains its chord tokens at the front of each bar (via `<BarStart>` & `<Chord_…>`), the model sees “the correct chord plan” at train time.
   * **Validation:** compute perplexity & token‐level accuracy on held‐out detail sequences.

6. **Saving**

   * At the end, we save both `bar_model.state_dict()` and `detail_model.state_dict()` into a single checkpoint `"two_stage_musicgen.pt"`, plus the shared `vocab`. During inference, you can load both sub‐models and run:

     1. Use `BarPlannerModel` to generate a chord sequence for N bars, given `<BOS> <time=…> <key=…> <Tempo=…> <PhraseStart>`.
     2. Concatenate that generated chord sequence (with `<BarStart>` markers) to form a prefix.
     3. Feed that prefix into `DetailGeneratorModel` to sample the note‐level continuation.
     4. Reverse‐transpose from C back to the original key, reinsert time signature, and merge with any SATB prompt if desired.

---

### How to Run

1. **Train Both Stages**

   ```bash
   python ai/train_two_stage.py \
     --input /path/to/your/abc_corpus \
     --bar_epochs 5 \
     --detail_epochs 5 \
     --batch_size 8 \
     --device cuda
   ```

   This will produce `two_stage_musicgen.pt` containing:

   ```json
   {
     "bar_model_state_dict": { … },
     "detail_model_state_dict": { … },
     "vocab": { "token_string": token_id, … }
   }
   ```

2. **Inference Sketch**
   After training, you can do something like:

   ```python
   import torch
   from remi_tokenizer import REMIABCTokenizer
   from ai.remi_detokenizer import remi_tokens_to_score
   from ai.train_two_stage import BarPlannerModel, DetailGeneratorModel
   from music21 import converter, interval, key as m21key, meter

   # 1) Load checkpoint
   ckpt = torch.load("two_stage_musicgen.pt", map_location="cpu")
   vocab = ckpt["vocab"]

   # 2) Reconstruct tokenizer
   tokenizer = REMIABCTokenizer()
   tokenizer.vocab = vocab
   tokenizer.rev_vocab = {v: k for k, v in vocab.items()}

   # 3) Instantiate BarPlannerModel and load weights
   bar_model = BarPlannerModel(vocab_size=len(vocab), d_model=512, nhead=8, num_layers=6)
   bar_model.load_state_dict(ckpt["bar_model_state_dict"])
   bar_model.eval()

   # 4) Instantiate DetailGeneratorModel and load weights
   detail_model = DetailGeneratorModel(vocab_size=len(vocab),
                                       d_model=1024, nhead=16, num_layers=12,
                                       dim_feedforward=4096, dropout=0.1, max_rel_dist=2048)
   detail_model.load_state_dict(ckpt["detail_model_state_dict"])
   detail_model.eval()

   # 5) Prepare a prompt Score and transpose to C
   prompt_score = converter.parse("path/to/prompt.abc")
   orig_key = prompt_score.analyze("key")
   if orig_key.mode == "major":
       target_key = m21key.Key("C")
   else:
       target_key = m21key.Key("C", "minor")
   iv_to_C = interval.Interval(orig_key.tonic, target_key.tonic)
   prompt_in_C = prompt_score.transpose(iv_to_C)

   # 6) Build chord-only prefix tokens for Stage 1
   tokens_prompt_full = tokenizer.tokenize(prompt_in_C)
   chord_prefix = extract_chord_sequence(tokens_prompt_full)

   # 7) Generate chord continuation with BarPlannerModel
   device = "cuda" if torch.cuda.is_available() else "cpu"
   bar_model.to(device)

   # Encode prefix, then autoregressively sample next chords
   input_ids = tokenizer.encode(chord_prefix)
   input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
   max_bars_to_generate = 16  # e.g. generate 16 bars
   for _ in range(max_bars_to_generate):
       with torch.no_grad():
           logits = bar_model(input_tensor)
           next_logits = logits[0, -1, :]
           next_id = torch.argmax(next_logits).item()  # greedy; or use top-k/top-p sampling
       input_ids.append(next_id)
       if tokenizer.rev_vocab[next_id] == "<EOS>":
           break
       input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

   generated_chord_seq = tokenizer.decode(input_ids)

   # 8) Build a full‐token prefix for Stage 2 by merging:
   #    (a) the generated chord tokens (which include <BarStart><Chord_...> pairs),
   #    (b) the first “detail” tokens from the original prompt (if you want to continue past prompt).
   #    For simplicity, here we just generate from scratch after chord plan:
   full_prefix = generated_chord_seq.copy()
   # If you want to include notes of prompt first, you could do:
   #   prompt_detail = extract_full_sequence(tokens_prompt_full)
   #   full_prefix = prompt_detail[:-1] + generated_chord_seq

   # 9) Generate note‐level continuation with DetailGeneratorModel
   detail_model.to(device)
   input_ids_2 = tokenizer.encode(full_prefix)
   input_tensor_2 = torch.tensor([input_ids_2], dtype=torch.long, device=device)
   max_tokens_to_generate = 500  # however many tokens you want
   for _ in range(max_tokens_to_generate):
       with torch.no_grad():
           logits2 = detail_model(input_tensor_2)
           next_logits2 = logits2[0, -1, :]
           next_id2 = torch.multinomial(torch.softmax(next_logits2, dim=-1), 1).item()
       input_ids_2.append(next_id2)
       if tokenizer.rev_vocab[next_id2] == "<EOS>":
           break
       input_tensor_2 = torch.tensor([input_ids_2], dtype=torch.long, device=device)

   generated_full_seq = tokenizer.decode(input_ids_2)

   # 10) Detokenize to a Score in C, then transpose back to original key
   score_in_C = remi_tokens_to_score(generated_full_seq)
   iv_to_orig = interval.Interval(target_key.tonic, orig_key.tonic)
   final_score = score_in_C.transpose(iv_to_orig)

   # Reinsert original time signature and key
   orig_ts = prompt_score.recurse().getElementsByClass(meter.TimeSignature).first()
   if orig_ts:
       final_score.insert(0, orig_ts)
   final_score.insert(0, orig_key)

   # Show or save
   final_score.show()
   ```

---

### Summary

* **Stage 1 (BarPlanner):**

  * We extract a compact token sequence for each score that contains only global tokens (`<time=...>`, `<key=...>`, `<Tempo=...>`, `<PhraseStart>`) plus `<BarStart>` → `<Chord_...>` pairs at every measure.
  * We train a 6-layer, d\_model=512 Transformer-Decoder (`BarPlannerModel`) to predict the *next chord token* at each time step. This forces the model to learn harmonic progression at the bar level.

* **Stage 2 (DetailGenerator):**

  * We take the *full* REMI token sequence (including chord tokens, voice tags, bar markers, relative positions, note events, durations, velocities, dynamics, etc.) and train a 12-layer, d\_model=1024 Transformer-Decoder with **relative attention** (`DetailGeneratorModel`).
  * During training, the model sees the *ground truth* chord plan at each bar, so it can condition on that and learn to generate the note‐level detail that fits those chords.

* **Inference Flow:**

  1. **Chord Planning:** Given a prompt (or “cold start”), build the chord‐only prefix and sample a chord progression with the BarPlannerModel.
  2. **Note Generation:** Take those generated `<BarStart><Chord_…>` tokens as a prefix for the DetailGeneratorModel (optionally preceded by the prompt’s note‐level tokens), then autoregressively sample the rest of the note‐level events.
  3. **Transpose & Render:** Detokenize back into a `music21.Score`, transpose from C back to the original key, reinsert the original time signature, and display or export.

By explicitly “pretraining” the BarPlanner first, you give the model a chance to master harmonic structure before attempting the much tougher problem of generating multi‐voice note events. This two‐stage separation often yields more coherent, “musically aware” outputs.

Feel free to adjust hyperparameters (e.g. number of layers, hidden dimension, learning rates), add dropout, or experiment with different sampling strategies (top-k, top-p) at inference time. Once you have a reasonable chord planner, you could even **freeze its weights** and only train the detail model, or **finetune both jointly** in a later phase.

Good luck, and happy modeling!
