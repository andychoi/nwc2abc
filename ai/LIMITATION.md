
# Version 0.3

## Limitations & Next Steps

* **Compute Budget & Data**

  * A 12‐layer, 512‐dim model with relative attention requires substantial GPU memory. You’ll need large‐memory GPUs (A100/RTX3090) or gradient‐checkpointing to train on tens of thousands of scores.
  * Gather a multi‐genre training corpus: Bach chorales, classical piano (MAESTRO), jazz lead sheets, folk tunes, pop MIDI. Convert everything to ABC or MusicXML so that your REMI tokenizer can parse it.

* **Dynamic Token Granularity**

  * We quantize relative position to `ticks_per_beat=4` (16th‐notes). If you want trills, tuplets, or micro‐timing, you might need to increase to `ticks_per_beat=8` or `16`—but that expands sequence length dramatically.

* **Hierarchical Planning**

  * A two‐stage approach (bar‐planner → note‐detail) may be more stable for very long sequences (e.g., 100+ bars). A purely autoregressive model still needs to learn long‐range structure.

* **Expressive Performance**

  * We added `<Dynamic_...>` and `<Tempo=...>`, but you could further enrich with `<Articulation_staccato>`, `<Pedal_on/off>`, or `<Accent>`, particularly if you have MIDI with pedal/dynamics annotations.

* **User‐Interactive Features**

  * As a next step, build a small notebook or web UI where the user can:

    1. Upload a chord progression (e.g. “I – IV – V – I”).
    2. Click “Generate” to have the model fill in SATB/piano under that structure.
    3. Adjust `<Style=...>`, `<Tempo=...>`, and see real‐time updates.

* **Advanced Metrics**

  * For a “GPT‐level” subjective evaluation, run small listening studies, or integrate pretrained style‐discrimination classifiers (e.g. “Is this sample Bach‐like?”).


# Version 0.2

* **Relative Positions** (`<RelPos_n>`): Model now sees how many ticks since the last event in each part, rather than an absolute “Position in Bar.” This often helps learning repeated patterns.
* **Chord Tokens** (`<Chord_I>`, `<Chord_ii7>`, etc.): These give the model an explicit harmonic plan each measure, making generated harmonies more coherent.
* **Dynamics** (`<Dynamic_p>`, `<Dynamic_mf>`, etc.): Whenever a note has a dynamic marking in the ABC/MusicXML, the tokenizer emits that token just before the note, so the model can learn expressive changes.
* **Flexible Time Signatures**: `<time=3/4>`, `<time=6/8>`, `<time=5/4>`, etc., let the model handle any meter.
* **Key Tokens**: `<key=Cmaj>`, `<key=Am>`, etc., are emitted up front so the model knows tonal context.
* **Label Smoothing**: Prevents over-confidence in next‐token prediction.
* **Top-k / Top-p Sampling**: More controlled decoding, producing higher coherence than pure multinomial.
* **Validation Split**: You can now track train vs. validation perplexity/accuracy to detect overfitting.
* **Musical Metrics**: KL divergence of pitch classes and rhythmic violation counts give you objective, numeric feedback on the “musical plausibility” of your generated output.

With these adjustments, your pipeline is now much closer to a “GPT-level” symbolic music model—able to handle:

* Arbitrary keys & time signatures
* Explicit harmonic planning (chords)
* Relative rhythmic structure
* Dynamics & expressivity
* Advanced decoding strategies
* Robust, musically-informed evaluation

Feel free to tweak further (e.g. add chord‐prediction as an auxiliary head, incorporate relative‐position encodings in the Transformer itself, or gather a larger, more diverse training corpus). But this fully working code should serve as a solid foundation for generating high-quality, musically coherent SATB (and multi-part/piano) output.


# Version 0.1

Below are some of the main limitations of the current training / evaluation setup, followed by concrete suggestions for how to push it closer to a “GPT-level” music model.

---

## 1. Model & Architecture Limitations

1. **Relatively Small Transformer**

   * We use a 4-layer, 256-dim decoder with 4 attention heads. That’s fine for experimentation, but it cannot learn very long-range dependencies (e.g. overarching musical form, large phrase structures) as well as larger models (hundreds of millions or billions of parameters).
   * Because it’s shallow, the receptive field is limited. In practice, this often leads to local coherence (notes make sense locally) but a failure to maintain motifs over dozens of bars.

2. **Fixed‐Length Sinusoidal Positional Encoding**

   * Currently we add a basic sinusoidal (or learned) position encoding. But recent “GPT-like” music models (e.g. Music Transformer, Pop Music Transformer) use **relative** position embeddings or “local” attention windows to better capture musical repeats, transpositions, and long-distance relationships.
   * Without relative encodings or advanced schemes, the model struggles to learn “this bar is a repeat of bar 12” or “this motif is transposed up a fifth” patterns.

3. **Purely Autoregressive without Enhanced Attention Masking**

   * We use a causal mask so that each token only sees past tokens. That’s correct for generation, but more advanced models sometimes incorporate auxiliary streams—e.g. a small chord‐tracking network, separate “beat/measure” streams, or extra side inputs (tempo changes, dynamics) that can improve rhythmic consistency.
   * We also feed a dummy “memory” (zeros) into the TransformerDecoder. A true GPT‐style model would be a stack of masked decoder blocks without introducing dummy memory. Our current pattern is functionally similar, but slightly unconventional and missing possible gains from a strictly decoder-only architecture.

---

## 2. Tokenization & Musical Representation Limitations

1. **Event Quantization to Fixed Ticks Per Beat**

   * We use a fixed `ticks_per_beat` (e.g. 4). That means sixteenth notes (or shorter) get quantized, but anything smaller (grace notes, tuplets like triplets in 6/8) is approximated. This can lose expressive microtiming information.
   * Real musical expression often involves swing, rubato, tuplets, and metric modulation. Our token scheme can’t represent that nuance—everything is in fixed 16th-note (or 8th-note) chunks.

2. **Limited Dynamics and Articulation**

   * We include only a single `<Velocity_n>` token per note. True expressive performance includes crescendos, diminuendos, articulations (staccato, legato), accents, fermatas, and so on. Capturing that requires extra token streams (e.g. `<Dynamic_p>`, `<Articulation_staccato>`), and only training on raw MIDI velocity bins is fairly crude.

3. **Simple Chord/Polyphony Handling**

   * When encountering a chord, we emit multiple `<Note-On_pitch>` tokens at the same position. That works for four‐note blocks, but there’s no “Chord\_III7” abstraction or harmonic label.
   * Many advanced models incorporate harmonic “control” tokens (e.g. “this chord is G7” or “this bar is ii–V–I”) so that the network can learn chord‐centric generation rather than treating each pitch independently.

4. **One‐Hot Voice Tags Only**

   * We prepend `<voice=S>`, `<voice=A>`, etc. But we do not allow, for example, “this piano part has left-hand chord pattern” vs. “right-hand melody.” If you want more granular multi-instrument control, you’d need tokens like `<instr=Piano-LH>` vs. `<instr=Piano-RH>` vs. `<instr=Organ>`, etc., and a richer scheme to specify orchestration cues.

---

## 3. Dataset & Data Limitations

1. **Limited Corpus Size and Diversity**

   * If you’re only training on a small folder of ABC choral scores, the model sees limited harmonic vocabulary (e.g. mostly standard chorales). A GPT-level music model typically trains on thousands of hours of MIDI or symbolic data across many genres (jazz, classical, pop).
   * Without huge, diverse data, the model can “overfit” to Bach-like chorale patterns and struggle to generalize.

2. **No Validation/held-Out Split**

   * We currently load all `.abc` files into a single `Dataset` and train on them. There’s no dedicated validation or test split. To know if the model is truly learning rather than overfitting, we need to set aside 10–20% of the songs as unseen during training and track perplexity/accuracy on that split each epoch.

---

## 4. Training & Optimization Limitations

1. **Basic Cross-Entropy Loss**

   * We use standard token‐level cross‐entropy. But music generation often benefits from additional objectives:

     * **Label Smoothing** (to prevent over-confidence)
     * **Sequence-level Losses** (e.g. comparing generated n-gram distributions to ground truth, or using BLEU-like music metrics)
     * **Contrastive or Adversarial Losses** (train a discriminator to classify “real vs. generated” chorales)

2. **Sampling Strategy**

   * We do simple multinomial sampling (“next token \~ softmax(logits/temperature)”). More advanced decoding uses:

     * **Top-k Sampling** (restrict to top k probable tokens)
     * **Nucleus (Top-p) Sampling** (restrict to smallest set whose cumulative probability ≥ p)
     * **Beam Search** (especially if you want highest‐likelihood continuation rather than random sampling)
   * Without these, generation can produce less coherent or overly random sequences.

---

## 5. Evaluation Limitations

1. **Perplexity & Token‐Accuracy Only**

   * We compute perplexity and token‐accuracy on the training or validation set. Those metrics tell you how well the model predicts the next token in the data, but not whether the music *sounds* good or follows musical rules.
   * We lack **musical-specific** evaluation:

     * **Pitch Class Histogram KL-divergence** (compare distribution of pitch classes in generated vs. real chorales)
     * **Rhythmic Consistency Metrics** (e.g. is total duration per measure correct? How often do we violate metric grid?)
     * **Harmonic Coherence** (e.g. percentage of bars that form a diatonic chord in the detected key)

2. **No Human Listening Tests**

   * Ultimately, a music model is judged by whether listeners find the output musically plausible. We’re not running any subjective tests, nor are we using pretrained “music classifiers” to detect style adherence.

---

## Next Steps to Improve Further

1. **Scale Up Model & Data**

   * **Larger Transformer**: Move from 4 layers to at least 12–24 layers; increase hidden dimension (512–1024) and attention heads.
   * **More Data**: Aggregate a broad corpus of symbolic pieces (e.g. Bach chorales, classical piano, folk tunes, jazz standards). Aim for at least tens of thousands of pieces across various genres.
   * **Mixed-Genre Pretraining**: Pretrain a bigger model on a huge, generic MIDI corpus for a few epochs, then fine-tune on chorales or specialized collections.

2. **Improve Musical Representation**

   * **Relative Positional Encodings**: Switch to the scheme used in Music Transformer (shaw-style relative attention), which helps the network more easily copy motifs at different bar offsets.
   * **Hierarchical Tokens**: Instead of only event tokens at 16th-note intervals, introduce bar-level and phrase-level tokens (e.g. `<BarStart>`, `<PhraseStart>`). A two-tiered “bar + position” scheme can help the model understand form.
   * **Chord & Harmony Tokens**: Precompute a chord label for each measure (e.g. `ii7`, `V7`) and insert `<Chord_ii7>` tokens at the start of each bar. That explicitly guides the model to generate pitch events consistent with a harmonic plan.

3. **Add Expressive & Performance Tokens**

   * **Dynamics (f, mf, p)**: Insert tokens like `<Dynamic_mf>` at certain positions, rather than just raw velocity numbers.
   * **Articulations**: Include `<Articulation_staccato>`, `<Articulation_legato>`, `<Pedal_on>`, `<Pedal_off>`.
   * **Tempo & Tempo Changes**: Rather than hardcoding 120 bpm, add `<Tempo=90>`, `<Tempo=120>` tokens whenever the tempo changes.

4. **Refine Training Objectives**

   * **Label Smoothing** (e.g. smoothing = 0.1) to prevent the model from becoming overly confident.
   * **Scheduled Sampling / Curriculum**: In early epochs, force more teacher-forcing; later, let the model sample its own tokens to reduce exposure bias.
   * **Auxiliary Losses**: E.g., a chord classifier head on top of encoder outputs: the model must both predict next token and predict the current chord label.

5. **Advanced Decoding Strategies**

   * Implement **top-k/top-p sampling** for generation, which typically yields more coherent, less random continuations.
   * Add **beam search** option when you want the single best likelihood sequence rather than randomized sampling.
   * Consider **length-control tokens**: Insert a `<Length=16_bars>` token at the start so the model knows how many bars to generate.

6. **Metrics & Evaluation**

   * Set aside 10–20% of data as a validation split. Track perplexity & token-accuracy on that split each epoch.
   * Compute **pitch class histogram distance**: e.g., KL-divergence between pitch distribution of generated vs. real chorales.
   * Compute **rhythmic violation counts**: how many notes violate the bar grid or exceed measure length.
   * Run small **listening tests**: Have musicians rate 20-bar excerpts of generated vs. real data on “musicality,” “coherence,” “style adherence.”

7. **Interactive & Conditioning Controls**

   * Add conditioning tokens for **mood**, **tempo**, or **style** (e.g. `<Style=Bach>`, `<Style=Jazz>`). During generation, you can prefix with these to steer output.
   * Expose **interactive interfaces** (e.g. Jupyter + widgets) so a user can tweak temperature, top-k, or manually edit chord tokens before generating.

8. **Long-Form Structure & Hierarchical Models**

   * For truly long pieces (multiple minutes), a single Transformer can struggle. Consider a **hierarchical approach**: a “bar-level” transformer that generates a chord/structure plan, then a “note-level” transformer that fills in details.
   * Or use **recurrent attention chunks** (e.g. windowed attention) so the model can handle thousands of tokens without blowing up memory.

---

### In Summary

* **Current Limitations**: small model, fixed quantization, minimal dynamics, no chord tokens, modest dataset, simple loss, limited evaluation metrics.
* **Next Steps**: scale up both model and data, enhance tokenization (relative positions, chords, dynamics, flexible TS), adopt advanced training objectives, refine decoding, and institute robust, musically informed evaluation.

By systematically addressing these areas, you’ll move from a solid research prototype toward a production-level, GPT-style symbolic music generator.
