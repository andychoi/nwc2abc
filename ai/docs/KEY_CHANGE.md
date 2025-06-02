To track key‐change metadata during normalization to C, and to emit/consume new `<key_change=…>` tokens. In short:

1. **In `REMIABCTokenizer`** (in `remi_tokenizer.py`):

   * Automatically transpose each incoming score to C (per‐segment if there are mid‐piece key signatures).
   * Extract all key‐signature changes from the original score and emit `<key_change=…>` tokens immediately after `<PhraseStart>`.
   * Continue tokenizing on the *normalized* (C-transposed) score for chords/notes/etc.

2. **In `remi_detokenizer.py`**:

   * Recognize `<key_change=…>` tokens and insert a new `KeySignature` at the current offset.

## Summary

1. **Normalization to C (per‐segment):**
   In `tokenize(...)`, we first scan `s_orig` for all `KeySignature` objects and record `(offset, "Gmaj")` etc. in `key_events`.
   We then build `s_norm` by cloning each part measure‐by‐measure, transposing each measure individually so that its local key → C (if major) or → A minor (if minor). From that point on, *all note/chord extraction runs on `s_norm`* (so the model “sees everything in C”).

2. **Emitting `<key_change=…>` tokens:**
   Right after `<PhraseStart>` (and before any chords), we loop over `key_events` and emit one `<key_change=TONICmode>` token for each mid‐piece key signature found in the original. These become part of the same token sequence that the encoder/decoder trains on.

3. **Detokenizer support (`remi_detokenizer.py`):**
   Whenever the detokenizer sees `<key_change=…>`, it pauses and does

   ```python
   ks2 = key.Key(tonic + "major" or "minor")
   s.insert(current_offset_quarters, ks2)
   ```

   i.e. we insert a new KeySignature at *exactly* the offset where that token appeared. Downstream, any score exporter (ABC/MusicXML) will then write out a proper key‐change markup at the right place.

With these two files in place, you do *not* need to touch any of your training/eval scripts. They will automatically:

* **Train** on everything expressed “in C,” but with explicit `<key_change=…>` tokens so the model can learn realistic modulations.
* **At inference**, generate those same modulation tokens, and then the detokenizer will re‐insert real `KeySignature` objects at the right offsets so the final score actually has the key changes.

All existing calls to `tokenizer.tokenize(...)` work exactly as before—now with full mid‐piece key normalization + metadata.
