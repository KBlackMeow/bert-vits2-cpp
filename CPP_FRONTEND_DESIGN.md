# Pure C++ Frontend Design

This runtime must not depend on Python at inference time. Python is only the
behavior reference used to validate the C++ implementation.

## Goal

Match the Bert-VITS2 Python frontend contract:

```text
text + language
  -> norm_text
  -> phones, tones, word2ph
  -> phone_ids, tone_ids, language_ids
  -> add_blank adjustment
  -> active language BERT
  -> three phone-level BERT channels
  -> VITS2 ONNX runtime
```

The synthesizer must only consume phone-level tensors. It must not know how a
language normalizes text, generates phones, tokenizes BERT input, or aligns BERT
tokens to phones.

## Frontend Contract

Every language frontend returns a raw, pre-blank result:

```cpp
struct RawFrontend {
    std::string norm_text;
    std::vector<std::string> phone_symbols;
    std::vector<int64_t> tones;
    std::vector<std::string> bert_tokens;
    std::vector<int64_t> word2ph;
};
```

Required invariants before blank insertion:

```text
phone_symbols.size() == tones.size()
phone_symbols.size() == sum(word2ph)
word2ph.front() == 1
word2ph.back() == 1
phone_symbols.front() == "_"
phone_symbols.back() == "_"
```

The assembled runtime result is:

```cpp
struct FrontendResult {
    std::string norm_text;
    std::vector<int64_t> phone_ids;
    std::vector<int64_t> tone_ids;
    std::vector<int64_t> language_ids;
    std::vector<int64_t> word2ph;
    Tensor bert_zh; // [T, 1024]
    Tensor bert_jp; // [T, 1024]
    Tensor bert_en; // [T, 1024]
};
```

Required invariants after blank insertion:

```text
T == phone_ids.size()
T == tone_ids.size()
T == language_ids.size()
T == sum(word2ph)
bert_zh.shape == [T, 1024]
bert_jp.shape == [T, 1024]
bert_en.shape == [T, 1024]
```

## Stable IDs

Language IDs:

```text
ZH = 0
JP = 1
EN = 2
```

Tone offsets:

```text
ZH = 0
JP = 6
EN = 8
```

Tone counts:

```text
ZH = 6
JP = 2
EN = 4
total = 12
```

These values are model-contract constants and must not be configurable per
request.

## Blank Insertion

When `add_blank` is enabled, C++ must mirror Python:

```cpp
phone_ids = intersperse(phone_ids, 0);
tone_ids = intersperse(tone_ids, 0);
language_ids = intersperse(language_ids, 0);

for (auto & n : word2ph) n *= 2;
word2ph.front() += 1;
```

BERT must be generated or repeated using the adjusted `word2ph`, so each BERT
channel length equals the blank-inserted phone length.

## BERT Routing

The active language BERT is generated from that language's tokenizer and ONNX
model. Non-active language channels follow the Python runtime behavior and use
deterministic random features by default.

```text
ZH -> bert_zh active, bert_jp random, bert_en random
JP -> bert_zh random, bert_jp active, bert_en random
EN -> bert_zh random, bert_jp random, bert_en active
```

Zero-filled inactive channels can be kept as an explicit debug option, but they
are not the default Python-compatible behavior.

## Language Frontends

### Chinese

Python reference:

```text
text/chinese.py
text/tone_sandhi.py
text/chinese_bert.py
```

C++ must replicate:

- number normalization compatible with `cn2an.an2cn` for supported inputs
- punctuation replacement and non-Chinese filtering
- segmentation behavior needed by tone sandhi
- pinyin lookup equivalent to `pypinyin`
- tone sandhi rules
- `opencpop-strict.txt` pinyin-to-phone mapping
- Chinese BERT WordPiece token IDs and `word2ph` repetition

Short-term compatibility target:

```text
norm_text, phones, tones, word2ph match Python for common Mandarin text.
```

### Japanese

Python reference:

```text
text/japanese.py
text/japanese_bert.py
```

C++ must replicate:

- NFKC normalization
- Japanese number expansion
- punctuation replacement and filtering
- OpenJTalk frontend surface/pron extraction
- katakana to phoneme conversion
- long-vowel handling
- OpenJTalk label accent extraction
- tone alignment
- WordPiece tokenization for `deberta-v2-large-japanese-char-wwm`
- `word2ph` distribution across BERT tokens

OpenJTalk C/C++ is allowed because it is not Python.

### English

Python reference:

```text
text/english.py
text/english_bert_mock.py
```

C++ must replicate:

- number, ordinal, decimal, currency normalization
- abbreviation expansion
- punctuation replacement and spacing
- DeBERTa SentencePiece tokenization
- CMUdict lookup
- OOV grapheme-to-phoneme fallback compatible with `g2p_en` as far as practical
- stress-to-tone mapping
- `word2ph` distribution across SentencePiece tokens

OOV fallback is the highest-risk English mismatch.

## Validation Levels

Use Python-generated fixtures only for validation, not runtime.

```text
Level 1: C++ produces valid shapes and can synthesize.
Level 2: phones, tones, word2ph match Python for fixture texts.
Level 3: tokenizer IDs and BERT repeat lengths match Python.
Level 4: ONNX audio is close to Python inference with the same exported model.
```

## Implementation Priority

1. Preserve and enforce the common frontend invariants.
2. Make Japanese the first strict-match language, because OpenJTalk is already
   available in C++.
3. Bring English normalization and token grouping closer to Python; then improve
   OOV fallback.
4. Replace the simplified Chinese frontend with table-driven pinyin, segmentation,
   and tone sandhi.
5. Add fixture-based C++ tests for `norm_text`, `phones`, `tones`, `word2ph`, and
   tokenizer IDs.

