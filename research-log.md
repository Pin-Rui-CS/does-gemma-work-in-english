# Research log — from replication to experiment

Working document. Records what has been established, what is still conjecture, and the reasoning that connects them. Written to be read in order; the point is the chain of inference, not the summary.

Notation used throughout:

- **✓** measured, in this repo, reproducible
- **?** conjecture — plausible, not yet tested
- **⚠** claimed by someone else and not yet independently verified

---

## 1. The question we started with

> Does gemma-2-2b route through English when translating Japanese → German?

This is Wendler et al. (2024)'s question, ported to a model, language, and parameter scale they didn't cover. Their method: run a few-shot translation prompt, apply the logit lens to the residual stream at every layer, and track the probability of the correct word in each of three languages across depth. If English rises in the middle layers and the target language only at the end, the model "detoured" through English.

The design includes a second condition, JA→JA repetition, where the model copies the input word back. English appears nowhere in the prompt or the answer. The original intent in this project was to treat repetition as a **null control** — a task where no English should appear, against which translation could be compared.

**That intent was wrong, and finding out why is where the project actually started.**

---

## 2. Phase 1 — the replication

### 2.1 What was built

A logit lens over gemma-2-2b (26 layers, d=2304), reading 27 positions per prompt (embedding + 26 layers). 54 Japanese/English/German triples, concrete single-kanji nouns, 4-shot prompts, 3 demonstration seeds per word, two conditions. 324 forward passes total.

Three things about gemma-2 specifically had to be right, and each fails silently rather than loudly:

- **bfloat16 is mandatory.** fp16 overflows Gemma-2's activations and yields NaNs with no error.
- **Logit soft-capping (30.0) must be reapplied by hand.** The model applies it internally; a hand-rolled lens bypasses that path. Skip it and every probability is inflated.
- **`W_U` cannot be upcast to fp32.** It is 2304 × 256,000 — a float32 copy is ~2.4 GB and OOMs a T4. The residual is cast *down* to `W_U`'s dtype instead.

### 2.2 What was verified before any result was trusted

✓ **Lens verification.** The hand-rolled lens reproduces the model's own final-layer distribution to ~0.6% relative (0.6715 vs 0.6755 on the first triple). Max |Δlogit| = 0.121, mean = 7.3e-4, identical top-5 token orderings. Mean deviation is 0.4% of max — centred noise, consistent with bf16's ~8 mantissa bits. A uniform logit shift would cancel in the softmax anyway; per-token scatter is the only thing that could produce the observed gap, and that is what we see.

✓ **Behavioural gates.** 12/12 on translation, 12/12 on repetition. If the model can't do the task, nothing measured inside it means anything.

**What the lens check does *not* prove**, and this turned out to matter enormously: it validates the *plumbing* — norm, unembedding, soft cap — but not the *token ids*. Both sides of the assertion index the same ids. A wrong id passes silently.

### 2.3 The bug that a passing assertion hid

The original scoring took **one** token id per language, and for Japanese it used the *unspaced* form. But the repetition prompt ends `日本語:` — with a colon and no trailing space — so the model emits a **space-prefixed** Japanese token. We were scoring a token the model would never produce.

The symptom: P(Japanese) read near zero at every layer **including the final one**, where Japanese is the correct answer and the behavioural check says the model gets it right 12/12. German and English were scored with the spaced form and behaved normally. That asymmetry is the whole proof — a bug that hits exactly one language, in exactly the condition where that language is the answer.

The fix is Wendler's Appendix A.2, which we had read but not implemented properly. P(language) is **not** the probability of one token. It is a sum over `Start(w)` — every vocabulary token that could *begin* the correct word: all token-level prefixes, with and without a leading space, plus byte-fallback tokens.

This is not cosmetic. The final-layer top-5 for 花 is `[' Blume', ' Blumen', ' Bl', ' Pflanze', ' Blüten']`. `' Bl'` carries real mass, and first-token-only scoring discards it.

### 2.3.1 The same bug, missed a second time (25 Jul)

⚠ **The fix above describes two corrections. Only the first was ever applied.**

The space fix landed. The prefix summation did not — `start_token_ids` in the notebook returned only the first token of the spaced and unspaced tokenizations, two ids for 花, with no vocabulary scan. The methods notes specified it, the project notes recorded it as done, and the n=54 numbers in §2.4 were produced without it.

It was caught by reading the notebook source against the methods notes, not by any test — the same way the original bug was caught, and for the same reason: **every gate in the pipeline was blind to it.** The lens assertion indexes the same ids on both sides. The behavioural check reads top-k strings, not id sets. The filter compared sets that happened to be singletons. Nothing in the notebook could tell a narrow id set from a correct one.

Remediation: the filter cell now **asserts** that id sets for the European words contain more than one token, so this specific omission halts the run instead of producing quietly wrong curves.

> **Generalised lesson:** the first bug taught us that a passing assertion doesn't mean correct scoring. The second one teaches something narrower and more useful — *a documented fix is not an applied fix.* The ground-truth document and the executed code drifted apart, and nothing was watching the gap. Verification has to run against the artifact that executes, not the one that describes it.

> **Methodological lesson, and the honest headline of Phase 1:** a detail buried in an appendix determined whether a published result replicated. The assertion that was supposed to catch errors passed the whole time, because it was blind to the class of error we had.

### 2.4 The result

⚠ **Superseded — produced with first-token-only scoring (see §2.3.1). Pending re-run.**
Retained because the before/after across the scoring fix is itself evidence. The
direction of change is predictable — P(EN) and P(DE) should both rise, since prefix
tokens like `' Bl'` are currently uncounted — but the effect on the *difference*
between conditions is not, which is why it has to be measured rather than argued.

| condition | peak P(EN) | ±95% CI | position |
|---|---|---|---|
| JA→DE translation | 0.6141 | 0.0732 | `23_pre` |
| JA→JA repetition | 0.6626 | 0.0648 | `24_pre` |

Δ (translation − repetition) = **−0.0485**, 95% bootstrap CI over words **[−0.1010, +0.0034]**, 2000 resamples.

For comparison, before the word list was expanded (n=24): Δ = −0.0090, CI [−0.0733, +0.0605]. Tripling the words moved the point estimate away from zero and halved the interval width — but the interval still contains zero, barely.

✓ Japanese single-token fraction: **53/54 = 98%**. English 100%, German 98%.

---

## 3. Reading that result correctly

The naive reading is "the control didn't separate, so the experiment failed." That reading is wrong, and the literature says so explicitly.

Wendler §6:

> The English-first pattern is **less pronounced** on repetition, where the input language rises earlier or (for Chinese) even simultaneously with, or faster than, English.

The repetition task **is not a null control.** It is a weaker-effect condition. Wendler's own explanation is that recognising "this task is a copy" still requires semantic understanding, which happens in concept space, which is English-biased. So English shows up in repetition too.

And the *degree* to which it shows up varies by language, in a way they tie to tokenization (§4 below). Chinese — 100% single-token — is their one exception where the source language rises *simultaneously with or faster than* English.

Japanese here is 98% single-token. Essentially the Chinese profile.

> **So a weak or absent separation is what Wendler's account predicts for this language.** The result is a replication, not a null. It is not evidence against the English-pivot picture.

### 3.1 One thing the summary statistic throws away

The two English curves have different **shapes**, which `max` collapses to nothing:

| residual position | 20 | 21 | 22 | 23 | 24 |
|---|---|---|---|---|---|
| translation P(EN) | ~0.41 | ~0.52 | ~0.58 | **0.61** | ~0.47 |
| repetition P(EN) | ~0.05 | ~0.17 | ~0.24 | ~0.38 | **0.66** |

⚠ Eyeballed off the rendered figures — the `.npy` arrays were lost when the Kaggle container was destroyed, so these are not yet exact.

Translation shows a **broad English plateau from ~layer 20**. Repetition shows a **narrow late spike at 24**, arriving just before Japanese resolves at 25–26. At position 20 that's ~0.41 vs ~0.05, which is not a subtle difference — but the peak heights are nearly equal, so the peak statistic reports "no difference."

? A shape-sensitive statistic — mean P(EN) over layers 19–22, or area under the curve — would plausibly separate these cleanly. This is a **post-hoc** choice suggested by looking at the data, and if used it must be reported as exploratory, not confirmatory.

---

## 4. Where this stops being enough

Everything above is a **measurement**. A careful one, with verified machinery and an honest error history — but it answers "does the known effect appear here too?" and the answer is "yes, about as expected."

State of knowledge after Phase 1:

| | |
|---|---|
| ~ | The English detour appears in gemma-2-2b, JA→DE, at 2B scale — *provisional, §2.3.1* |
| ~ | It also appears in JA→JA repetition, at comparable peak magnitude — *provisional* |
| ✓ | Japanese is 98% single-token in Gemma's vocabulary (independent of the scoring bug) |
| ✓ | Scoring the wrong token space silently destroys one language's curve — observed twice |
| ✗ | Whether full `Start(w)` prefix summation changes the between-condition difference |
| ✗ | **Why** concept space is English-skewed |
| ✗ | Whether the detour is functional or a passenger |
| ✗ | Whether tokenization causes the cross-language variation |

The qualitative shape of the curves is unlikely to move — the space fix was the one that
determined whether a language registered at all, and prefix summation adds mass to
already-present curves rather than creating them. But "unlikely" is a prediction, and §2.4
is marked provisional until it is checked.

The last row is the one with a testable, tractable, six-day-sized experiment behind it.

---

## 5. The concept that reframed the project: single-token availability

The fraction of a word list where the correct answer exists as **one entry in the model's vocabulary**.

Why it is load-bearing for this method: the logit lens reads the distribution at *one* position. If the answer is one token, that token *is* the answer and P(language) is read directly. If the answer takes three tokens, the model first emits a fragment — and a fragment usually doesn't identify a language. `Start(w)` prefix summation exists precisely as a workaround for this.

Measured fractions:

| | single-token | detour on repetition |
|---|---|---|
| this project, Japanese | 98% | weak (Δ CI contains 0) |
| Wendler, zh | 100% | weak — their stated exception |
| Wendler, fr | 55% | strong |
| Wendler, de | 43% | strong |
| Wendler, ru | 13% | strong |
| Wendler, et | **1%** | **weak** ← see §7 |

### 5.1 Wendler's hypothesis

§6, stated as a conjecture, never tested:

> This may be due to tokenization... **Where language-specific tokens are available, the detour through English seems less pronounced.**

The mechanism, reconstructed: if the model can name the concept in one target-language token, it can commit to that language immediately and leave concept space. If the word takes three tokens, the first is a non-committal fragment, so the latent lingers in the English-biased concept region — producing a larger apparent detour.

They then use this to make a claim with real-world stakes:

> This supports prior concerns about tokenization, which not only burdens minority languages with more tokens per word but also **forces latents through an English-biased semantic space.**

---

## 6. The confound

Their evidence is four data points across four languages, and **single-token fraction is not an independent variable.**

A language has good single-token coverage *because the tokenizer was built to cover it*, which tracks how much of that language was in the training corpus. So their correlation cannot distinguish:

- **H1 — tokenization per se.** One-token availability lets the model exit concept space early.
- **H2 — competence.** The model simply knows Chinese better than Russian, and single-token fraction is a proxy for that.

Chinese and Russian differ in script, morphology, word frequency distribution, and corpus share. Any of these could drive the difference. Wendler cannot separate them, because for them **changing token count means changing language.**

This is the gap. It is not a flaw they hid — it's a conjecture in a discussion section, correctly hedged with "may be." But nobody has tested it, and downstream work repeats it as though it were established.

---

## 7. The crack in the hypothesis — Estonian

⚠ **This claim needs verification against Figure 21 of the original paper before the project is built on it.** The source here is a markdown conversion where figures survive only as prose summaries, and the claim rests on one table cell.

Appendix B.1. Estonian is **1 of 99 words single-token — 1%**, the worst tokenization in the paper. H1 predicts the *strongest* English detour of any language studied. What they report for the copy task:

> **Copy** — Behaves most similarly to Chinese — Estonian probability exceeds English already in intermediate layers.

Chinese at 100% single-token and Estonian at 1% — opposite extremes of the proposed causal variable — produce **the same behaviour**, and it's the weak-detour behaviour.

The explanation in the discussion section does not cover the result in the appendix.

### 7.1 The competing explanation, which must be addressed

There is a plausible alternative that rescues neither hypothesis cleanly. Estonian cloze scores **0%** success — the model may be so weak in Estonian that copying degenerates into mechanical token-matching with no semantic processing at all. Weak detour, but for an entirely different reason: the latent never enters concept space, rather than exiting it quickly.

If that's right, Estonian isn't a counterexample to H1 — it's off-distribution for the whole framework. Distinguishing "exits concept space fast" from "never enters it" is checkable: the two predict different **entropy** and **token energy** trajectories. Phase 1 never measured either.

---

## 8. The experiment

**Japanese dissociates tokenization from language.** The same word, same meaning, same language, same model, written three ways:

| written | expected tokens | word |
|---|---|---|
| 猫 | 1 | cat |
| ねこ | 2 (ね + こ) | cat |
| ネコ | 2+ | cat |

⚠ Those counts are **predictions**. Gemma's vocabulary has 256,000 entries and may well contain ねこ as a single token. See §10.

Design: **JA→JA repetition**, varying only the orthography of the target. Repetition is the correct venue because that is where Wendler's tokenization claim lives — it is the task where their Chinese exception appears.

Why this is a better design than the cross-language comparison:

- **Language, model, task, and training exposure are held fixed.** Only token count moves.
- **Fully paired.** Same concept in every condition, so the comparison is within-word. Far more statistical power than the current unpaired peak comparison — which is exactly what left Phase 1 with a CI grazing zero.
- **All 54 concepts already have kana spellings.** The list is reused, not rebuilt.

### 8.1 Predictions

| | H1 (tokenization) | H2 (competence) |
|---|---|---|
| kana vs kanji detour | kana **stronger** | **no difference** — same language, same competence |

The hypotheses make opposite predictions on a within-language manipulation. That is the point of the design.

### 8.2 What each outcome buys

| outcome | interpretation |
|---|---|
| kana detour > kanji | H1 supported. The Anglocentric routing is partly an **engineering artifact** of tokenizer design — addressable with better tokenizers, not only with balanced training data. |
| no difference | H1 not supported, consistent with the Estonian anomaly. The cross-language variation must come from competence, morphology, or script. A widely-repeated explanation doesn't hold. |

Both outcomes are informative. This is the key structural difference from Phase 1: there, a null meant "underpowered." Here, a null is a substantive result about a published claim.

### 8.3 Known limitation, stated up front

The dissociation is **improved, not clean.** ねこ and 猫 are the same word in the same language, but kanji spellings are more frequent in training text for most concrete nouns. We have traded a large cross-language confound for a smaller within-language **orthographic frequency** confound.

Partial mitigations: prefer concepts where kana spelling is genuinely common in natural text (some animal names, onomatopoeia); report the direction of the frequency asymmetry; treat it as a limitation rather than claiming tokenization has been isolated.

### 8.4 A scoring trap specific to this design

`Start(w)` for ねこ includes `ね`, and single-kana tokens are extremely common in Japanese text for reasons unrelated to the target word. P(Japanese) will be **inflated** for kana targets by construction — which biases *toward* finding a difference, in the direction of H1.

This must be handled or the result is worthless: report a strict variant alongside the standard `Start(w)`, and check whether the effect survives it.

---

## 9. Second experiment — token energy

Phase 1 replicated only Wendler §4.1 (probabilistic). §4.2 (geometric) is the stronger half of the paper and is cheap to add:

$$E(h)^2 = \frac{v}{d}\cdot\frac{\lVert \hat{U}h \rVert^2}{\lVert \hat{U}^\top\hat{U} \rVert_F^2}$$

where $\hat{U}$ is the unembedding matrix with 2-normalised rows. Use $\hat{U}^\top\hat{U}$ (2304 × 2304), **not** $\hat{U}\hat{U}^\top$ (256k × 256k, will not materialise) — their footnote 2.

Two reasons it earns its place:

1. It tests whether a **2B** model shows the same three-phase structure Wendler found at 70B. Gemma-2 is RMSNorm, so the hypersphere geometry carries over.
2. It is the measurement that distinguishes "exits concept space fast" from "never enters concept space" — the §7.1 ambiguity. Entropy and token energy separate those; language probability alone does not.

---

## 10. Kill criteria — check before spending a GPU session

The centerpiece rests on an unverified tokenization assumption. **Check it first. It needs only the tokenizer — no GPU, no model weights.**

```
for each concept: token counts of 漢字 / ひらがな / カタカナ spellings
```

- If kana spellings are mostly **multi-token** → design holds, proceed.
- If Gemma stores common kana words as **single tokens** → the manipulation collapses and the experiment must be redesigned (longer words, or compounds) *before* any GPU time is spent.

The second failure mode is behavioural: the model must actually copy a kana target when shown kana demonstrations. If accuracy collapses on kana, the condition is measuring task failure, not tokenization. Run the behavioural gate per orthography and report it.

---

## 11. Alternatives considered

| option | why not now |
|---|---|
| **Causal ablation of the English phase** — project the English direction out mid-stack, measure whether German accuracy drops | Genuinely deeper: tests whether the detour is *functional* or a passenger, which the logit lens can never answer. Kept as the next priority if time allows. Needs matched controls (random direction, German direction) or it proves nothing, since projecting out anything degrades a model. Bigger claim, harder to make airtight in the time available. |
| **Grammatical vs lexical targets** (particles は/が/を) — the sharpest open seam in the literature (Schut vs Brinkmann) | Breaks the methodology. Particles have no English analog, so there is no P(EN) to measure without building vocabulary-wide language classification — a substantial methodological lift with its own ambiguity problems, which is exactly what Wendler designed their word lists to avoid. Belongs in future work. |
| **Scale comparison** (gemma-2-9b) — tests Schut's claim that smaller models share more representations | 9B in bf16 is ~18 GB; a T4 has 16. Would need 8-bit quantization plus TransformerLens compatibility work. Poor time-to-payoff before the deadline. |
| **Instruction-tuned comparison** (`gemma-2-2b-it`) | Cheap and interesting — same size, same tokenizer, differs only in post-training. Genuine stretch goal, not core. |
| **Tuned lens** | Actively wrong here. Wendler §2: the tuned lens is trained to map intermediate states to the *final* prediction, which is in the target language — training it would optimize away the signal being measured. Declining it deliberately, and saying why, is worth more than running it. |

---

## 12. Open risks

1. **The Estonian claim is unverified** (§7). If Figure 21 doesn't support it, the motivation weakens to "an untested conjecture" — still adequate, but less compelling.
2. **The kana tokenization assumption is unverified** (§10). This can invalidate the design outright.
3. **Orthographic frequency confound** (§8.3) is real and only partly mitigable.
4. **`Start(w)` inflation for kana** (§8.4) biases toward the hypothesis. Must be controlled explicitly.
5. **Raw `.npy` arrays from the n=54 run were lost** when the Kaggle container was destroyed. Figures and summary numbers survive in the notebook; exact curve values need one re-run.
6. **Compute:** Kaggle free tier, T4, 12-hour session cap, ephemeral storage. Every session must push results before it ends.

---

## 13. Current state

- Phase 1 complete and reproducible; interpretation settled (§3).
- Phase 2 designed, **gated on the §10 tokenizer check.**
- Token energy (§9) is unblocked and can be implemented in parallel.
