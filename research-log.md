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

**Resolved 25 Jul.** Prefix summation applied and the full sweep re-run. The id sets widened as expected — `Blume` 2 → 8, `flower` 2 → 10, 花 2 → 2 (a single kanji has no shorter string prefix) — and the filter still passes 54/54, so German capitalisation does keep the DE and EN sets disjoint even under broad prefix matching.

✓ **The correction barely moved the numbers**: peak P(EN) 0.6141 → 0.6170 (translation), 0.6626 → 0.6627 (repetition).

That non-result is worth keeping. With a word list that is 98–100% single-token, the canonical token already holds nearly all the mass and the fragmentary prefixes hold almost none, so there is very little for prefix summation to recover. **The size of this correction is itself a function of single-token availability** — it is large for Wendler's Russian (13%) and negligible here. Which means the measurement is robust to this scoring choice *for high-single-token languages specifically*, and will not be for the kana word lists in §8. Worth re-checking there rather than assuming it stays negligible.

> **Generalised lesson:** the first bug taught us that a passing assertion doesn't mean correct scoring. The second one teaches something narrower and more useful — *a documented fix is not an applied fix.* The ground-truth document and the executed code drifted apart, and nothing was watching the gap. Verification has to run against the artifact that executes, not the one that describes it.

> **Methodological lesson, and the honest headline of Phase 1:** a detail buried in an appendix determined whether a published result replicated. The assertion that was supposed to catch errors passed the whole time, because it was blind to the class of error we had.

### 2.4 The result

⚠ **Superseded 25 Jul — logit-lens correction, see §2.5. Awaiting re-run.**

Final, with `Start(w)` prefix summation, n=54 words × 3 seeds:

| condition | peak P(EN) | ±95% CI | position |
|---|---|---|---|
| JA→DE translation | 0.6170 | 0.0728 | `23_pre` |
| JA→JA repetition | 0.6627 | 0.0648 | `24_pre` |

Δ peak (translation − repetition) = **−0.0457**, 95% bootstrap CI over words **[−0.0975, +0.0041]**.

History of that interval: n=24 gave Δ = −0.0090, CI [−0.0733, +0.0605]; n=54 with first-token-only scoring gave −0.0485, CI [−0.1010, +0.0034]; n=54 corrected gives the row above. Tripling the word count halved the interval and moved the point estimate away from zero. It still contains zero, barely — and §3.1 explains why chasing it further would have been the wrong move.

✓ Japanese single-token fraction: **53/54 = 98%**. English 100%, German 98%.

---

## 2.5 The lens itself was wrong (25 Jul)

Found while extending the method to a second model family, and it invalidates every
intermediate-layer number produced so far.

`lens_probs` called `cache.apply_ln_to_stack(resid, layer=-1)`. That TransformerLens
helper normalises **every** component by the *final* layer's cached scale, so that
per-component contributions sum linearly to the real logits. That is the right tool
for **logit attribution** — decomposing a final logit into per-head contributions. It
is not the logit lens.

Wendler §3.2 define the lens as treating a latent "as if it were a final-layer
latent", which means normalising each latent **by its own scale** and then
unembedding. §3.1 gives the invariant that makes this checkable: after RMS
normalisation every latent lies on a hypersphere of radius √d.

The two conventions agree exactly at the **last** row, where the cached scale *is*
that latent's scale. They diverge everywhere else, increasingly with depth, because
residual norms grow across layers — so intermediate latents were being divided by a
scale much larger than their own and systematically suppressed.

### 2.5.1 How it surfaced, and why nothing caught it

Gemma tolerated it silently. Llama-3.2-1B did not: `tgt_final` read 0.0000 where
`model()` on the identical prompt and ids gave 1.0. The id sets were provably fine —
`' Bl'` was in the German set and the model put probability 1.0 on it.

Three gates existed and all three were blind to it:

| gate | why it missed |
|---|---|
| lens-vs-model assertion | only checks the **final row**, the one row where the two conventions agree |
| behavioural check | calls `model()` and reads decoded strings — never touches the lens |
| `Start(w)` disjointness | operates on id sets, not on probabilities |

> **This is the third instance of one failure mode.** First: an assertion that indexed
> the same ids on both sides could not detect wrong ids. Second: a documented fix that
> was never applied, with nothing comparing the notes to the code. Third: a
> verification that covered one row of twenty-seven. Every time, the gate did not cover
> the code path that produced the numbers. That generalisation is worth more than any
> individual fix, and it belongs in the write-up.

### 2.5.2 The fix

Use the model's own modules — `model.ln_final(h)` then `model.unembed(h)` — so
architecture differences are TransformerLens's responsibility rather than ours, and
assert Wendler's √d invariant on every latent, not just the last:

```python
radius = h[0].norm(dim=-1) / model.cfg.d_model ** 0.5
assert (radius - 1).abs().max() < 0.05
```

That check is architecture-independent and covers all 27 rows. Added alongside it: a
per-model `verify_lens()` gate in the multi-model notebook, run *before* any sweep, so
a model whose lens does not reproduce its own output is refused rather than measured.

? **Expected direction of change.** Intermediate latents were suppressed, so corrected
mid-stack probabilities should be **higher**, and the English rise possibly earlier.
Whether the translation-vs-repetition *difference* in §3.1 survives is not predictable
— both conditions were distorted, but not necessarily equally. It has to be re-measured.

**Unaffected:** behavioural accuracy, single-token fractions, `Start(w)` id sets, all
final-layer probabilities, and the entire scoring audit in §2.3.

---

## 3. Reading that result correctly

The naive reading is "the control didn't separate, so the experiment failed." That reading is wrong, and the literature says so explicitly.

Wendler §6:

> The English-first pattern is **less pronounced** on repetition, where the input language rises earlier or (for Chinese) even simultaneously with, or faster than, English.

The repetition task **is not a null control.** It is a weaker-effect condition. Wendler's own explanation is that recognising "this task is a copy" still requires semantic understanding, which happens in concept space, which is English-biased. So English shows up in repetition too.

And the *degree* to which it shows up varies by language, in a way they tie to tokenization (§4 below). Chinese — 100% single-token — is their one exception where the source language rises *simultaneously with or faster than* English.

Japanese here is 98% single-token. Essentially the Chinese profile.

> **So a weak or absent separation is what Wendler's account predicts for this language.** The result is a replication, not a null. It is not evidence against the English-pivot picture.

### 3.1 The summary statistic was hiding the result

⚠ **Every number in this section is an intermediate-row measurement and is superseded
by §2.5. Awaiting re-run.** The qualitative claim — that the conditions separate on
timing rather than height — may well survive, since both conditions were distorted by
the same mechanism, but the magnitudes will move and the claim has to be re-checked
rather than assumed.

The peak comparison above is the pre-specified test and it does not separate. But `max` collapses a 27-point trajectory to one scalar, and the trajectories are not remotely alike:

| residual position | 19 | 20 | 21 | 22 | 23 | 24 |
|---|---|---|---|---|---|---|
| translation P(EN) | 0.054 | 0.417 | 0.530 | 0.583 | **0.617** | 0.483 |
| repetition P(EN) | 0.007 | 0.053 | 0.175 | 0.242 | 0.384 | **0.663** |
| Δ, paired bootstrap | +0.047 | **+0.364** | +0.355 | +0.342 | +0.233 | −0.180 |
| CI excludes 0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

At position 20: translation 0.417 vs repetition 0.053, Δ = +0.364, CI [+0.307, +0.422]. Area under the English curve across the whole pass: Δ = **+1.200, CI [+0.998, +1.406]**.

✓ **The conditions separate decisively. The peak statistic is blind to how.** Under translation English occupies a broad plateau from position ~20 onward; under repetition it appears only as a brief transient at 24, immediately before Japanese resolves at 25–26.

This is what Wendler's §6 "less pronounced on repetition" looks like when measured properly: **pronouncedness is duration, not height.** Both tasks reach the same peak English probability. Only translation dwells there.

⚠ **Status: exploratory, and it must be labelled that way.** The shape difference was noticed by eye first and the statistic chosen afterwards; 27 positions are tested with no multiple-comparison correction. Two things argue it is not a selection artifact — the effect is ~6× the CI half-width, and the area statistic has no free parameter (no window was chosen). But the honest description is "hypothesis for a pre-registered test," not "established." A cautionary detail from the same table: positions 17–18 also have CIs excluding zero, at Δ ≈ 0.0005. Resolvable and meaningless.

### 3.2 A result that does not fit Wendler's explanation

Their §6 attributes the weak Chinese repetition effect to tokenization — Chinese being 100% single-token, the model can commit to the target language immediately and skip the English detour. For Chinese specifically they report the source language rising *simultaneously with or faster than* English.

Japanese here is **98% single-token** — effectively the same profile. It should behave like Chinese. It does not: in repetition, English still clearly precedes Japanese, peaking at position 24 while Japanese only resolves at 25–26.

? So a language with Chinese-like tokenization does *not* show the Chinese pattern. Single-token availability alone does not appear to determine whether the English detour is skipped. That is a second crack in the same explanation, independent of the Estonian one in §7 — and this one is in our own data rather than an appendix we cannot verify.

Confounds to hold in mind before leaning on this: different model family (Gemma vs Llama-2), different scale (2B vs 7B+), and Japanese is not Chinese in any respect other than kanji overlap. It weakens the tokenization account; it does not refute it. What it does do is raise the value of §8, which tests the account directly rather than by cross-language analogy.

---

## 4. Where this stops being enough

Everything above is a **measurement**. A careful one, with verified machinery and an honest error history — but it answers "does the known effect appear here too?" and the answer is "yes, about as expected."

State of knowledge after Phase 1:

| | |
|---|---|
| ✓ | The English detour appears in gemma-2-2b, JA→DE, at 2B scale |
| ✓ | It also appears in JA→JA repetition, at the same peak height but far shorter duration |
| ✓ | Japanese is 98% single-token in Gemma's vocabulary |
| ✓ | Prefix summation is near-negligible at 98–100% single-token; its size scales with tokenization |
| ✓ | Scoring the wrong token space silently destroys one language's curve — observed twice |
| ~ | Translation and repetition separate on *timing* — large effect, exploratory status (§3.1) |
| ~ | Japanese does not reproduce the Chinese repetition pattern despite matching its tokenization (§3.2) |
| ✗ | **Why** concept space is English-skewed |
| ✗ | Whether the detour is functional or a passenger |
| ✗ | Whether tokenization causes the variation — **§8 tests this** |

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

- **Blocked on a re-run of both notebooks (§2.5).** The logit lens used the
  attribution normalisation rather than the lens normalisation, so every
  intermediate-row number in this document is superseded. Both notebooks are fixed
  and carry the √d invariant plus a per-model verification gate; neither has been
  re-executed. Nothing in §3 or §4 should be quoted until it has.
- The cross-model sweep additionally needs its Llama cells discarded outright —
  those were not merely distorted, they were noise (§2.5.1).
- Phase 1 complete and reproducible; interpretation settled (§3).
- **Phase 2 reordered, 25 Jul.** §3.2 claims Japanese fails to reproduce Wendler's
  Chinese pattern, but compares Japanese-on-Gemma against Chinese-on-Llama-2 — a
  language claim resting on a cross-model comparison. That has to be measured
  before any explanation of it is tested. `tokenization_vs_detour.ipynb` does so:
  five languages × multiple model families, testing whether single-token fraction
  predicts the English detour at all.
- That notebook also supplies a cleaner dissociation than the kana design in §8.
  Vocabulary size differs by family (Llama-2 32k, Llama-3 128k, Gemma-2 256k), so
  **the same word list has a different single-token fraction on each model** —
  language, words, prompt and scoring all held fixed while the independent variable
  moves. And for the 17 concepts written identically in Japanese and Chinese, the
  target token is *the same token*, holding tokenization exactly constant while the
  language framing varies. Together those two comparisons cross the design.
- §8 (kana) is now partly redundant and drops below the above in priority.
- Token energy (§9) is unblocked and can be implemented in parallel.
