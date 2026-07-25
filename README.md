# Does Gemma Work in English?

A logit-lens replication testing whether **gemma-2-2b** routes through English when translating Japanese → German, measured against a Japanese → Japanese repetition control.

## Background and attribution

This replicates the probabilistic half of:

> Wendler, C., Veselovsky, V., Monea, G., & West, R. (2024). *Do Llamas Work in English? On the Latent Language of Multilingual Transformers.* ACL 2024, 15366–15394. [arXiv:2402.10588](https://arxiv.org/abs/2402.10588) · [epfl-dlab/llm-latent-language](https://github.com/epfl-dlab/llm-latent-language)

**Theirs:** the experimental design (few-shot translation and repetition tasks), the logit-lens reading of intermediate residual streams, the word-selection criteria, and the `Start(w)` prefix-summation scoring method (their Appendix A.2).

**Mine:** the port to gemma-2-2b (they use Llama-2 7B/13B/70B and Mistral-7B), the Japanese source language (they cover Chinese, German, French, Russian, Estonian), the German target pairing, and the scoring audit described below.

Wendler's conclusion is narrower than their title: the internal lingua franca is *not English but concepts — concepts biased toward English*. This replication inherits that framing.

## Result

All numbers below are at n=54 words × 3 demonstration seeds, with `Start(w)` prefix-summation scoring.

**The two conditions reach the same peak, at different times.** Measured by peak height alone they are indistinguishable:

| condition | peak P(EN) | ±95% CI | position |
|---|---|---|---|
| JA→DE translation | 0.6170 | 0.0728 | `23_pre` |
| JA→JA repetition | 0.6627 | 0.0648 | `24_pre` |

Δ peak (translation − repetition) = −0.0457, 95% bootstrap CI **[−0.0975, +0.0041]** — straddles zero.

But peak height discards the trajectory, and the trajectories are not close:

<p align="center">
  <img src="results/fig_timing.png" width="88%" alt="P(English) by condition and their paired difference">
</p>

| residual position | 19 | 20 | 21 | 22 | 23 | 24 |
|---|---|---|---|---|---|---|
| translation P(EN) | 0.054 | 0.417 | 0.530 | 0.583 | **0.617** | 0.483 |
| repetition P(EN) | 0.007 | 0.053 | 0.175 | 0.242 | 0.384 | **0.663** |
| Δ, paired | +0.047 | **+0.364** | +0.355 | +0.342 | +0.233 | −0.180 |
| 95% CI excludes 0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

At position 20 translation is at 0.417 while repetition is at 0.053 — an eightfold gap, CI [+0.307, +0.422]. Summed across the whole forward pass, Δ area under the English curve = **+1.200, CI [+0.998, +1.406]**.

So the conditions separate decisively; **the peak statistic is simply blind to how they differ.** Under translation English occupies a broad plateau from position ~20; under repetition it appears only as a brief transient at 24, immediately before Japanese resolves.

⚠️ The position-wise and area statistics are **exploratory** — they were chosen after seeing that the curves differed in shape, and 27 positions are tested without correction. The pre-specified peak comparison is the confirmatory one, and it does not separate. Treat the timing result as a hypothesis for a pre-registered replication, not as a confirmed effect. (Note also that positions 17–18 have CIs excluding zero at Δ ≈ 0.0005 — statistically resolvable, substantively nothing.)

**Read against Wendler, this is a replication with a sharper instrument.** They report (§6) that the English-first pattern persists in repetition but is *less pronounced*. Measured as peak height, "less pronounced" is invisible here. Measured as duration, it is unmistakable.

One result does **not** fit their account. Wendler attribute the weak Chinese repetition effect to tokenization, Chinese being 100% single-token. Japanese here is 98% single-token — essentially the same profile — yet English still clearly *precedes* Japanese in repetition (English peaks at 24, Japanese resolves at 25–26). Japanese does not reproduce the Chinese pattern despite matching its tokenization profile, which is evidence that single-token availability is not by itself what drives the effect. See [research-log.md](research-log.md) §7.

### Scoring audit

The headline number depends on a detail buried in Wendler's appendix. P(language) is **not** the probability of one token; it is the sum over `Start(w)`, every vocabulary token that could begin the correct word — all token-level prefixes, with and without a leading space, plus byte-fallback tokens. An earlier version of this notebook scored a single first token per language and, for Japanese, scored the *unspaced* form while the prompt guarantees a space-prefixed answer. That made P(Japanese) read near zero at every layer *including the final one*, where Japanese is the correct answer. The asymmetry — German reaching 0.86 while Japanese flatlined — is what exposed it.

That fix has two parts, and **only the first was applied for most of this project's life.** The space correction landed; the prefix summation did not — `start_token_ids` returned two ids per word with no vocabulary scan, and an entire n=54 sweep was run that way. It was caught by reading the notebook source against the methods notes, not by any test, because no gate in the pipeline could see it: the lens assertion indexes the same ids on both sides, the behavioural check reads strings rather than ids, and the filter was comparing sets that happened to be singletons. The filter cell now **asserts** that id sets are wider than one token, so the same omission halts the run.

Applying it widened the id sets substantially — `Blume` 2 → 8 ids (gaining `' B'`, `' Bl'`, `' Blu'`, `' Blum'`), `flower` 2 → 10, and 花 2 → 2, since a single kanji has no shorter string prefix.

**It barely moved the results:** peak P(EN) went 0.6141 → 0.6170 for translation and 0.6626 → 0.6627 for repetition. That is worth reporting rather than hiding, and it is not a coincidence — with a word list that is 98–100% single-token, the canonical token already carries nearly all the mass and the fragmentary prefixes carry almost none. The magnitude of the prefix-summation correction is itself a function of single-token availability, which is precisely why Wendler need it for Russian (13%) and why it is nearly a no-op here.

## Reproduction

**Environment** (versions the reported results were produced on):

| | |
|---|---|
| numpy | 2.0.2 |
| transformers | 4.57.6 |
| transformer_lens | 2.18.0 |
| hardware | Tesla T4 (Kaggle), CUDA |
| dtype | bfloat16 |

See [requirements.txt](requirements.txt). Two installation constraints, both load-bearing:

1. **Install `transformer_lens` with `--no-deps`.** It pins `numpy<2` and will otherwise downgrade numpy underneath an imported binary, causing a C-ABI import error.
2. **Restart the kernel after installing, before importing anything.** Pip replacing packages on disk does not change what is already loaded in memory.

**Running:** open [mechanistic_interpretability.ipynb](mechanistic_interpretability.ipynb) and run top to bottom. It needs a Hugging Face token with access to `google/gemma-2-2b` (prompted for, or set `HF_TOKEN`). Output location defaults to `/kaggle/working` and is overridable via `OUT_DIR`. The sweep is 54 words × 3 seeds × 2 conditions = 324 forward passes; wall-clock is dominated by the ~10 GB model download.

Two sanity gates run before any result is produced, and both should be checked:

- **Behavioural** — the model must actually do the task: 12/12 on translation, 12/12 on repetition.
- **Lens verification** — the hand-rolled lens must reproduce the model's own output distribution. Observed agreement to ~0.5% relative (0.6849 vs 0.6885), max |Δlogit| 0.121 with mean 7.3e-4 and identical top-5 orderings, consistent with bf16 rounding. Note this checks the lens plumbing, *not* the token ids — both sides index the same ids, so a wrong id passes silently. That is exactly how the scoring bug survived.

Gemma-2 applies logit soft-capping (30.0) internally. A hand-rolled lens bypasses that path, so the notebook reapplies it; without this every probability is inflated. bfloat16 is required — fp16 overflows Gemma-2 activations and produces silent NaNs.

## Deviations from Wendler, disclosed

- **No quotes in prompts.** Wendler wrap words in quotes so the answer carries no leading space; these prompts end on a colon, so it does. Internally consistent because scoring covers both spaced and unspaced variants.
- **54 words** vs their 139 (zh) / 104 (de) / 56 (fr) / 115 (ru).
- **Prefix filtering by set disjointness**, not their exact procedure. German noun capitalisation separates DE from EN on case alone (`▁Silber`/`▁silver`), so collision risk here is structurally lower than their French–English pairing. All 54 triples survived the filter.
- **bf16 matmul in the lens** (~8 mantissa bits); agreement with the model is measured, not assumed — see above.
- **No tuned lens**, deliberately. Following Wendler §5: the tuned lens is trained to map intermediate states onto the *final* prediction, which is in the target language, so training it would optimize away the very English signal being measured.

## Limitations and open threads

- The timing result is **exploratory**. The confirmatory statistic was fixed in advance (peak height) and does not separate; the position-wise and area statistics were chosen after inspecting the curves, and 27 positions are tested without multiple-comparison correction. The effect sizes are large enough (Δ ≈ 0.36 against a CI half-width of ~0.06) that they are unlikely to be an artifact of selection, but the honest status is "hypothesis for a pre-registered test", not "established".
- Peak position is read at a resolution of one residual-stream position, so "translation leads by ~1–2 positions" is coarse. Nothing here identifies *which* layers or components produce the shift.
- Single model, single language pair, single prompt template. Concrete single-kanji nouns only — no grammatical items (particles, counters), which is where the current literature actually disagrees.
- The logit lens is blind to any component of a latent orthogonal to token space.

## Repository contents

```
mechanistic_interpretability.ipynb   full experiment, outputs and figures embedded
research-log.md                      reasoning from replication to next experiment
notebook-audit.md                    cell-by-cell correction log / methods reference
latent-language-lit-review.md        Wendler et al. and follow-up literature
do-llamas-work-in-english.md         the source paper (CC BY 4.0, ACL 2024)
results/curves_*.npy                 per-prompt language probabilities [162, 27, 3]
results/widx_*.npy                   word index per prompt, for resampling by word
results/fig_translation.png          JA→DE, all three languages across depth
results/fig_control.png              JA→JA, all three languages across depth
results/fig_timing.png               P(English) by condition + paired difference
requirements.txt                     pinned environment
```

The `.npy` arrays let a reviewer regenerate every figure and re-run the bootstrap without a GPU. `curves_*` is indexed `[prompt, residual position, language]` with languages ordered Japanese, English, German; `widx_*` gives the word index of each prompt, which is the unit any confidence interval must resample over — the three seeds per word are not independent.

The notebook's final cell pushes `results/` to GitHub from inside the Kaggle session. This is not a convenience: the container is destroyed when the session ends, and an earlier run's arrays were lost that way.
