# `mechanistic_interpretability.ipynb` — corrected notebook

Repo: `does-gemma-work-in-english`. Ground-truth reference.

**One numbering scheme only: cell numbers.** Each cell below is the full code to paste. Under each, a `CHANGED:` line saying what differs from your current version and why. Cells with no `CHANGED:` line are already correct — leave them.

Run order is top to bottom. Cells 1–5 are setup, 6–10 build the word list, 11–14 build and verify the lens, 15–20 produce results.

---

## Cell 1 — install

```python
%pip install -q --no-deps "transformer_lens<3"
%pip install -q "transformers>=4.44,<5" einops fancy_einsum jaxtyping beartype rich better-abc accelerate sentencepiece
```

`--no-deps` stops transformer_lens downgrading numpy and causing a C-ABI import error. `%pip` installs into the kernel's own interpreter, which `!pip` does not guarantee on a remote kernel.

**Already applied.** ✅

---

## Cell 2 — markdown: restart note

> **Run this cell once per session, then restart the kernel before running anything else.**
>
> The restart is required because the interpreter has already loaded the pre-installed versions of these packages. Pip replacing them on disk does not change what is in memory, so `import transformer_lens` would still pick up the old dependencies and fail.
>
> After restarting, skip Cell 1 and continue from Cell 3. Do run Cell 1 again at the start of any **new** session — a new Kaggle session is a new container with a fresh filesystem.

`CHANGED:` your current note says to continue "without running the above cell again". True within a session, wrong across sessions. This wording covers both.

---

## Cell 3 — version record

```python
import numpy, transformers, transformer_lens
import importlib.metadata as md

print("numpy           ", numpy.__version__)
print("transformers    ", transformers.__version__)
print("transformer_lens", md.version("transformer_lens"))
```

These three numbers are your reproducibility record — they go in `requirements.txt` and the write-up. The install completed in 5.1s, which means most packages were already satisfied by the Kaggle image, so the pip commands are *not* a reliable record of what you're running. This output is.

---

## Cell 4 — GPU check

```python
import torch

print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.randn(1000, 1000, device="cuda", dtype=torch.bfloat16)
print((x @ x).sum().item())
```

`CHANGED:` smoke test now uses `bfloat16`, matching the model dtype, instead of `float16`. Cosmetic, but a reader shouldn't see fp16 anywhere in a notebook whose whole point is that fp16 breaks Gemma-2.

---

## Cell 5 — paths and token

```python
import os, getpass
from pathlib import Path

os.environ["HF_HOME"]  = "/tmp/hf"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN") or getpass.getpass("HF token: ")

OUT = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
OUT.mkdir(parents=True, exist_ok=True)
print("results ->", OUT)
```

`CHANGED:` two things. The token now falls back to an env var, so re-runs don't re-prompt — in VS Code the `getpass` box appears at the *top* of the window and is easy to miss, making the cell look hung. And `OUT` replaces every hardcoded `/kaggle/working/` path later, so the same notebook runs against a local checkout.

---

## Cell 6 — load model

```python
from transformer_lens import HookedTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL  = "google/gemma-2-2b"

model = HookedTransformer.from_pretrained(MODEL, device=device, dtype=torch.bfloat16)
model.eval()
print(device, model.cfg.n_layers, model.cfg.d_model, model.cfg.output_logits_soft_cap)
```

Expect `cuda 26 2304 30.0`. If that last value ever prints `None`, TransformerLens has not picked up Gemma-2's logit soft-capping and every probability downstream is wrong. bfloat16 is required — fp16 overflows Gemma-2 activations and produces silent `nan`.

---

## Cell 7 — word list

```python
WORDS = [
    # (japanese, english, german)
    ("花",   "flower",   "Blume"),
    ("犬",   "dog",      "Hund"),
    ("山",   "mountain", "Berg"),
    ("水",   "water",    "Wasser"),
    ("本",   "book",     "Buch"),
    ("家",   "house",    "Haus"),
    ("月",   "moon",     "Mond"),
    ("魚",   "fish",     "Fisch"),
    ("鳥",   "bird",     "Vogel"),
    ("木",   "tree",     "Baum"),
    ("目",   "eye",      "Auge"),
    ("車",   "car",      "Auto"),
    ("王",   "king",     "König"),
    ("雪",   "snow",     "Schnee"),
    ("塩",   "salt",     "Salz"),
    ("石",   "stone",    "Stein"),
    ("血",   "blood",    "Blut"),
    ("卵",   "egg",      "Ei"),
    ("靴",   "shoe",     "Schuh"),
    ("船",   "ship",     "Schiff"),
    ("夏",   "summer",   "Sommer"),
    ("銀",   "silver",   "Silber"),
    ("道",   "road",     "Straße"),
    ("空",   "sky",      "Himmel"),
]
print(len(WORDS), "triples")
```

Unchanged. Whether 24 is enough is decided empirically in Cell 20, not now.

---

## Cell 8 — prompts

```python
import random

def make_prompt(target_ja, examples, n_shot=4, seed=0):
    """JA -> DE translation."""
    rng = random.Random(seed)
    shots = rng.sample([e for e in examples if e[0] != target_ja], n_shot)
    lines = [f"日本語: {ja} - ドイツ語: {de}" for ja, en, de in shots]
    lines.append(f"日本語: {target_ja} - ドイツ語:")
    return "\n".join(lines)

def make_repeat_prompt(target_ja, examples, n_shot=4, seed=0):
    """JA -> JA repetition control."""
    rng = random.Random(seed)
    shots = rng.sample([e for e in examples if e[0] != target_ja], n_shot)
    lines = [f"日本語: {ja} - 日本語: {ja}" for ja, en, de in shots]
    lines.append(f"日本語: {target_ja} - 日本語:")
    return "\n".join(lines)

print(make_prompt("花", WORDS))
print("---")
print(make_repeat_prompt("花", WORDS))
```

`CHANGED:` both prompt builders now live in one cell so you can see that they end identically — with a colon, no trailing space. That is why the answer token carries a leading space, which is the whole cause of the bug fixed in Cell 9.

Note this deviates from Wendler, who wraps every word in quotes so the prompt ends on an opening quote and the answer has *no* leading space. Yours is internally consistent as long as scoring matches, which is what Cell 9 enforces. Disclose the deviation in the write-up.

---

## Cell 9 — token id sets **(this is the bug fix)**

```python
# Build the decoded vocabulary once. SentencePiece marks a leading space as U+2581.
_vocab_toks = model.tokenizer.convert_ids_to_tokens(list(range(model.cfg.d_vocab)))
VOCAB_STRS  = [t.replace("\u2581", " ") if isinstance(t, str) else "" for t in _vocab_toks]

def start_token_ids(model, word):
    """Every vocab token that could begin `word` (Wendler, Appendix A.2)."""
    ids = set()

    # (a) first token of the canonical tokenization, spaced and unspaced.
    #     Catches byte-fallback tokens, which are not string prefixes.
    for s in (word, " " + word):
        toks = model.to_tokens(s, prepend_bos=False)[0]
        if len(toks):
            ids.add(toks[0].item())

    # (b) every vocab token that is a string prefix of the word.
    #     This is the part that catches ' Bl' alongside ' Blume'.
    spaced = " " + word
    for i, vs in enumerate(VOCAB_STRS):
        if vs.strip() and (word.startswith(vs) or spaced.startswith(vs)):
            ids.add(i)

    return ids

# Make the space bug visible:
print("bare  \u82b1 ->", model.to_str_tokens("\u82b1",  prepend_bos=False))
print("space \u82b1 ->", model.to_str_tokens(" \u82b1", prepend_bos=False))

# Show what prefix summation adds over first-token-only:
for w in ("Blume", "flower", "\u82b1"):
    first = {model.to_tokens(s, prepend_bos=False)[0][0].item() for s in (w, " " + w)}
    full  = start_token_ids(model, w)
    print(f"{w:8} first-token {len(first)} -> full {len(full)}; "
          f"added {[repr(VOCAB_STRS[i]) for i in sorted(full - first)][:8]}")
```

`CHANGED:` this replaces `first_token_id`.

Your old function took **one** id, and for Japanese it was called with `prepend_space=False`. But the repetition prompt ends with `日本語:`, so the model emits a **space-prefixed** Japanese token. You were scoring a token the model was never going to produce, which is why `P(Japanese)` read near zero at every layer including the last, where Japanese is the correct answer.

German and English were scored with `prepend_space=True`, which matched — that is why German reached 0.86 and only Japanese flatlined. That asymmetry is the proof.

**Second correction, 24 Jul — prefix summation.** The first version of this function took only the *first* token of two tokenizations. That is narrower than Wendler, who sums over **every vocabulary token that is a prefix of the word**: their worked example for *flower* covers `f`, `fl`, `flow`, `_f`, `_fl`, `_flo`, `_flow`, `_flower` — eight tokens, not one.

This is not hypothetical here. The Cell 13 diagnostic returned a top-5 of `[' Blume', ' Blumen', ' Bl', ' Blüten', ' Pflanze']` — `' Bl'` is carrying real mass and first-token-only scoring throws it away. P(German) was undercounted, and P(English) almost certainly is too, which moves the headline number.

Branch (a) is retained because byte-fallback tokens (e.g. `<0xE8>`) are not string prefixes and branch (b) cannot see them. Wendler handles these explicitly for Chinese and Russian; whether Japanese needs them in Gemma's 256k vocab is an empirical question the print statement answers.

⚠️ Broad prefix sets make the Cell 11 disjointness check load-bearing rather than a formality: `' Bl'` and `' bl'` differ only by case. German noun capitalisation is what keeps DE and EN apart — verify it rather than assuming it.

If the two `to_str_tokens` lines above print different first tokens, the space-prefix diagnosis is confirmed.

---

## Cell 10 — tokenization statistics

```python
from collections import Counter

def n_tokens(model, word, space=True):
    s = (" " + word) if space else word
    return len(model.to_str_tokens(s, prepend_bos=False))

for lang, i in [("Japanese", 0), ("English", 1), ("German", 2)]:
    counts = Counter(n_tokens(model, w[i]) for w in WORDS)
    single = counts.get(1, 0)
    print(f"{lang:9} single-token {single:2}/{len(WORDS)} = {single/len(WORDS):4.0%}   lengths {dict(sorted(counts.items()))}")
```

`CHANGED:` new cell. You printed tokenizations for five words but never aggregated.

This is the single most citable number in your write-up. Wendler reports single-token fractions of 100% Chinese, 55% French, 43% German, 13% Russian, and uses exactly that spread to explain why the English detour is weaker in some languages than others. Your Japanese fraction is what connects your result to their explanation. Measured on the space-prefixed form, since that is what the model must emit.

---

## Cell 11 — filter

```python
clean, dropped = [], []
for ja, en, de in WORDS:
    S = {"ja": start_token_ids(model, ja),
         "en": start_token_ids(model, en),
         "de": start_token_ids(model, de)}
    clash = [(a, b) for a in S for b in S if a < b and S[a] & S[b]]
    if clash:
        dropped.append((ja, en, de, clash))
    else:
        clean.append((ja, en, de, sorted(S["ja"]), sorted(S["en"]), sorted(S["de"])))

print(f"{len(clean)}/{len(WORDS)} usable triples")
for ja, en, de, clash in dropped:
    print(f"  dropped {ja} / {en} / {de} — overlapping sets {clash}")
```

`CHANGED:` was `len(set(ids)) == 3`, requiring three distinct *first tokens*. That was fine while you scored one token each. Now that Cell 9 produces *sets*, two sets can overlap even when their first elements differ, so the check must be disjointness.

Wendler drops words whose target-language form shares a token prefix with the English form — their example is French *photographier* against English *photograph*. Print the dropped list; it is evidence you did the filter and belongs in an appendix.

Worth stating in the write-up: German capitalises nouns and English does not, so `▁Silber`/`▁silver` and `▁Blut`/`▁blood` separate on case alone. Wendler's French–English pair had no such protection, so your collision risk is structurally lower than theirs.

---

## Cell 12 — the lens

```python
import gc

def lens_probs(model, prompt, id_sets):
    """
    id_sets : list of iterables of token ids, one per language.
    returns : (probs [n_rows, n_langs], labels)
    """
    with torch.no_grad():
        _, cache = model.run_with_cache(prompt)
        resid, labels = cache.accumulated_resid(layer=-1, incl_mid=False, return_labels=True)
        resid = cache.apply_ln_to_stack(resid, layer=-1)[:, 0, -1, :]
        del cache

        # Cast the small residual DOWN to W_U's dtype rather than upcasting W_U:
        # W_U is 2304 x 256000, and an fp32 copy is ~2.4 GB, which OOMs.
        lg = (resid.to(model.W_U.dtype) @ model.W_U).float() + model.b_U.float()

        # Gemma-2 applies logit soft-capping internally. A hand-rolled lens
        # bypasses that path, so reapply it or every probability is inflated.
        cap = getattr(model.cfg, "output_logits_soft_cap", None)
        if cap:
            lg = cap * torch.tanh(lg / cap)

        p = lg.softmax(-1)
        out = torch.stack([p[:, list(ids)].sum(-1) for ids in id_sets], dim=-1).cpu()

    gc.collect(); torch.cuda.empty_cache()
    return out, labels
```

`CHANGED:` takes id **sets** and sums within each, instead of indexing three single ids. Everything else — the final-norm application, the dtype trick, the soft cap — was already correct and is unchanged.

---

## Cell 13 — verify the lens

```python
ja, en, de, ja_ids, en_ids, de_ids = clean[0]
prompt   = make_prompt(ja, WORDS)
id_sets  = [ja_ids, en_ids, de_ids]

probs, labels = lens_probs(model, prompt, id_sets)

real_full = model(prompt)[0, -1].float().softmax(-1)
real = torch.tensor([real_full[list(s)].sum().item() for s in id_sets])

print("lens final:", [f"{v:.4f}" for v in probs[-1].tolist()])
print("model     :", [f"{v:.4f}" for v in real.tolist()])
assert torch.allclose(probs[-1], real, rtol=0.05, atol=1e-4), (probs[-1], real)
print("lens verified")
print(f"{len(labels)} rows;", labels[:3], "...", labels[-2:])
```

`CHANGED:` relative tolerance instead of absolute, and it now prints `len(labels)`.

**Correction, 24 Jul.** An earlier version of this document said to tighten `atol` from 2e-2 to 5e-3. That was wrong and it made this cell fail. Observed on `clean[0]`:

| | lens | model | abs Δ | rel Δ |
|---|---|---|---|---|
| Japanese | 2.7759e-05 | 2.6900e-05 | 8.6e-07 | 3.19% |
| English | 6.4465e-05 | 6.4192e-05 | 2.7e-07 | 0.43% |
| German | 0.72253 | 0.70833 | 0.0142 | 2.00% |

Only German tripped the threshold, because it is the only value large enough for a 2% relative error to exceed an absolute one — Japanese was off by *more* in relative terms and passed. That is the argument for `rtol` in one line. The original `atol=2e-2` was correctly calibrated.

~2% relative error is expected from a 2304-dimensional dot product in bfloat16 (~8 mantissa bits).

**Resolved, 24 Jul.** The diagnostic below returned `max |Δlogit| = 0.1252`, `mean Δlogit = 5.03e-4`, and identical top-5 token lists in identical order (`' Blume', ' Blumen', ' Bl', ' Blüten', ' Pflanze'`). Mean deviation is 0.4% of the max, i.e. centred noise — and a uniform logit shift cancels exactly in the softmax anyway, so it could not have caused the 2% probability gap. Per-token scatter of ±0.125 can. Verdict: bf16 rounding. An earlier soft-cap-precision hypothesis is not supported.

**Before accepting that it is noise, run this once:**

```python
with torch.no_grad():
    _, cache = model.run_with_cache(prompt)
    r, labels = cache.accumulated_resid(layer=-1, incl_mid=False, return_labels=True)
    r = cache.apply_ln_to_stack(r, layer=-1)[:, 0, -1, :]
    del cache
    lens_lg = (r[-1].to(model.W_U.dtype) @ model.W_U).float() + model.b_U.float()
    cap = getattr(model.cfg, "output_logits_soft_cap", None)
    if cap:
        lens_lg = cap * torch.tanh(lens_lg / cap)
    real_lg = model(prompt)[0, -1].float()

d = lens_lg - real_lg
print("max |Δlogit| :", d.abs().max().item())
print("mean Δlogit  :", d.mean().item())
print("top-5 lens   :", [model.to_string(t) for t in lens_lg.topk(5).indices])
print("top-5 model  :", [model.to_string(t) for t in real_lg.topk(5).indices])
```

Mean Δ near zero and matching top-5 lists → rounding, proceed. Mean Δ systematically nonzero → the norm or the soft cap differs between the two paths, and no tolerance should paper over it.

Record the observed max Δlogit and put it in the write-up: "lens and model agree to within 2% relative, consistent with bf16" is stronger evidence than an assertion that merely passed.

Expect 27 rows — embedding plus 26 layers. That is why `argmax` is not a layer number, fixed in Cell 17.

**Understand what this assertion does not cover.** It proves your lens reproduces the model's output distribution — norm, unembedding, soft cap all wired right. It does *not* prove your token ids are the ones the model would emit, because both sides index the same ids. A wrong id passes silently. That is exactly how the Cell 9 bug survived.

---

## Cell 14 — behavioural check, both conditions

```python
def behavioural_check(prompt_fn, target_idx, label, n=12):
    correct = 0
    for i, (ja, en, de) in enumerate(WORDS[:n]):
        want = (ja, en, de)[target_idx]
        top  = model(prompt_fn(ja, WORDS, seed=i))[0, -1].softmax(-1).topk(3).indices
        toks = [model.to_string(t) for t in top]
        hit  = any(t.strip() and want.startswith(t.strip()) for t in toks)
        correct += hit
        print(f"  {ja} -> {want:10} {'OK ' if hit else '   '} {toks}")
    print(f"{label}: {correct}/{n}\n")
    return correct / n

acc_tr   = behavioural_check(make_prompt,        2, "translation accuracy")
acc_ctrl = behavioural_check(make_repeat_prompt, 0, "repetition accuracy")
```

`CHANGED:` two things. It now runs on the **control** as well as translation — you only ever checked translation, and running this on the control would have caught the Cell 9 bug on day one. And `seed=i` varies the demonstration set per word instead of every word using seed 0.

Report both accuracies in the write-up. If the model cannot do the task, nothing measured inside it means anything.

---

## Cell 15 — sweep

```python
import numpy as np

def sweep(prompt_fn, n_seeds=3):
    curves, word_idx, labels = [], [], None
    for wi, (ja, en, de, ja_ids, en_ids, de_ids) in enumerate(clean):
        for s in range(n_seeds):
            p = prompt_fn(ja, WORDS, seed=s)
            pr, labels = lens_probs(model, p, [ja_ids, en_ids, de_ids])
            curves.append(np.asarray(pr))
            word_idx.append(wi)
    return np.stack(curves), np.array(word_idx), labels
```

`CHANGED:` returns `word_idx` and `labels` alongside the curves.

`word_idx` matters because your three seeds share the same 24 target words, so the independent unit is the **word**, not the prompt. Any confidence interval must resample words. Without this array you cannot compute an honest one.

---

## Cell 16 — plotting

```python
import matplotlib.pyplot as plt

def per_word_means(curves, word_idx):
    return np.stack([curves[word_idx == w].mean(0) for w in np.unique(word_idx)])

def plot_curves(curves, word_idx, labels, title, fname=None):
    pw   = per_word_means(curves, word_idx)
    mean = pw.mean(0)
    ci   = 1.96 * pw.std(0, ddof=1) / np.sqrt(len(pw))

    plt.figure(figsize=(9, 5))
    for k, name in enumerate(["Japanese (source)", "English (pivot?)", "German (target)"]):
        plt.plot(mean[:, k], marker="o", label=name)
        plt.fill_between(range(len(mean)), mean[:, k] - ci[:, k], mean[:, k] + ci[:, k], alpha=0.2)
    plt.xlabel("residual stream position (0 = embedding)")
    plt.ylabel("mean probability")
    plt.title(f"{title}   n={len(pw)} words, shaded = 95% CI")
    plt.legend()
    if fname:
        plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()

    i = int(mean[:, 1].argmax())
    print(f"peak P(English) = {mean[i,1]:.4f} ± {ci[i,1]:.4f} at {labels[i]} (row {i})")
    return mean, ci
```

`CHANGED:` three fixes.

The band was ±1 SEM (~68%); Wendler reports **95% CIs**, so comparing yours to theirs was comparing different things. Now 1.96 × SEM, and the legend says so.

Variance is computed over **words** via `per_word_means`, not over all 72 prompts, because the prompts aren't independent.

The peak is now reported as `labels[i]`, not the raw index. With 27 rows, your previously reported "layer 23" may be off by one.

---

## Cell 17 — run translation

```python
curves_tr, widx_tr, labels = sweep(make_prompt)
np.save(OUT / "curves_translation.npy", curves_tr)
np.save(OUT / "widx_translation.npy",   widx_tr)
mean_tr, ci_tr = plot_curves(curves_tr, widx_tr, labels,
                             "JA→DE translation", OUT / "fig_translation.png")
```

---

## Cell 18 — run control

```python
curves_ct, widx_ct, labels = sweep(make_repeat_prompt)
np.save(OUT / "curves_control.npy", curves_ct)
np.save(OUT / "widx_control.npy",   widx_ct)
mean_ct, ci_ct = plot_curves(curves_ct, widx_ct, labels,
                             "JA→JA repetition control", OUT / "fig_control.png")
```

`CHANGED:` Cells 17–18 use `OUT` rather than hardcoded `/kaggle/working/`, and save the word-index arrays too.

---

## Cell 19 — summary table

```python
print(f"{'condition':<24}{'peak P(EN)':>12}{'±95% CI':>10}{'position':>12}")
for name, m, c in [("JA→DE translation", mean_tr, ci_tr),
                   ("JA→JA repetition",  mean_ct, ci_ct)]:
    i = int(m[:, 1].argmax())
    print(f"{name:<24}{m[i,1]:>12.4f}{c[i,1]:>10.4f}{labels[i]:>12}")
```

---

## Cell 20 — does the control actually separate?

```python
rng = np.random.default_rng(0)

def peak_en(curves, word_idx, pick):
    """Mean peak P(English) over a bootstrap resample of words."""
    words = np.unique(word_idx)
    pw = np.stack([curves[word_idx == words[i]].mean(0) for i in pick])
    return pw.mean(0)[:, 1].max()

n_words = len(np.unique(widx_tr))
full    = np.arange(n_words)
obs     = peak_en(curves_tr, widx_tr, full) - peak_en(curves_ct, widx_ct, full)

boot = np.array([
    peak_en(curves_tr, widx_tr, idx) - peak_en(curves_ct, widx_ct, idx)
    for idx in (rng.integers(0, n_words, n_words) for _ in range(2000))
])
lo, hi = np.percentile(boot, [2.5, 97.5])

print(f"Δ peak P(English), translation − repetition = {obs:+.4f}")
print(f"95% bootstrap CI over {n_words} words: [{lo:+.4f}, {hi:+.4f}]")
print("→ conditions separate" if (lo > 0 or hi < 0)
      else f"→ cannot resolve a difference at n={n_words}")
```

`CHANGED:` new cell, and it decides your entire write-up.

Your two peaks currently differ by 0.006 (0.6215 vs 0.6156). Whether that is a null result or noise is exactly what this interval tells you, and you have not computed it.

- **Interval excludes zero** → the conditions separate. 24 words was enough. Report it.
- **Interval straddles zero** → your claim is not "the control does not separate", it is "I cannot resolve a difference at n=24". Only one of those is honest, and it is the second.

If it straddles zero and you have time left, expand `WORDS` to ~50 using the same criteria — concrete single-kanji nouns, unambiguous translation. You study Japanese; that is a real advantage here. Budget 45 minutes, then re-run from Cell 10. Do this **only** if this cell says you need to.

---

# Stale-state hazard

The notebook has one dependency chain that does not error when broken:

```
Cell 9  start_token_ids  ->  Cell 11  clean  ->  Cell 15  sweep  ->  Cells 17-20  results
```

Redefining `start_token_ids` does **not** rebuild `clean`. If you change Cell 9 and re-run the sweep without re-running Cell 11, everything executes cleanly and the results silently come from the previous scoring. There is no traceback and no warning.

Check before trusting any sweep output:

```python
print("ids per language in clean[0]:", [len(x) for x in clean[0][3:6]])
```

Length 1 for German or English means first-token-only ids — Cell 11 is stale, rebuild it. Prefix summation should give more than one for at least the European words.

**Rule: after editing any of Cells 6–12, re-run from Cell 11 downward before sweeping.** Cheaper insurance is *Run All* after a restart once the notebook is stable, which also proves it runs top to bottom — something a reviewer will assume and you should verify at least once before submitting.

Related: `OUT` is defined in Cell 5. If Cells 17–18 raise `NameError: name 'OUT' is not defined`, the Cell 5 change was never applied. Re-running Cell 5 will not re-prompt for the HF token, because `os.environ.get("HF_TOKEN")` short-circuits the `getpass` once the variable is set in the session.

---

# What to do, in order

| # | Task | Time |
|---|---|---|
| 1 | Set up git push from inside the Kaggle session (below) | 20 min |
| 2 | Run Cell 9's two `to_str_tokens` lines — confirm the bug | 5 min |
| 3 | Apply Cells 9–16, re-run 17–20 | 2 hr |
| 4 | Read Cell 20's verdict, decide the write-up's claim | — |
| 5 | Repo tidy, README, requirements.txt | 3 hr |
| 6 | Write-up (1,000 words, five pillar headings) and video | 1.5 days |

Keep the old `.npy` files. A before/after on the scoring fix is good material — a detail buried in an appendix decided whether a published result replicated.

---

# Protecting your results

**This is the highest-risk item and it is not about code.**

In the Kaggle web notebook, *Save Version* commits `/kaggle/working/` as a persistent output. Over a tunnel or SSH there is no equivalent: when the session ends the container is destroyed and everything in `/kaggle/working/` goes with it. Sessions cap at 12 hours and drop earlier if the connection dies.

Clone the repo inside the session and push after every sweep:

```bash
git clone https://github.com/<you>/does-gemma-work-in-english.git /kaggle/working/repo
cd /kaggle/working/repo
git add results/ && git commit -m "sweep n=72, scoring fixed" && git push
```

The `.npy` files are small. Commit them — they also let a reviewer regenerate your figures without a GPU.

Free tier gives roughly 30 GPU-hours per week on a 12-hour session cap, and a tunnel burns quota while you edit or sleep. Shut the session down on days 5–7; prose and video need no GPU.

---

# Repo layout

```
does-gemma-work-in-english/
├── README.md
├── requirements.txt
├── notebooks/
│   └── mechanistic_interpretability.ipynb
├── src/
│   ├── prompts.py      # make_prompt, make_repeat_prompt
│   ├── scoring.py      # start_token_ids, lens_probs
│   └── words.py        # WORDS
├── results/
│   ├── curves_*.npy
│   ├── widx_*.npy
│   ├── fig_*.png
│   └── tokenization_stats.md
└── writeup.md
```

With VS Code the `src/` split is worth doing — put `%load_ext autoreload` and `%autoreload 2` at the top of the notebook and you can edit `scoring.py` and re-run a cell **without reloading gemma-2-2b**, which otherwise costs minutes each time. Note autoreload handles pure-Python modules only; it does not substitute for the Cell 1 restart.

**README order** — reviewers read the repo before the write-up:

1. One sentence on what this does.
2. Wendler et al. citation and a link to `epfl-dlab/llm-latent-language`, stating plainly what is theirs (method, task design, word-selection criteria) and what is yours (gemma-2-2b, Japanese, the scoring audit).
3. Headline result in two or three lines, with numbers and the CI.
4. Reproduction: environment from Cell 3, the restart requirement, runtime, hardware.
5. Limitations — link to the write-up rather than duplicating.

Topics: `mechanistic-interpretability`, `logit-lens`, `transformerlens`, `multilinguality`, `gemma`.

---

# Deviations to disclose in the write-up

Not bugs — choices, which reviewers penalise only if hidden.

- **No quotes in prompts.** Wendler wraps words in quotes so the answer has no leading space; yours ends on a colon so it does. Internally consistent because Cell 9 scores both variants.
- **24 words vs their 139/104/56/115.** Cell 20 decides whether this is adequate.
- **Prefix filtering by set disjointness**, not their exact procedure. Justify via German noun capitalisation.
- **bf16 matmul in the lens**, ~8 bits of mantissa. Measured: lens and model final-layer probabilities agree to ~2% relative (0.7225 vs 0.7083 on the first triple). Quote that number rather than implying exactness.
- **No tuned lens** — and say why. Wendler declines it deliberately: the tuned lens is trained to map intermediate states to the final non-English prediction, which would optimize away the signal. Stating this is worth more than running it.
