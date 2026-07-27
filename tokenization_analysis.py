"""Does single-token fraction predict the English detour? Answered locally.

The notebook runs one model per kernel session, because loading a second model
into a live kernel produced a silently broken load. That makes the across-model
comparison -- the cleanest test in the design, since it holds language, words,
prompt and scoring fixed while only the tokenizer moves -- impossible to compute
inside any single session. So it is computed here, over whatever sessions have
been extracted into results/.

    python tokenization_analysis.py          # every model found in results/
    python tokenization_analysis.py --check  # additionally assert recorded values

Inputs, produced by `extract_results.py tokenization_vs_detour.ipynb`:

    results/<model>__<lang>__curves.npy   [prompt, position, 2]  0=target 1=English
    results/<model>__<lang>__widx.npy     word index per prompt
    results/summary_<tag>.json            single-token fractions, accuracies, drops

Three things this script refuses to do, all of which would flatter the result:

  * It will not include a cell whose copy accuracy is below MIN_ACC. A model that
    cannot do the task has a curve that means nothing, and including it would let
    task failure masquerade as a weak detour.
  * It will not include a cell whose final-row P(target) is below MIN_TGT. That is
    the signature of a lens that is not reading its model, and it is invisible to
    the accuracy check -- see verdict() for why this one is load-bearing.
  * It will not report the pooled correlation as the headline. Cells sharing a
    model share a tokenizer and a training corpus; cells sharing a language share
    a word list. Pooling double-counts. The stratified breakdowns are the honest
    version and are printed first.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MIN_ACC = 0.7       # copy accuracy: can the model do the task at all
MIN_TGT = 0.20      # final-row P(target): is the lens actually reading the model
EN = 1              # curve axis: 0 = target language, 1 = English

# Recorded values, asserted by --check. Filled in once a run is trusted; see
# analysis.py for why this exists (isolating the analysis as a constant so any
# future movement is attributable upstream).
BASELINE = {}

# Filled by report() -> task_contrast(); written into the summary JSON by main().
# A module-level handoff rather than a return value threaded through report(), whose
# signature is already the boundary between "what was measured" and "what is said".
CONTRAST = {}


TASKS = ("repeat", "translate")


def split_key(parts):
    """(model, lang, task) from the trailing components of a name.

    Two layouts coexist and both must keep working. The first sweep emitted
    "<model>__<lang>" with no task field, because the notebook only ran repetition
    at the time; those arrays are still the repetition data and are read as
    task="repeat" rather than re-run. Everything since carries the task explicitly.
    Sniffing the last component against the known task names is what lets one
    results/ directory hold both without a migration step.
    """
    if parts and parts[-1] in TASKS:
        return "__".join(parts[:-2]), parts[-2], parts[-1]
    return "__".join(parts[:-1]), parts[-1], "repeat"


def load(results):
    """Every (model, language, task) cell present in results/, with its metadata."""
    meta = {}
    for path in sorted(results.glob("summary*.json")):
        for key, val in json.loads(path.read_text(encoding="utf-8")).items():
            model, lang, task = split_key(key.split("|"))
            meta[(model.split("/")[-1], lang, task)] = val

    cells = {}
    for path in sorted(results.glob("*__*__curves.npy")):
        m = re.fullmatch(r"(.+)__curves", path.stem)
        if not m:
            continue
        model, lang, task = split_key(m.group(1).split("__"))
        if not model or not lang:
            continue
        stem = m.group(1)
        widx = path.with_name(f"{stem}__widx.npy")
        if not widx.exists():
            print(f"  ! {path.name} has no matching widx -- skipped")
            continue
        info = meta.get((model, lang, task))
        if info is None:
            print(f"  ! {model}/{lang}/{task} has arrays but no summary entry -- skipped, "
                  "single-token fraction is unknown so it cannot be placed on the x-axis")
            continue
        cells[(model, lang, task)] = dict(
            curves=np.load(path), widx=np.load(widx),
            stf=info["single_token_frac"], acc=info["accuracy"],
            n_words=info["n_words"], d_vocab=info["d_vocab"],
            words=info.get("words", []), task=task,
        )
    return cells


def summarise(cell, n_boot=2000, seed=0):
    """Peak and area of the English curve, with bootstrap CIs resampling words."""
    c, w = cell["curves"], cell["widx"]
    per_word = np.stack([c[w == u].mean(0) for u in np.unique(w)])
    en = per_word[:, :, EN]
    mean = en.mean(0)

    rng = np.random.default_rng(seed)
    b = en[rng.integers(0, len(en), (n_boot, len(en)))].mean(1)
    peak_ci = np.percentile(b.max(1), [2.5, 97.5])
    auc_ci = np.percentile(b.sum(1), [2.5, 97.5])

    return dict(
        peak=float(mean.max()), peak_lo=float(peak_ci[0]), peak_hi=float(peak_ci[1]),
        auc=float(mean.sum()), auc_lo=float(auc_ci[0]), auc_hi=float(auc_ci[1]),
        peak_pos=int(mean.argmax()),
        peak_depth=float(mean.argmax() / (len(mean) - 1)),   # comparable across depths
        target_final=float(per_word[:, -1, 0].mean()),
        mean_en=mean,
    )


def spearman(x, y):
    """Rank correlation without a scipy dependency; ties averaged.

    No p-value is returned, deliberately. At these n a p-value invites exactly the
    over-reading the notebook's own commentary warns against.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    rank = lambda v: np.array([np.mean(np.flatnonzero(np.sort(v) == vi)) + 1 for vi in v])
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom else None


def verdict(cell, summ):
    """Why a cell is or is not usable. Two independent ways to be unusable.

    MIN_TGT is the analysis-side twin of the notebook's verify_lens gate, and it
    exists because the behavioural gate provably cannot catch a broken lens: the
    behavioural check calls model() directly while every curve comes through the
    lens. A dead lens yields a uniform distribution -- about 1e-5 per token -- so
    its curves read as a flat, plausible-looking, entirely meaningless zero, and
    they sail through a copy-accuracy check at 100%.

    The tell is the final row. The model demonstrably copies the word, so P(target)
    at the last position must be high. If it is not, the lens is not reading the
    model, and the fault is upstream of anything this script can interpret. That is
    not a weak detour, and treating it as one silently inverts the result: on the
    pre-fix arrays, four dead Llama cells at peak 0.000 turned a positive
    correlation into a negative one.
    """
    if cell["acc"] < MIN_ACC:
        return f"excluded: copy accuracy {cell['acc']:.0%} < {MIN_ACC:.0%}, task failure"
    if summ["target_final"] < MIN_TGT:
        return (f"EXCLUDED: final P(target) {summ['target_final']:.3f} -- the lens is "
                "not reading this model, curves are not a measurement")
    return None


def coverage(cells):
    """What the design asked for against what exists. Printed before any result.

    The first pass of this experiment reported five repetition cells as though they
    were the finding, while the translation half of the design had never been built
    and two of three models had never produced a number. Stating the fraction up
    front makes that impossible to do again by accident.
    """
    models = sorted({k[0] for k in cells})
    langs = sorted({k[1] for k in cells})
    print("\n--- COVERAGE (design vs delivered) ---")
    print(f"{'model':<20}" + "".join(f"{t:>12}" for t in TASKS))
    for m in models:
        row = f"{m:<20}"
        for t in TASKS:
            got = sorted(k[1] for k in cells if k[0] == m and k[2] == t)
            row += f"{len(got):>7} langs" if got else f"{'--':>12}"
        print(row)
    print(f"  languages seen: {langs}")
    print("  A model-by-task pair with no entry was not run, not run-and-excluded.")


def report(cells, summ):
    keys = sorted(summ)
    reasons = {k: verdict(cells[k], summ[k]) for k in keys}
    used = [k for k in keys if reasons[k] is None]

    coverage(cells)
    print(f"\n{len(cells)} (model, language, task) cells from "
          f"{len({k[0] for k in keys})} model(s)\n")
    print(f"{'model':<20}{'task':<11}{'lang':<6}{'vocab':>8}{'STF':>7}{'acc':>6}"
          f"{'P(tgt)':>8}{'peakEN':>9}{'AUC':>8}{'depth':>7}")
    for k in sorted(keys, key=lambda k: (k[2], k[0], k[1])):
        c, s = cells[k], summ[k]
        print(f"{k[0]:<20}{k[2]:<11}{k[1]:<6}{c['d_vocab']:>8}{c['stf']:>6.0%}"
              f"{c['acc']:>6.0%}{s['target_final']:>8.3f}{s['peak']:>9.3f}"
              f"{s['auc']:>8.3f}{s['peak_depth']:>7.2f}")

    for k in keys:
        if reasons[k]:
            print(f"  {k[0]}/{k[1]}/{k[2]}: {reasons[k]}")
    if not used:
        print("\nnothing usable -- no correlation to report")
        return used

    # Every correlation below is computed WITHIN a task. The two tasks are separate
    # experiments against the same x-axis; pooling them would fit one line through
    # two conditions and describe neither.
    print("\n--- WITHIN LANGUAGE, ACROSS MODELS "
          "(same words, same prompt, different tokenizer) ---")
    print("The cleanest comparison available: only the tokenizer moves.")
    any_within = False
    for task in TASKS:
        for lang in sorted({k[1] for k in used if k[2] == task}):
            ks = [k for k in used if k[1] == lang and k[2] == task]
            if len(ks) < 2:
                continue
            any_within = True
            row = "  ".join(f"{k[0]} STF {cells[k]['stf']:.0%} -> peak {summ[k]['peak']:.3f}"
                            for k in sorted(ks, key=lambda k: cells[k]["stf"]))
            rho = spearman([cells[k]["stf"] for k in ks], [summ[k]["peak"] for k in ks])
            tail = f"   rho = {rho:+.3f}" if rho is not None else "   (need 3+ models for rho)"
            print(f"  {task}/{lang}: {row}{tail}")
    if not any_within:
        print("  not computable -- only one model has been swept so far.")
        print("  Run the notebook again on another model, extract, and re-run this.")

    print("\n--- WITHIN MODEL, ACROSS LANGUAGES (tokenizer fixed) ---")
    for model in sorted({k[0] for k in used}):
        for task in TASKS:
            ks = [k for k in used if k[0] == model and k[2] == task]
            if not ks:
                continue
            for stat in ("peak", "auc"):
                rho = spearman([cells[k]["stf"] for k in ks], [summ[k][stat] for k in ks])
                if rho is None:
                    print(f"  {model:<18} {task:<10} {stat:<5} too few languages ({len(ks)})")
                else:
                    direction = "NEGATIVE" if rho < 0 else "POSITIVE"
                    print(f"  {model:<18} {task:<10} {stat:<5} rho = {rho:+.3f}  "
                          f"({len(ks)} langs)  Wendler predicts negative; "
                          f"observed {direction}")

    CONTRAST.update(task_contrast(cells, summ, used))

    # Printed last and labelled, so it is not mistaken for the headline.
    print("\n--- POOLED WITHIN TASK "
          "(reported for completeness; the cells are not independent) ---")
    for task in TASKS:
        ks = [k for k in used if k[2] == task]
        for stat in ("peak", "auc"):
            rho = spearman([cells[k]["stf"] for k in ks], [summ[k][stat] for k in ks])
            if rho is not None:
                print(f"  {task:<10} {stat:<5} rho = {rho:+.3f} over {len(ks)} cells")
    return used


def task_contrast(cells, summ, used, n_boot=2000, seed=0):
    """Translation minus repetition, per (model, language), paired over words.

    This is the tokenization notebook's version of the companion notebook's headline,
    and it is the reason both tasks have to be run: if the relationship between
    single-token fraction and the detour has a different sign in the two tasks, then
    "the English detour" is not one phenomenon and one explanation cannot cover it.

    The two tasks do not always cover the same words -- the translation sweep drops
    the 17 concepts written identically in JA and ZH -- so the contrast is computed
    on the INTERSECTION and the surviving n is printed. Comparing a 54-word mean
    against a 37-word mean would put a word-list difference into a task difference.
    """
    pairs = sorted({(k[0], k[1]) for k in used if (k[0], k[1], "repeat") in summ
                    and (k[0], k[1], "translate") in summ})
    print("\n--- TRANSLATION vs REPETITION, same model and language ---")
    if not pairs:
        print("  no (model, language) has both tasks yet -- this contrast needs a")
        print("  session with TASKS covering the half that is missing.")
        return {}

    print("  Paired over the words both tasks share; CI resamples those words.")
    print(f"  {'model':<16}{'lang':<5}{'STF':>5}{'n':>5}{'d peak':>9}"
          f"{'95% CI':>18}{'d AUC':>9}{'95% CI':>18}")
    out = {}
    for model, lang in pairs:
        cr = _per_concept_en(cells[(model, lang, "repeat")])
        ct = _per_concept_en(cells[(model, lang, "translate")])
        common = sorted(set(cr) & set(ct))
        if not common:
            print(f"  {model:<16}{lang:<5} no shared words between the two task sweeps")
            continue
        a = np.stack([ct[w] for w in common])       # translation
        b = np.stack([cr[w] for w in common])       # repetition
        n = len(common)

        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, (n_boot, n))       # one resample, both tasks
        ba, bb = a[idx].mean(1), b[idx].mean(1)
        pk = np.percentile(ba.max(1) - bb.max(1), [2.5, 97.5])
        au = np.percentile(ba.sum(1) - bb.sum(1), [2.5, 97.5])
        d_pk = float(a.mean(0).max() - b.mean(0).max())
        d_au = float(a.mean(0).sum() - b.mean(0).sum())
        star = lambda ci: "*" if ci[0] * ci[1] > 0 else " "
        print(f"  {model:<16}{lang:<5}{cells[(model, lang, 'repeat')]['stf']:>4.0%}"
              f"{n:>5}{d_pk:>+9.3f}"
              f"{'[' + f'{pk[0]:+.3f},{pk[1]:+.3f}' + ']' + star(pk):>18}"
              f"{d_au:>+9.3f}"
              f"{'[' + f'{au[0]:+.3f},{au[1]:+.3f}' + ']' + star(au):>18}")
        out[f"{model}|{lang}"] = dict(
            n_paired=n, diff_peak=d_pk, diff_peak_lo=float(pk[0]),
            diff_peak_hi=float(pk[1]), diff_auc=d_au, diff_auc_lo=float(au[0]),
            diff_auc_hi=float(au[1]), n_boot=n_boot, seed=seed)
    print("  Positive = translation carries more English than repetition. * = CI excludes 0.")
    print("  Several languages x two statistics, uncorrected -- read the sizes, not the stars.")
    return out


def _per_concept_en(cell):
    """{word form: English curve}, one entry per distinct word, averaged over seeds."""
    us = np.unique(cell["widx"])
    per_word = np.stack([cell["curves"][cell["widx"] == u].mean(0) for u in us])
    words = cell["words"][:len(per_word)]
    return {w: per_word[i, :, EN] for i, w in enumerate(words)}


def shared_character_control(cells, summ, used, n_boot=2000, seed=0):
    """JA and ZH concepts written with the same character are the SAME token.

    Tokenization is therefore held exactly constant -- not approximately -- and only
    the language framing of the prompt changes. Any difference on this subset cannot
    be tokenization.

    The interval is a PAIRED bootstrap: concepts are resampled, and the same resample
    indexes both languages, because the same concept contributes to both sides. An
    unpaired interval would be wider and would not be measuring this comparison.
    Two statistics are tested on n<=17 concepts with no correction, so an interval
    that just clears zero is suggestive and is reported as such.
    """
    print("\n--- SHARED-CHARACTER CONTROL (identical target tokens, JA vs ZH) ---")
    print("Repetition only, by construction: the control works because the same")
    print("character is the answer under both framings, and the translation sweep")
    print("drops exactly those concepts because JA->ZH would show the answer in the")
    print("question. There is no translation version of this control to run.")
    results, done = {}, False
    for model in sorted({k[0] for k in used}):
        ja, zh = (model, "ja", "repeat"), (model, "zh", "repeat")
        if ja not in used or zh not in used:
            continue
        wa, wb = cells[ja].get("words", []), cells[zh].get("words", [])
        shared = set(wa) & set(wb)
        if not shared:
            print(f"  {model}: no shared word forms recorded in the summary -- skipped")
            continue

        cja, czh = _per_concept_en(cells[ja]), _per_concept_en(cells[zh])
        common = sorted(w for w in shared if w in cja and w in czh)
        a = np.stack([cja[w] for w in common])          # [n_concepts, n_positions]
        b = np.stack([czh[w] for w in common])
        en_ja, en_zh, n = a.mean(0), b.mean(0), len(common)

        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, (n_boot, n))           # one resample, both languages
        ba, bb = a[idx].mean(1), b[idx].mean(1)
        peak_ci = np.percentile(ba.max(1) - bb.max(1), [2.5, 97.5])
        auc_ci = np.percentile(ba.sum(1) - bb.sum(1), [2.5, 97.5])
        d_peak = float(en_ja.max() - en_zh.max())
        d_auc = float(en_ja.sum() - en_zh.sum())

        def verdict(ci):
            return "excludes 0" if ci[0] * ci[1] > 0 else "contains 0"

        print(f"  {model}   n={n} shared concepts")
        print(f"    ja  peak {en_ja.max():.3f}  AUC {en_ja.sum():.3f}")
        print(f"    zh  peak {en_zh.max():.3f}  AUC {en_zh.sum():.3f}")
        print(f"    diff peak {d_peak:+.3f}  95% CI [{peak_ci[0]:+.3f}, "
              f"{peak_ci[1]:+.3f}]  {verdict(peak_ci)}")
        print(f"    diff AUC  {d_auc:+.3f}  95% CI [{auc_ci[0]:+.3f}, "
              f"{auc_ci[1]:+.3f}]  {verdict(auc_ci)}")
        print("    tokenization is held exactly constant here, so whatever moves "
              "is not tokenization")
        results[model] = dict(
            n_shared=n, ja_peak=float(en_ja.max()), ja_auc=float(en_ja.sum()),
            zh_peak=float(en_zh.max()), zh_auc=float(en_zh.sum()),
            diff_peak=d_peak, diff_peak_lo=float(peak_ci[0]),
            diff_peak_hi=float(peak_ci[1]),
            diff_auc=d_auc, diff_auc_lo=float(auc_ci[0]),
            diff_auc_hi=float(auc_ci[1]), n_boot=n_boot, seed=seed)
        done = True
    if not done:
        print("  needs both ja and zh swept on the same model -- not available yet")
    return results


def figure(cells, summ, used, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Categorical slots 1-3, validated colourblind-safe on a white surface.
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    models = sorted({k[0] for k in used})
    colour = {m: palette[i % len(palette)] for i, m in enumerate(models)}
    ink, muted, grid, axis = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"

    # Task is marker shape, model is colour. The two tasks are separate experiments
    # sharing an x-axis, so they get separate trend lines and are never fitted jointly.
    mark = {"repeat": "o", "translate": "^"}
    line = {"repeat": "--", "translate": ":"}
    tasks_seen = [t for t in TASKS if any(k[2] == t for k in used)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, stat, label in ((axes[0], "peak", "peak P(English)"),
                            (axes[1], "auc", "total English (area under curve)")):
        for k in used:
            x, s = cells[k]["stf"], summ[k]
            ax.errorbar(x, s[stat], yerr=[[s[stat] - s[stat + "_lo"]],
                                          [s[stat + "_hi"] - s[stat]]],
                        fmt=mark.get(k[2], "s"), ms=7, color=colour[k[0]],
                        capsize=3, lw=1.2)
            ax.annotate(k[1], (x, s[stat]), textcoords="offset points",
                        xytext=(7, 4), fontsize=9, color=muted)
        for t in tasks_seen:
            kt = [k for k in used if k[2] == t]
            xs = np.array([cells[k]["stf"] for k in kt])
            ys = np.array([summ[k][stat] for k in kt])
            if len(set(xs)) > 2:
                b, a = np.polyfit(xs, ys, 1)
                grid_x = np.linspace(xs.min(), xs.max(), 10)
                ax.plot(grid_x, a + b * grid_x, ls=line.get(t, "-"), lw=1.2,
                        color=muted, zorder=0)
        ax.set_xlabel("single-token fraction of the target language", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(axis)
        ax.tick_params(colors=muted, labelsize=9, length=0)

    # A legend for one series puts a lone marker inside the axes, where it reads as
    # a sixth data point. One model and one task -> name them in the title instead.
    handles = []
    if len(tasks_seen) > 1:
        handles += [plt.Line2D([], [], marker=mark.get(t, "s"), ls="", color=ink,
                               label=t) for t in tasks_seen]
    if len(models) > 1:
        handles += [plt.Line2D([], [], marker="o", ls="", color=colour[m], label=m)
                    for m in models]
    if handles:
        axes[0].legend(handles=handles, frameon=False, fontsize=9, labelcolor=muted)
    title = "Does single-token availability predict the English detour?"
    if len(models) == 1:
        title += f"   —   {models[0]}"
    if len(tasks_seen) == 1:
        title += f"   ({tasks_seen[0]} only)"
    fig.suptitle(title, fontsize=13, color=ink)
    fig.text(0.5, -0.02, "Wendler et al. §6 predict a NEGATIVE slope · bars = 95% "
             f"bootstrap CI over words · excluded: copy accuracy < {MIN_ACC:.0%}, "
             f"or final P(target) < {MIN_TGT:.0%} (lens not reading the model)",
             ha="center", fontsize=9, color=muted)
    fig.tight_layout()
    path = results / "fig_tokenization_correlation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path.name


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    cells = load(args.results)
    if not cells:
        sys.exit(f"no per-model arrays in {args.results}.\n"
                 "Run tokenization_vs_detour.ipynb, run its emit cell, then:\n"
                 "  python extract_results.py tokenization_vs_detour.ipynb")

    summ = {k: summarise(v, args.boot) for k, v in cells.items()}
    used = report(cells, summ, )
    shared = shared_character_control(cells, summ, used, args.boot) if used else {}

    out = {f"{m}|{l}|{t}": {kk: vv for kk, vv in s.items() if kk != "mean_en"}
           for (m, l, t), s in summ.items()}
    for (m, l, t), c in cells.items():
        out[f"{m}|{l}|{t}"].update(single_token_frac=c["stf"], accuracy=c["acc"],
                                   n_words=c["n_words"], d_vocab=c["d_vocab"])
    for m, s in shared.items():
        out[f"{m}|shared_character_control"] = s
    for k, s in CONTRAST.items():
        out[f"{k}|translate_minus_repeat"] = s
    (args.results / "tokenization_analysis_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    written = ["tokenization_analysis_summary.json"]
    if used and not args.no_figures:
        written.insert(0, figure(cells, summ, used, args.results))
    print(f"\nwrote {', '.join(written)} -> {args.results}")

    if args.check:
        if not BASELINE:
            print("\n--check: BASELINE is empty. Once a run is trusted, paste its "
                  "values in so later runs are pinned against them.")
            return 0
        bad = [k for k, want in BASELINE.items()
               if abs(out[k[0]][k[1]] - want) >= 5e-4]
        print("\nbaseline check:", "PASS" if not bad else f"FAIL on {bad}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
