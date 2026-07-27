"""Every number and every figure in the write-up, computed from results/*.npy.

The sweep needs a GPU. Nothing after it does. Splitting them matters for a
practical reason: a Kaggle session is ephemeral and rate-limited, so any statistic
that only exists as a notebook cell can only be recomputed by spending another
one. That is how the timing analysis -- the headline of Phase 1 -- came to be
reported in the README with no code behind it and a figure no cell produces.

So the division is: the notebook produces arrays, this produces conclusions.

    python analysis.py            # numbers -> stdout + results/analysis_summary.json
                                  # figures -> results/fig_*.png
    python analysis.py --check    # additionally assert the recorded baseline

Every statistic here resamples WORDS, not prompts. The three demonstration seeds
per word are three measurements of one word, not three independent draws, so a
CI that resamples the 162 rows would be roughly sqrt(3) too narrow. And the
resample is PAIRED -- one draw of word indices, applied to both conditions --
because the two conditions are measured on the same words and the comparison is
within-word.

--check asserts BASELINE, the numbers currently reported in the README. It exists
to isolate one variable: if a re-run moves a number, this analysis did not move,
so something upstream did. It earned that keep immediately -- the baseline was
first recorded from the broken lens, so the gap between SUPERSEDED and BASELINE
below is a *measurement* of what the bug cost rather than an assertion about it.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

# Categorical slots 1-3 of the reference palette, validated all-pairs on a white
# surface (worst CVD dE 9.2, worst normal-vision dE 24.0). Aqua sits at 2.82:1
# against white, below the 3:1 bar, so the language plots carry direct labels on
# the final value and every number is also printed and written to JSON.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

LANG_STYLE = [("Japanese (source)", AQUA), ("English (pivot?)", BLUE),
              ("German (target)", ORANGE)]
EN = 1                                      # language axis: 0 = ja, 1 = en, 2 = de

# gemma-2-2b, n=54 words x 3 seeds, corrected logit lens (re-run 26 Jul).
# These are the numbers the README reports. See the module docstring.
BASELINE = {
    "peak_translation": 0.5907, "peak_control": 0.6997,
    "peak_pos_translation": 23, "peak_pos_control": 24,
    "delta_peak": -0.1090, "delta_peak_ci": [-0.1494, -0.0563],
    "delta_at_20": 0.4266, "delta_at_20_ci": [0.3555, 0.5017],
    "delta_auc": 1.4583, "delta_auc_ci": [1.2184, 1.7109],
}

# The same statistics under the broken lens, which normalised every layer by the
# FINAL layer's scale instead of its own. Kept because the before/after is itself
# evidence -- it puts a number on what the bug cost, and the pattern is diagnostic:
# the peak positions and the final row do not move at all, while everything that
# depends on a mid-stack magnitude does. That is the signature of a normalisation
# error confined to intermediate layers, and it is what the fix predicted.
#
# One caveat on the AUC interval: the pre-fix figure quoted in earlier drafts was
# [+0.998, +1.406], computed ad hoc with an unrecorded seed. Re-deriving it here
# at seed 0 gives the value below. The difference is Monte-Carlo noise across
# seeds at 2000 replicates, not a disagreement about the statistic -- which is the
# whole reason the analysis now lives in a file with a fixed seed.
SUPERSEDED = {
    "peak_translation": 0.6170, "peak_control": 0.6627,
    "peak_pos_translation": 23, "peak_pos_control": 24,
    "delta_peak": -0.0457, "delta_peak_ci": [-0.0975, 0.0041],
    "delta_at_20": 0.3636, "delta_at_20_ci": [0.3078, 0.4224],
    "delta_auc": 1.2000, "delta_auc_ci": [1.0059, 1.3968],
}


def load(results):
    """Per-word mean curves for both conditions, plus the shared word index."""
    need = ["curves_translation", "widx_translation", "curves_control", "widx_control"]
    missing = [n for n in need if not (results / f"{n}.npy").exists()]
    if missing:
        sys.exit(f"missing {', '.join(n + '.npy' for n in missing)} in {results}\n"
                 "Run the sweep + emit cells in mechanistic_interpretability.ipynb, "
                 "then extract_results.py.")
    a = {n: np.load(results / f"{n}.npy") for n in need}

    # The pairing is the whole basis of the comparison, so check it rather than
    # assume it: both sweeps must have visited the same words in the same order.
    if not np.array_equal(a["widx_translation"], a["widx_control"]):
        sys.exit("the two conditions did not sweep the same words in the same order; "
                 "a paired comparison is not valid on these arrays.")

    w = a["widx_translation"]
    words = np.unique(w)
    per_word = lambda c: np.stack([c[w == u].mean(0) for u in words])
    return per_word(a["curves_translation"]), per_word(a["curves_control"])


def bootstrap(P_tr, P_ct, n_boot, seed=0):
    """Paired resample over words. Returns [n_boot, n_positions] English deltas.

    One draw of word indices per replicate, used for BOTH conditions -- resampling
    them independently would discard the pairing and inflate every interval.
    """
    n = len(P_tr)
    rng = np.random.default_rng(seed)
    # Drawn one replicate at a time to match the notebook's own bootstrap cell
    # exactly, so the confirmatory CI is reproducible to the last decimal.
    picks = [rng.integers(0, n, n) for _ in range(n_boot)]
    return np.stack([P_tr[p].mean(0)[:, EN] - P_ct[p].mean(0)[:, EN] for p in picks]), picks


def peak_delta_ci(P_tr, P_ct, picks):
    """CI for the difference of peaks -- not the peak of the difference.

    max() is applied inside each replicate, so the statistic bootstrapped is the
    same one that was pre-specified, including its freedom to peak at different
    positions in the two conditions.
    """
    d = np.array([P_tr[p].mean(0)[:, EN].max() - P_ct[p].mean(0)[:, EN].max() for p in picks])
    return np.percentile(d, [2.5, 97.5])


def analyse(P_tr, P_ct, n_boot):
    m_tr, m_ct = P_tr.mean(0), P_ct.mean(0)
    n, n_pos = len(P_tr), P_tr.shape[1]

    # Normal-approximation band for the curves themselves. The bootstrap below is
    # the inferential statistic; this is what the shaded band in the plots shows.
    se = lambda P: 1.96 * P.std(0, ddof=1) / np.sqrt(n)
    ci_tr, ci_ct = se(P_tr), se(P_ct)

    B, picks = bootstrap(P_tr, P_ct, n_boot)
    lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
    auc_lo, auc_hi = np.percentile(B.sum(1), [2.5, 97.5])

    i_tr, i_ct = int(m_tr[:, EN].argmax()), int(m_ct[:, EN].argmax())
    obs = m_tr[:, EN] - m_ct[:, EN]
    j = int(np.abs(obs).argmax())

    return dict(
        n_words=n, n_positions=n_pos, n_boot=n_boot,
        peak_translation=float(m_tr[i_tr, EN]), peak_translation_pm=float(ci_tr[i_tr, EN]),
        peak_control=float(m_ct[i_ct, EN]), peak_control_pm=float(ci_ct[i_ct, EN]),
        peak_pos_translation=i_tr, peak_pos_control=i_ct,
        delta_peak=float(m_tr[:, EN].max() - m_ct[:, EN].max()),
        delta_peak_ci=[float(v) for v in peak_delta_ci(P_tr, P_ct, picks)],
        delta_by_position=[float(v) for v in obs],
        delta_ci_lo=[float(v) for v in lo], delta_ci_hi=[float(v) for v in hi],
        delta_largest=float(obs[j]), delta_largest_pos=j,
        delta_at_20=float(obs[20]) if n_pos > 20 else None,
        delta_at_20_ci=[float(lo[20]), float(hi[20])] if n_pos > 20 else None,
        delta_auc=float(obs.sum()), delta_auc_ci=[float(auc_lo), float(auc_hi)],
        # Lens-independent sanity: the final row is identical under either
        # normalisation convention, so these survive the lens correction and are
        # the one place the behavioural result is visible inside the measurement.
        final_translation=[float(v) for v in m_tr[-1]],
        final_control=[float(v) for v in m_ct[-1]],
        _mean=dict(translation=m_tr, control=m_ct, ci_translation=ci_tr, ci_control=ci_ct,
                   lo=lo, hi=hi),
    )


def report(r):
    sep = lambda ci: "separates" if (ci[0] > 0 or ci[1] < 0) else "contains zero"
    n_pos = r["n_positions"]

    print(f"\nn = {r['n_words']} words x 3 seeds, {n_pos} residual positions "
          f"(0 = embedding, {n_pos - 1} = final), {r['n_boot']} bootstrap replicates\n")

    print(f"{'condition':<22}{'peak P(EN)':>12}{'+-95% CI':>11}{'position':>10}")
    for name, k in (("JA->DE translation", "translation"), ("JA->JA repetition", "control")):
        print(f"{name:<22}{r['peak_' + k]:>12.4f}{r['peak_' + k + '_pm']:>11.4f}"
              f"{r['peak_pos_' + k]:>10d}")

    ci = r["delta_peak_ci"]
    print(f"\nCONFIRMATORY -- pre-specified before the sweep")
    print(f"  delta peak P(EN), translation - repetition = {r['delta_peak']:+.4f}")
    print(f"  95% paired bootstrap CI over words: [{ci[0]:+.4f}, {ci[1]:+.4f}]  -> {sep(ci)}")

    print(f"\nEXPLORATORY -- statistic chosen after seeing the curves differ in shape;"
          f"\n{' ' * 14}{n_pos} positions tested with no multiple-comparison correction")
    d, lo, hi = r["delta_by_position"], r["delta_ci_lo"], r["delta_ci_hi"]
    lead = max(0, r["delta_largest_pos"] - 4)
    print(f"\n  {'position':<10}" + "".join(f"{p:>9d}" for p in range(lead, n_pos)))
    print(f"  {'translation':<10}" + "".join(f"{v:>9.3f}" for v in r['_mean']['translation'][lead:, EN]))
    print(f"  {'repetition':<10}" + "".join(f"{v:>9.3f}" for v in r['_mean']['control'][lead:, EN]))
    print(f"  {'delta':<10}" + "".join(f"{v:>+9.3f}" for v in d[lead:]))
    print(f"  {'CI excl 0':<10}" + "".join(f"{'yes' if (lo[p] > 0 or hi[p] < 0) else '-':>9}"
                                           for p in range(lead, n_pos)))

    j = r["delta_largest_pos"]
    print(f"\n  largest separation at position {j}: {d[j]:+.4f}, "
          f"CI [{lo[j]:+.4f}, {hi[j]:+.4f}]")
    a = r["delta_auc_ci"]
    print(f"  area under the English curve:   {r['delta_auc']:+.4f}, "
          f"CI [{a[0]:+.4f}, {a[1]:+.4f}]  -> {sep(a)}")

    # Guard against reading significance as importance. A CI can exclude zero at
    # |delta| ~ 1e-3 -- statistically clean, substantively nothing -- so the count
    # is reported rather than the bare "CI excludes 0" ticks above. Below ~1e-4
    # both curves are flat at zero and the interval is measuring bf16 rounding.
    trivial = [p for p in range(n_pos)
               if (lo[p] > 0 or hi[p] < 0) and 1e-4 <= abs(d[p]) < 0.01]
    noise = sum(1 for p in range(n_pos)
                if (lo[p] > 0 or hi[p] < 0) and abs(d[p]) < 1e-4)
    if trivial:
        j2 = max(trivial, key=lambda p: abs(d[p]))
        print(f"  note: {len(trivial)} more position(s) exclude zero at |delta| < 0.01 "
              f"(largest, position {j2}: {d[j2]:+.4f}) -- resolvable, not meaningful")
    if noise:
        print(f"        and {noise} at |delta| < 1e-4, where both curves are flat "
              "at zero -- that is rounding, not a measurement")

    print(f"\nFINAL ROW (unaffected by the lens normalisation; ja / en / de)")
    print(f"  translation {np.round(r['final_translation'], 4)}"
          f"   repetition {np.round(r['final_control'], 4)}")


def figures(r, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def chrome(ax):
        ax.grid(True, color=GRID, lw=0.8, alpha=1.0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
            ax.spines[s].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)

    M, n_pos = r["_mean"], r["n_positions"]
    x = np.arange(n_pos)

    # --- one panel per condition: all three languages across depth --------------
    for key, title, fname in (("translation", "JA→DE translation", "fig_translation.png"),
                              ("control", "JA→JA repetition control", "fig_control.png")):
        mean, ci = M[key], M["ci_" + key]
        fig, ax = plt.subplots(figsize=(9, 5))
        for k, (label, color) in enumerate(LANG_STYLE):
            ax.plot(x, mean[:, k], color=color, lw=2, marker="o", ms=4, label=label)
            ax.fill_between(x, mean[:, k] - ci[:, k], mean[:, k] + ci[:, k],
                            color=color, alpha=0.18, lw=0)
            # Direct label on the final value -- the relief the palette's contrast
            # WARN requires, and the number a reader most wants off this plot.
            if mean[-1, k] > 0.02:
                ax.annotate(f"{mean[-1, k]:.2f}", (x[-1], mean[-1, k]),
                            textcoords="offset points", xytext=(8, 0),
                            fontsize=9, color=INK_2, va="center")
        chrome(ax)
        ax.set_xticks(np.arange(0, n_pos, 2))
        ax.set_xlabel(f"residual stream position  (0 = embedding, {n_pos - 1} = final)",
                      fontsize=10, color=INK_2)
        ax.set_ylabel("mean probability", fontsize=10, color=INK_2)
        ax.set_title(title, fontsize=13, color=INK, pad=12)
        ax.legend(frameon=False, fontsize=10, labelcolor=INK_2)
        fig.text(0.5, -0.02, f"n={r['n_words']} words, 3 seeds each · bands = 95% CI over words",
                 ha="center", fontsize=9, color=MUTED)
        fig.savefig(results / fname, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # --- the timing figure: same peak, different dwell -------------------------
    # Two stacked panels sharing an x-axis rather than two y-scales on one plot:
    # probability and a difference of probabilities are not the same quantity, and
    # overlaying them would invent a visual correspondence the data does not have.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                   gridspec_kw=dict(height_ratios=[2, 1], hspace=0.12))

    # Peak labels are placed away from the curve's own rising side, so the text
    # never lands on the line it annotates. Which side that is depends on where
    # the peak falls, so it is computed rather than hardcoded.
    peaks = {k: int(M[k][:, EN].argmax()) for k in ("translation", "control")}
    left_first = peaks["translation"] <= peaks["control"]

    for key, label, color, marker in (("translation", "JA→DE translation", BLUE, "o"),
                                      ("control", "JA→JA repetition", ORANGE, "s")):
        mean, ci = M[key][:, EN], M["ci_" + key][:, EN]
        ax1.plot(x, mean, color=color, lw=2, marker=marker, ms=4, label=label)
        ax1.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.18, lw=0)
        i = peaks[key]
        to_left = (key == "translation") == left_first
        ax1.annotate(f"peak {mean[i]:.3f}", (i, mean[i]), textcoords="offset points",
                     xytext=(-10 if to_left else 10, 10), fontsize=10, color=INK_2,
                     ha="right" if to_left else "left")
    # Headroom for those labels; without it the higher peak's text is clipped.
    ax1.set_ylim(top=max(M[k][:, EN].max() for k in peaks) * 1.22)
    chrome(ax1)
    ax1.set_ylabel("P(English)", fontsize=10, color=INK_2)
    ax1.legend(frameon=False, fontsize=10, loc="upper left", labelcolor=INK_2)
    ax1.set_title("English peaks at the same height in both conditions "
                  "— but not at the same time", fontsize=13, color=INK, pad=14)

    d = np.asarray(r["delta_by_position"])
    ax2.axhline(0, color=AXIS, lw=1)
    ax2.fill_between(x, r["delta_ci_lo"], r["delta_ci_hi"], color=INK, alpha=0.12, lw=0)
    ax2.plot(x, d, color=INK, lw=2, marker="o", ms=4)
    # Both annotations sit in the right half of the axis, where the curve lives, so
    # they are anchored right-aligned and pushed away from the line rather than
    # trailing off the edge.
    j = r["delta_largest_pos"]
    ax2.annotate(f"{'translation' if d[j] > 0 else 'repetition'} leads by "
                 f"{abs(d[j]):.2f} at position {j}", (j, d[j]),
                 textcoords="offset points", xytext=(-12, 14), ha="right",
                 fontsize=10, color=INK_2)
    k = int(d.argmin())
    if d[k] < -0.05 and k != j:
        ax2.annotate(f"repetition leads at {k}", (k, d[k]), textcoords="offset points",
                     xytext=(10, -2), ha="left", va="top", fontsize=10, color=INK_2)
    ax2.margins(y=0.28)
    chrome(ax2)
    ax2.set_xticks(np.arange(0, n_pos, 2))
    ax2.set_ylabel("Δ P(English)\ntranslation − repetition", fontsize=10, color=INK_2)
    ax2.set_xlabel(f"residual stream position  (0 = embedding, {n_pos - 1} = final)",
                   fontsize=10, color=INK_2)

    fig.text(0.5, 0.045, f"n={r['n_words']} words, 3 seeds each · bands = 95% bootstrap CI "
             "resampling words (paired across conditions)",
             ha="center", fontsize=9, color=MUTED)
    fig.savefig(results / "fig_timing.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return ["fig_translation.png", "fig_control.png", "fig_timing.png"]


def fmt(v):
    if isinstance(v, list):
        return "[" + ", ".join(f"{x:+.4f}" for x in v) + "]"
    return f"{v:d}" if isinstance(v, int) else f"{v:+.4f}"


def check(r):
    """Assert BASELINE, alongside what the same statistic read pre-lens-fix."""
    bad = []
    for k, want in BASELINE.items():
        got = r[k]
        ok = (all(abs(a - b) < 5e-4 for a, b in zip(got, want))
              if isinstance(want, list) else abs(got - want) < 5e-4)
        print(f"  {'ok  ' if ok else 'FAIL'}  {k:<22} {fmt(got):<20}"
              f"expected {fmt(want):<20}broken lens read {fmt(SUPERSEDED[k])}")
        if not ok:
            bad.append(k)
    if bad:
        print(f"\n{len(bad)} value(s) differ from the recorded baseline.\n"
              "The arrays in results/ are not the ones these numbers came from. Either\n"
              "the sweep was re-run -- in which case record the new values and update\n"
              "BASELINE, keeping the old ones in SUPERSEDED -- or this script changed,\n"
              "and that is a bug.")
        return 1
    print("\nreproduces the recorded baseline exactly.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--boot", type=int, default=2000,
                    help="bootstrap replicates (default matches the notebook's own cell)")
    ap.add_argument("--check", action="store_true", help="assert the recorded baseline")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    P_tr, P_ct = load(args.results)
    r = analyse(P_tr, P_ct, args.boot)
    report(r)

    written = [] if args.no_figures else figures(r, args.results)
    summary = {k: v for k, v in r.items() if not k.startswith("_")}
    (args.results / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("\nwrote " + ", ".join(written + ["analysis_summary.json"]) + f" -> {args.results}")

    if args.check:
        print("\nbaseline check (last column: the same statistic under the broken lens):")
        return check(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
