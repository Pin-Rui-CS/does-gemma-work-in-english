"""Pull results out of the notebooks into results/.

The kernel runs on a remote GPU, so nothing it writes with np.save() is reachable
from this machine -- that filesystem belongs to the container and dies with it.
Everything travels back inside the .ipynb instead, which is a local file:

  * figures   -- matplotlib embeds them as PNG in the cell outputs automatically
  * arrays    -- the save cell packs them into an npz, base64s it, and prints it

This script decodes both into results/. No network, no Kaggle filesystem, no git.

    python extract_results.py                  # both notebooks
    python extract_results.py <notebook.ipynb> # just one
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BEGIN, END = "#RESULTS_NPZ_B64#", "#END_RESULTS_NPZ_B64#"

# Cells that save a figure name it; that name is how the PNG gets a filename.
# fig_tokenization_correlation.png is deliberately absent: that figure is drawn
# locally by tokenization_analysis.py across every model, and pulling a
# single-session version out of a notebook would overwrite it with a partial one.
FIG_HINTS = ("fig_translation.png", "fig_control.png", "fig_timing.png",
             "fig_tokenization_session.png")


def cell_text(cell):
    s = cell.get("source", "")
    return s if isinstance(s, str) else "".join(s)


def stream_text(cell):
    out = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            t = o.get("text", "")
            out.append(t if isinstance(t, str) else "".join(t))
    return "".join(out)


def extract_arrays(nb, written):
    """Decode the base64 npz block emitted by the save cell."""
    for cell in nb.get("cells", []):
        text = stream_text(cell)
        if BEGIN not in text:
            continue
        lines = text.split("\n")
        start = next(i for i, l in enumerate(lines) if BEGIN in l)
        try:
            stop = next(i for i, l in enumerate(lines) if END in l)
        except StopIteration:
            print("  ! found the start marker but not the end -- output was truncated.")
            print("    Re-run the save cell; if it truncates again, split ARRAYS in two.")
            return
        b64 = "".join(l.strip() for l in lines[start + 1:stop])
        if not b64:
            print("  ! marker present but no payload -- was the cell run?")
            return

        with np.load(io.BytesIO(base64.b64decode(b64))) as z:
            for key in z.files:
                arr = z[key]
                if key.startswith("_meta_json"):              # scalars ride as bytes
                    # A run may tag its metadata: "_meta_json__gemma-2-2b" lands in
                    # summary_gemma-2-2b.json. Without that, a notebook run once per
                    # model would have each session overwrite the previous session's
                    # summary, while the .npy arrays -- which are already keyed by
                    # model -- accumulated correctly. Untagged keys keep the old name.
                    tag  = key[len("_meta_json"):].strip("_")
                    name = f"summary_{tag}.json" if tag else "summary.json"
                    meta = json.loads(bytes(arr).decode())
                    path = RESULTS / name
                    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
                    written.append((path.name, path.stat().st_size))
                    continue
                path = RESULTS / f"{key}.npy"
                np.save(path, arr)
                written.append((path.name, path.stat().st_size))
        return
    print("  (no array block found -- run the save cell at the bottom of the notebook)")


def extract_figures(nb, written):
    """Pull embedded PNGs out of plotting cells, named after the file they saved."""
    anon = 0
    for cell in nb.get("cells", []):
        text = cell_text(cell)
        name = next((h for h in FIG_HINTS if h in text), None)
        for out in cell.get("outputs", []):
            png = out.get("data", {}).get("image/png")
            if not png:
                continue
            if name is None:
                anon += 1
                name = f"figure_{anon}.png"
            data = base64.b64decode(png if isinstance(png, str) else "".join(png))
            path = RESULTS / name
            path.write_bytes(data)
            written.append((path.name, path.stat().st_size))
            name = None      # one figure per hint


def main(argv):
    targets = [Path(a) for a in argv[1:]] or sorted(ROOT.glob("*.ipynb"))
    RESULTS.mkdir(exist_ok=True)
    total = []
    for nb_path in targets:
        if not nb_path.exists():
            print(f"{nb_path}: not found")
            continue
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        print(f"\n{nb_path.name}")
        written = []
        extract_figures(nb, written)
        extract_arrays(nb, written)
        for n, size in written:
            print(f"   {n:36} {size/1024:9.1f} KB")
        if not written:
            print("   (nothing extracted)")
        total += written
    print(f"\n{len(total)} file(s) -> {RESULTS}")


if __name__ == "__main__":
    main(sys.argv)
