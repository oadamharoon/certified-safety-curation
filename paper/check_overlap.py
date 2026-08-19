"""Report overlapping text in a rendered matplotlib figure.

Eyeballing each render missed collisions repeatedly; this measures them.
"""
import sys, importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def overlaps(path, func="main"):
    spec = importlib.util.spec_from_file_location("figmod", path)
    m = importlib.util.module_from_spec(spec)
    sys.argv = [path]
    spec.loader.exec_module(m)
    getattr(m, func)()
    fig = plt.gcf()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for ax in fig.get_axes():
        for t in ax.texts + ([ax.title] if ax.get_title() else []):
            s = t.get_text().strip()
            if s:
                items.append((s, t.get_window_extent(renderer=r)))
        for lbl in (ax.xaxis.label, ax.yaxis.label):
            if lbl.get_text().strip():
                items.append((lbl.get_text(), lbl.get_window_extent(renderer=r)))
        lg = ax.get_legend()
        if lg:
            for t in lg.get_texts():
                items.append((t.get_text(), t.get_window_extent(renderer=r)))
    bad = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > 1.5 and oy > 1.5:
                bad.append((items[i][0][:34], items[j][0][:34], round(ox), round(oy)))
    # text running outside its own axes is invisible to the pair test above
    clipped = []
    for ax in fig.get_axes():
        ab = ax.get_window_extent(renderer=r)
        for t in ax.texts + ([ax.title] if ax.get_title() else []):
            if not t.get_text().strip():
                continue
            bb = t.get_window_extent(renderer=r)
            if bb.x0 < ab.x0 - 2 or bb.x1 > ab.x1 + 2:
                clipped.append((t.get_text()[:34].replace("\n", " "),
                                round(ab.x0 - bb.x0), round(bb.x1 - ab.x1)))
    return bad, len(items), clipped


if __name__ == "__main__":
    bad, n, clipped = overlaps(sys.argv[1])
    print(f"  {n} text items, {len(bad)} overlapping pairs, {len(clipped)} clipped")
    for a, b, ox, oy in bad:
        print(f"    overlap: {a!r} x {b!r}  ({ox}x{oy} px)")
    for t, l, rr in clipped:
        print(f"    clipped: {t!r}  (left {l}px, right {rr}px beyond axes)")
