"""Generate docs/preprocessing_pipeline.svg, the animated figure of the README.

Usage:
    python tools/build_pipeline_svg.py [OUTPUT]

Needs only the standard library. One synthetic MALDI-TOF spectrum (fixed seed, so
the output is reproducible) walks through the pipeline in a loop: raw, smoothed,
baseline removed, peaks detected, binned, AMR prediction. The spectrum and the
predictions are illustrative, they are not computed by MaldiAMRKit.

The animation is pure SMIL: GitHub shows README images through <img>, where
JavaScript and external files do not run. For the same reason colours are
mid-tones readable on both themes: inside <img>, prefers-color-scheme follows
the operating system, not the GitHub theme.
"""

import math
import random
import sys
from pathlib import Path

DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent / "docs" / "preprocessing_pipeline.svg"
)

D = 12.0  # loop duration (s)
W, H = 800, 320
X0, X1 = 64.0, 756.0  # plot x-range (px)
YTOP, Y0 = 64.0, 230.0  # top of plot / zero-intensity axis (px)
MZ0, MZ1 = 2000.0, 20000.0
YMAX = 1.2
N = 401  # samples per trace
STEP_MZ = (MZ1 - MZ0) / (N - 1)


def xpx(mz):
    """Convert m/z to a horizontal pixel position."""
    return X0 + (mz - MZ0) / (MZ1 - MZ0) * (X1 - X0)


def ypx(v):
    """Convert an intensity to a vertical pixel position."""
    return Y0 - v / YMAX * (Y0 - YTOP)


def f1(v):
    """Format a number with at most one decimal."""
    s = f"{v:.1f}"
    if s.endswith(".0"):
        s = s[:-2]
    return "0" if s in ("-0", "") else s


def f2(v):
    """Format a number with at most two decimals."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


# (nominal m/z, height, sigma in Da); bacterial-like, dense below 12 kDa
PEAKS = [
    (2560, 0.26, 55),
    (3130, 0.42, 58),
    (4365, 0.64, 62),
    (5096, 0.48, 66),
    (6255, 0.92, 70),
    (7158, 0.36, 76),
    (7705, 0.28, 80),
    (9060, 0.56, 86),
    (9535, 0.24, 90),
    (10300, 0.42, 96),
    (11780, 0.22, 106),
    (13100, 0.15, 116),
    (14200, 0.07, 124),
    (15400, 0.10, 134),
    (16900, 0.05, 144),
    (18300, 0.04, 150),
]
MAIN_PEAKS = [p for p in PEAKS if p[1] >= 0.2]
LABELLED = {4365, 6255, 9060}

mz = [MZ0 + i * STEP_MZ for i in range(N)]


def snap(m):
    """Round m/z to the nearest sample of the trace."""
    return MZ0 + round((m - MZ0) / STEP_MZ) * STEP_MZ


signal = [0.0] * N
for m0, h, s in PEAKS:
    c = snap(m0)
    for i, m in enumerate(mz):
        signal[i] += h * math.exp(-0.5 * ((m - c) / s) ** 2)


def baseline(m):
    """Return the drifting baseline at the given m/z."""
    return 0.045 + 0.24 * math.exp(-(m - MZ0) / 3800.0)


rng = random.Random(20240611)
raw = []
for i, m in enumerate(mz):
    noise = rng.gauss(0.0, 0.020 + 0.045 * math.sqrt(signal[i]))
    raw.append(signal[i] + baseline(m) + noise)

smooth_b = [signal[i] + baseline(m) for i, m in enumerate(mz)]
smooth_f = signal[:]


def _num(t):
    """Format an integer count of tenths compactly (12 -> 1.2, -5 -> -.5)."""
    s = f"{t / 10:.1f}"
    if s.endswith(".0"):
        s = s[:-2]
    s = s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s
    return "0" if s in ("-0", "") else s


def poly(xs, ys):
    """Polyline as 'M x,y l dx,dy ...' with drift-free 0.1 px rounding."""
    ax = [round(x * 10) for x in xs]
    ay = [round(y * 10) for y in ys]
    out = [f"M{_num(ax[0])},{_num(ay[0])}l"]
    out.append(
        " ".join(
            f"{_num(ax[i] - ax[i - 1])},{_num(ay[i] - ay[i - 1])}"
            for i in range(1, len(ax))
        )
    )
    return "".join(out)


def rdp(xs, ys, eps):
    """Return the indices kept by Ramer-Douglas-Peucker simplification."""
    keep = {0, len(xs) - 1}
    stack = [(0, len(xs) - 1)]
    while stack:
        a, b = stack.pop()
        dx, dy = xs[b] - xs[a], ys[b] - ys[a]
        norm = math.hypot(dx, dy) or 1.0
        best, idx = -1.0, None
        for i in range(a + 1, b):
            d = abs(dy * (xs[i] - xs[a]) - dx * (ys[i] - ys[a])) / norm
            if d > best:
                best, idx = d, i
        if idx is not None and best > eps:
            keep.add(idx)
            stack += [(a, idx), (idx, b)]
    return keep


XS = [xpx(m) for m in mz]
YB = [ypx(v) for v in smooth_b]
YF = [ypx(v) for v in smooth_f]
# shared index set so both morph keyframes have identical command structure
KEEP = sorted(rdp(XS, YB, 0.2) | rdp(XS, YF, 0.2))
D_RAW = poly(XS, [ypx(v) for v in raw])
D_SB = poly([XS[i] for i in KEEP], [YB[i] for i in KEEP])
D_SF = poly([XS[i] for i in KEEP], [YF[i] for i in KEEP])

BL_MZ = [MZ0 + i * (MZ1 - MZ0) / 36 for i in range(37)]
D_BL = poly([xpx(m) for m in BL_MZ], [ypx(baseline(m)) for m in BL_MZ])
D_BL0 = poly([xpx(m) for m in BL_MZ], [Y0 for _ in BL_MZ])

# bins (60 x 300 Da) -> bar heights = max corrected intensity in the bin
NB = 60
BW_MZ = (MZ1 - MZ0) / NB
bins = []
for b in range(NB):
    lo, hi = MZ0 + b * BW_MZ, MZ0 + (b + 1) * BW_MZ
    vals = [
        smooth_f[i]
        for i, m in enumerate(mz)
        if lo <= m < hi or (b == NB - 1 and m == hi)
    ]
    bins.append(max(max(vals), 0.012))
BMAX = max(bins)

EASE = "0.42 0 0.58 1"
LIN = "0 0 1 1"


def _val(v):
    """Format a keyframe value."""
    if isinstance(v, (tuple, list)):
        return " ".join(f2(x) for x in v)
    if isinstance(v, (int, float)):
        return f2(v)
    return v


def _kt(t):
    """Convert a time in seconds to a SMIL keyTimes value."""
    if t <= 0:
        return "0"
    if t >= D:
        return "1"
    return f"{t / D:.4f}".rstrip("0").rstrip(".")


def anim(attr, kfs, tag="animate", extra=""):
    """Return a SMIL animation from absolute (time, value) keyframes."""
    kfs = list(kfs)
    if kfs[0][0] > 0:
        kfs.insert(0, (0.0, kfs[0][1]))
    if kfs[-1][0] < D:
        kfs.append((D, kfs[-1][1]))
    for (ta, _), (tb, _) in zip(kfs, kfs[1:], strict=False):
        assert tb > ta, (attr, kfs)
    vals = [_val(v) for _, v in kfs]
    splines = [LIN if a == b else EASE for a, b in zip(vals, vals[1:], strict=False)]
    extra = (" " + extra) if extra else ""
    return (
        f'<{tag} attributeName="{attr}"{extra} dur="{f2(D)}s" repeatCount="indefinite" '
        f'calcMode="spline" keyTimes="{";".join(_kt(t) for t, _ in kfs)}" '
        f'keySplines="{";".join(splines)}" values="{";".join(vals)}"/>'
    )


def op(kfs):
    """Return an opacity animation."""
    return anim("opacity", kfs)


def tr_translate(kfs):
    """Return a translate animation."""
    return anim("transform", kfs, tag="animateTransform", extra='type="translate"')


def tr_scale(kfs):
    """Return a scale animation."""
    return anim("transform", kfs, tag="animateTransform", extra='type="scale"')


# stage switch times (caption / stepper): 1->2, 2->3, 3->4, 4->5, 5->6, 6->1
SW = [1.75, 3.30, 5.25, 7.20, 9.05, 11.70]
STAGES = [
    ("Raw", "Raw spectrum", "noisy intensities on a drifting baseline"),
    ("Smooth", "Smoothing", "Savitzky-Golay filter suppresses high-frequency noise"),
    (
        "Baseline",
        "Baseline removal",
        "SNIP baseline subtracted; the trace settles on zero",
    ),
    ("Peaks", "Peak detection", "local maxima flagged along the m/z axis"),
    ("Bin", "Binning", "intensities pooled into fixed-width m/z bins → feature vector"),
    (
        "Predict",
        "AMR prediction",
        "an ML classifier maps the feature vector to Resistant (R) / Susceptible (S)",
    ),
]
FADE = 0.18


def stage_opacity(k):
    """Return opacity keyframes of the stage-k text (crossfade at SW)."""
    if k == 0:
        return [(0, 1), (SW[0] - FADE, 1), (SW[0], 0), (SW[5], 0), (SW[5] + FADE, 1)]
    a, b = SW[k - 1], SW[k]
    return [(0, 0), (a, 0), (a + FADE, 1), (b - FADE, 1), (b, 0)]


def build() -> str:
    """Return the SVG document as a string."""
    o = []
    add = o.append

    add('<?xml version="1.0" encoding="UTF-8"?>')
    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        'role="img" aria-labelledby="t d">'
    )
    add('<title id="t">MaldiAMRKit preprocessing pipeline</title>')
    add(
        '<desc id="d">Animated loop of one MALDI-TOF mass spectrum (2,000 to 20,000 m/z) moving '
        "through the MaldiAMRKit pipeline: a noisy raw spectrum on a sloped baseline is smoothed "
        "(Savitzky-Golay), its SNIP baseline is subtracted so the trace drops onto zero, the main "
        "peaks are marked, the intensities are binned into a feature vector, and a classifier "
        "outputs antimicrobial-resistance phenotypes: Ciprofloxacin R (resistant), Ceftriaxone R "
        "(resistant), Meropenem S (susceptible).</desc>"
    )
    # Defaults are mid-tone colours legible on BOTH white and #0d1117 (inside <img>,
    # prefers-color-scheme follows the OS, not GitHub's theme). Contrast white/dark:
    # fg #6e7781 4.6/4.2, mu #7d8590 3.7/5.1, accent #1a7fc1 4.3/4.4, peak #c98500
    # 3.1/6.2, R #D55E00 3.9/4.9, S #009E73 3.4/5.5. The dark override is only a
    # nudge that still keeps >= 3:1 on white (fg #8b949e 3.1, accent #3b8fcf 3.5).
    add(
        "<style>"
        'text{font-family:system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}'
        ".fg{fill:#6e7781}.mu{fill:#7d8590}.ax{stroke:#7d8590}.bl{stroke:#7d8590}"
        ".tr{stroke:#1a7fc1}.ac{fill:#1a7fc1}.fgs{stroke:#6e7781}.pk{fill:#c98500}"
        ".pill{fill:#1a7fc1;fill-opacity:.12;stroke:#1a7fc1;stroke-opacity:.7}"
        ".card{fill:#1a7fc1;fill-opacity:.06;stroke:#1a7fc1;stroke-opacity:.5}"
        "@media (prefers-color-scheme:dark){"
        ".fg{fill:#8b949e}.fgs{stroke:#8b949e}.tr{stroke:#3b8fcf}.ac{fill:#3b8fcf}"
        ".pill{fill:#3b8fcf;stroke:#3b8fcf}.card{fill:#3b8fcf;stroke:#3b8fcf}"
        "}"
        "</style>"
    )

    STRIP_Y, STRIP_H = 256.0, 14.0
    add(
        '<defs><clipPath id="mak-strip-clip">'
        f'<rect x="{f1(X0)}" y="{f1(STRIP_Y - 4)}" width="0" height="{f1(STRIP_H + 8)}">'
        + anim("width", [(0, 0), (7.35, 0), (8.0, X1 - X0), (11.8, X1 - X0), (11.9, 0)])
        + "</rect></clipPath></defs>"
    )

    STEP_W = (X1 - X0) / 6
    CX = [X0 + (k + 0.5) * STEP_W for k in range(6)]
    PILL_W, PILL_H, PILL_CY = 100.0, 26.0, 31.0
    pk = [(0, 0)]
    moves = [(1.50, 2.00), (3.10, 3.55), (5.05, 5.50), (7.00, 7.45), (8.85, 9.30)]
    for k, (a, b) in enumerate(moves):
        pk += [(a, CX[k] - CX[0]), (b, CX[k + 1] - CX[0])]
    pk += [(11.45, CX[5] - CX[0]), (11.95, 0)]
    add("<g>")
    add(
        f'<rect class="pill" x="{f1(CX[0] - PILL_W / 2)}" y="{f1(PILL_CY - PILL_H / 2)}" '
        f'width="{f1(PILL_W)}" height="{f1(PILL_H)}" rx="{f1(PILL_H / 2)}">'
        + tr_translate([(t, (dx, 0)) for t, dx in pk])
        + "</rect>"
    )
    for k in range(1, 6):
        add(
            f'<text class="mu" x="{f1(X0 + k * STEP_W)}" y="{f1(PILL_CY + 4.5)}" font-size="13" '
            'text-anchor="middle" opacity=".55">›</text>'
        )
    for k, (short, _, _) in enumerate(STAGES):
        on = stage_opacity(k)
        off = [(t, 1 - v) for t, v in on]
        lab = (
            f'<tspan font-size="10.5" opacity=".7">{k + 1}</tspan>'
            f'<tspan dx="5">{short}</tspan>'
        )
        common = (
            f'x="{f1(CX[k])}" y="{f1(PILL_CY + 4.5)}" font-size="12.5" font-weight="600" '
            'text-anchor="middle"'
        )
        add(
            f'<text class="mu" {common} opacity="{f2(off[0][1])}">{lab}{op(off)}</text>'
        )
        add(f'<text class="ac" {common} opacity="{f2(on[0][1])}">{lab}{op(on)}</text>')
    add("</g>")

    add('<g fill="none" stroke-width="1">')
    add(f'<path class="ax" d="M{f1(X0)},{f1(YTOP - 4)}V{f1(Y0)}H{f1(X1)}"/>')
    for t in (2000, 5000, 10000, 15000, 20000):
        add(f'<path class="ax" d="M{f1(xpx(t))},{f1(Y0)}v4"/>')
    add("</g>")
    add('<g font-size="11" class="mu" text-anchor="middle">')
    for t in (2000, 5000, 10000, 15000, 20000):
        add(f'<text x="{f1(xpx(t))}" y="{f1(Y0 + 16)}">{t:,}</text>')
    add("</g>")
    add(
        f'<text class="mu" x="{f1(X1 + 6)}" y="{f1(Y0 + 4)}" font-size="11.5" '
        'font-style="italic">m/z</text>'
    )
    add(
        f'<text class="mu" font-size="11" text-anchor="middle" '
        f'transform="translate({f1(X0 - 16)},{f1((YTOP + Y0) / 2)}) rotate(-90)">'
        "intensity (a.u.)</text>"
    )

    T_DROP_A, T_DROP_B = 3.45, 4.05
    add(
        f'<path class="bl" d="{D_BL}" fill="none" stroke-width="1.4" stroke-dasharray="5 4" '
        'opacity="0">'
        + op([(0, 0), (3.15, 0), (3.40, 1), (4.0, 1), (4.35, 0)])
        + anim(
            "d",
            [
                (0, D_BL),
                (T_DROP_A, D_BL),
                (T_DROP_B, D_BL0),
                (11.75, D_BL0),
                (11.85, D_BL),
            ],
        )
        + "</path>"
    )

    bw_px = (X1 - X0) / NB
    add(f'<g transform="translate(0,{f1(Y0)})">')
    add(
        '<g class="ac" opacity="0" transform="scale(1,0.001)">'
        + op(
            [
                (0, 0),
                (7.0, 0),
                (7.1, 0.9),
                (9.0, 0.9),
                (9.4, 0.6),
                (11.4, 0.6),
                (11.75, 0),
            ]
        )
        + tr_scale(
            [
                (0, (1, 0.001)),
                (7.05, (1, 0.001)),
                (7.75, (1, 1)),
                (11.8, (1, 1)),
                (11.9, (1, 0.001)),
            ]
        )
    )
    for b, v in enumerate(bins):
        hpx = v / YMAX * (Y0 - YTOP)
        add(
            f'<rect x="{f1(X0 + b * bw_px + 1.2)}" y="{f1(-hpx)}" width="{f1(bw_px - 2.4)}" '
            f'height="{f1(hpx)}" rx="1"/>'
        )
    add("</g></g>")

    add(
        f'<path class="tr" d="{D_RAW}" fill="none" stroke-width="1.1" stroke-linejoin="round">'
        + op([(0, 1), (1.5, 1), (2.0, 0), (11.6, 0), (12.0, 1)])
        + "</path>"
    )

    add(
        f'<path class="tr" d="{D_SB}" fill="none" stroke-width="1.7" stroke-linejoin="round" '
        'stroke-linecap="round" opacity="0">'
        + op(
            [
                (0, 0),
                (1.5, 0),
                (2.0, 1),
                (7.05, 1),
                (7.6, 0.25),
                (11.4, 0.25),
                (11.7, 0),
            ]
        )
        + anim(
            "d",
            [
                (0, D_SB),
                (T_DROP_A, D_SB),
                (T_DROP_B, D_SF),
                (11.75, D_SF),
                (11.85, D_SB),
            ],
        )
        + "</path>"
    )

    add("<g>")
    for j, (m0, _h, _s) in enumerate(MAIN_PEAKS):
        c = snap(m0)
        i = round((c - MZ0) / STEP_MZ)
        px, py = xpx(c), ypx(smooth_f[i])
        tp = 5.25 + 0.06 * j
        inner = '<path class="pk" d="M-4.5,-14H4.5L0,-6.5Z"/>'
        if m0 in LABELLED:
            inner += (
                f'<text class="fg" y="-18" font-size="10" text-anchor="middle">'
                f"{m0:,}</text>"
            )
        add(
            f'<g transform="translate({f1(px)},{f1(py)})"><g opacity="0" transform="scale(0.001)">'
            + op([(0, 0), (tp, 0), (tp + 0.12, 1), (7.0, 1), (7.35, 0)])
            + tr_scale(
                [
                    (0, 0.001),
                    (tp, 0.001),
                    (tp + 0.22, 1.3),
                    (tp + 0.36, 1),
                    (11.9, 1),
                    (11.95, 0.001),
                ]
            )
            + inner
            + "</g></g>"
        )
    add("</g>")

    add('<g clip-path="url(#mak-strip-clip)">')
    add(
        '<g class="ac" opacity="0">'
        + op([(0, 0), (7.3, 0), (7.4, 1), (11.4, 1), (11.75, 0)])
    )
    for b, v in enumerate(bins):
        a = 0.07 + 0.93 * (v / BMAX) ** 0.6
        add(
            f'<rect x="{f1(X0 + b * bw_px + 0.6)}" y="{f1(STRIP_Y)}" width="{f1(bw_px - 1.2)}" '
            f'height="{f1(STRIP_H)}" fill-opacity="{f2(a)}"/>'
        )
    add("</g></g>")

    # scanning cursor reading the feature vector
    add(
        f'<rect class="fgs" x="{f1(X0 - 6)}" y="{f1(STRIP_Y - 3)}" width="12" '
        f'height="{f1(STRIP_H + 6)}" rx="3" fill="none" stroke-width="1.5" opacity="0">'
        + op([(0, 0), (8.9, 0), (9.0, 1), (9.55, 1), (9.7, 0)])
        + anim(
            "x",
            [
                (0, X0 - 6),
                (9.0, X0 - 6),
                (9.6, X1 - 6),
                (11.9, X1 - 6),
                (11.95, X0 - 6),
            ],
        )
        + "</rect>"
    )

    CARD_X, CARD_Y, CARD_W, CARD_H = 572.0, 70.0, 180.0, 118.0
    add(
        '<g opacity="0" transform="translate(0,8)">'
        + op([(0, 0), (9.3, 0), (9.65, 1), (11.4, 1), (11.7, 0)])
        + tr_translate(
            [
                (0, (0, 8)),
                (9.3, (0, 8)),
                (9.65, (0, 0)),
                (11.75, (0, 0)),
                (11.8, (0, 8)),
            ]
        )
    )
    add(
        f'<rect class="card" x="{f1(CARD_X)}" y="{f1(CARD_Y)}" width="{f1(CARD_W)}" '
        f'height="{f1(CARD_H)}" rx="10"/>'
    )
    add(
        f'<text class="mu" x="{f1(CARD_X + 16)}" y="{f1(CARD_Y + 22)}" font-size="10.5" '
        'font-weight="600" letter-spacing="1">AMR PREDICTION</text>'
    )
    ROWS = [("Ciprofloxacin", "R"), ("Ceftriaxone", "R"), ("Meropenem", "S")]
    COL = {"R": "#D55E00", "S": "#009E73"}
    for k, (drug, lab) in enumerate(ROWS):
        cy = CARD_Y + 46 + 26 * k
        tr = 9.6 + 0.17 * k
        add(
            '<g opacity="0">'
            + op([(0, 0), (tr, 0), (tr + 0.15, 1), (11.75, 1), (11.8, 0)])
        )
        add(
            f'<text class="fg" x="{f1(CARD_X + 16)}" y="{f1(cy + 4.5)}" font-size="13">{drug}</text>'
        )
        pcx = CARD_X + CARD_W - 30
        add(
            f'<g transform="translate({f1(pcx)},{f1(cy)})"><g transform="scale(0.001)">'
            + tr_scale(
                [
                    (0, 0.001),
                    (tr, 0.001),
                    (tr + 0.2, 1.25),
                    (tr + 0.32, 1),
                    (11.8, 1),
                    (11.85, 0.001),
                ]
            )
            + f'<rect x="-14" y="-10" width="28" height="20" rx="10" fill="{COL[lab]}"/>'
            f'<text y="4.4" font-size="12.5" font-weight="700" text-anchor="middle" '
            f'fill="#ffffff">{lab}</text></g></g>'
        )
        add("</g>")
    add("</g>")

    CAP_X, CAP_Y = (X0 + X1) / 2, 302.0
    for k, (_, title, desc) in enumerate(STAGES):
        on = stage_opacity(k)
        add(
            f'<text x="{f1(CAP_X)}" y="{f1(CAP_Y)}" font-size="13" text-anchor="middle" '
            f'opacity="{f2(on[0][1])}">'
            f'<tspan class="fg" font-weight="600">{title}</tspan>'
            f'<tspan class="mu" dx="8">{desc}</tspan>{op(on)}</text>'
        )

    add("</svg>")
    return "\n".join(o) + "\n"


def main() -> None:
    """Write the SVG to the path given on the command line, or to DEFAULT_OUT."""
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    svg = build()
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(svg)
    print(f"Wrote {out} ({len(svg.encode('utf-8'))} bytes)")


if __name__ == "__main__":
    main()
