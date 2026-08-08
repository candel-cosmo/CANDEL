"""Render the TRGB forward-model DAG for the TRGBH0 paper.

Manual TikZ layout. Sized for an MNRAS two-column figure.
"""
import subprocess
import sys
from pathlib import Path

from trgbh0_plot_style import OUTPUT_DIR

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(path for path in SCRIPT_DIR.parents
                if path.name == "paper_TRGBH0")
for path in (SCRIPT_DIR, PLOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


TEX_FILE = SCRIPT_DIR / "trgb_forward_model_dag.tex"
PDF_FILE = OUTPUT_DIR / "trgb_forward_model_dag.pdf"

# =========================================================================
# Manual node positions (x, y) in cm
# =========================================================================
pos = {
    # Global/model-level quantities, on two rows: the TRGB calibration block
    # above, the cosmology, field, and selection block below. Two rows rather
    # than one keeps the figure narrow, and width is what sets the on-page
    # font size once \includegraphics scales it to the text width.
    "Mtrgb": (0.65, 9.75),
    "cstar": (2.35, 9.75),
    "cpars": (4.15, 9.75),
    "sigint": (5.95, 9.75),
    "H0": (0.65, 8.35),
    "rho": (2.35, 8.35),
    "Vfield": (4.05, 8.35),
    "bias": (5.75, 8.35),
    "pv": (7.45, 8.35),
    "sigv": (9.15, 8.35),
    "selcuts": (10.95, 8.35),
    # Anchor constraints
    "anchprior": (-0.25, 7.05),
    "anchmu": (-0.25, 6.30),
    "anchgeom": (-0.25, 5.45),
    "anchmtrue": (2.45, 5.45),
    "anchsky": (-0.25, 4.25),
    "anchmobs": (2.45, 4.25),
    # Host-level distance and PV model
    "skydelta": (5.05, 3.55),
    "rdist": (8.75, 3.55),
    "rtrue": (8.75, 2.45),
    "mu": (6.45, 1.40),
    "zcos": (8.75, 1.40),
    "vpec": (11.15, 1.40),
    # Host-level colour and observables
    "cdist": (2.25, 3.15),
    "ctrue": (2.25, 2.05),
    "csamp": (1.65, -0.65),
    "cobs": (2.25, -1.95),
    "mtrue": (5.45, 0.45),
    "cztrue": (11.15, 0.40),
    "msamp": (5.45, -0.65),
    "czsamp": (11.15, -0.65),
    "mobs": (4.75, -1.95),
    "czobs": (11.15, -1.95),
    "selected": (13.90, -1.95),
    "detfrac": (13.90, -3.05),
}

latex_labels = {
    "Mtrgb": r"$M_0$",
    "cstar": r"$c_\star$",
    "cpars": r"$(\bar c,\,w_c)$",
    "sigint": r"$\sigma_{\rm int}$",
    "H0": r"$H_0$",
    "pv": r"$\mathbf{V}_{\rm ext}$",
    "sigv": r"$\sigma_v$",
    "rho": r"$\delta(\bm{x})$",
    "Vfield": r"$\bm{V}(\bm{x})$",
    "bias": r"$\bm{b}$",
    "selcuts": r"TRGB\\selection",
    "anchprior": r"$\mu_a \sim p(\mu_a)$",
    "anchmu": r"$\mu_a$",
    "anchgeom": (
        r"$\mu_{a,\rm obs} \sim$\\"
        r"$\mathcal{N}(\mu_a,\epsilon_{\mu,a}^2)$"
    ),
    "anchmtrue": r"$m_a$",
    "anchsky": (
        r"$(\ell_a,b_a) \sim$\\"
        r"$\delta(\ell_a-\ell_{{\rm obs},a})$\\"
        r"$\delta(b_a-b_{{\rm obs},a})$"
    ),
    "anchmobs": (
        r"$m_{{\rm obs},a} \sim$\\"
        r"$\mathcal{N}(m_a,$\\"
        r"$\epsilon_{m,a}^2)$"
    ),
    "skydelta": (
        r"$(\ell_i,b_i) \sim$\\"
        r"$\delta(\ell_i-\ell_{{\rm obs},i})$\\"
        r"$\delta(b_i-b_{{\rm obs},i})$"
    ),
    "rdist": (
        r"$(r_i,\ell_i,b_i) \sim$\\"
        r"$p(\bm{x}\mid\delta,\,\bm{b},\,H_0)$"
    ),
    "rtrue": r"$r_i$",
    "mu": r"$\mu_i$",
    "zcos": r"$z_{{\rm cos},i}$",
    "vpec": r"$V_{{\rm pec},i}$",
    "cdist": (
        r"$c_i \sim$\\"
        r"$\mathcal{N}(\bar c,\,w_c^2)$"
    ),
    "ctrue": r"$c_i$",
    "csamp": (
        r"$c_i^{\rm obs} \sim$\\"
        r"$\mathcal{N}(c_i,\epsilon_{c,i}^2)$"
    ),
    "cobs": r"$c_i^{\rm obs}$",
    "mtrue": r"$m_i$",
    "cztrue": r"$cz_i$",
    "msamp": (
        r"$m_{{\rm obs},i} \sim$\\"
        r"$\mathcal{N}(m_i,\epsilon_{m,i}^2+\sigma_{\rm int}^2)$"
    ),
    "czsamp": (
        r"$cz_{{\rm CMB},i} \sim$\\"
        r"$\mathcal{N}(cz_i,\epsilon_{cz,i}^2+\sigma_v^2)$"
    ),
    "mobs": r"$m_{{\rm obs},i}$",
    "czobs": r"$cz_{{\rm CMB},i}$",
    "selected": r"$S_i=1$",
    "detfrac": r"$p(S=1\mid\Lambda)$",
}

node_styles = {
    "Mtrgb": "global",
    "cstar": "global",
    "cpars": "global",
    "sigint": "global",
    "H0": "global",
    "pv": "global",
    "sigv": "global",
    "rho": "input",
    "Vfield": "input",
    "bias": "global",
    "selcuts": "global",
    "anchprior": "popdist",
    "anchmu": "latent",
    "anchgeom": "sample",
    "anchmtrue": "det",
    "anchsky": "sample",
    "anchmobs": "data",
    "skydelta": "sample",
    "rdist": "popdist",
    "rtrue": "latent",
    "mu": "det",
    "zcos": "det",
    "vpec": "det",
    "cdist": "sample",
    "ctrue": "latent",
    "csamp": "sample",
    "cobs": "data",
    "mtrue": "det",
    "cztrue": "det",
    "msamp": "sample",
    "czsamp": "sample",
    "mobs": "data",
    "czobs": "data",
    "selected": "conditioned",
    "detfrac": "selection",
}

edges = [
    ("Mtrgb", "anchmtrue"),
    ("cpars", "cdist"),
    ("cdist", "ctrue"),
    ("ctrue", "csamp"),
    ("csamp", "cobs"),
    ("anchprior", "anchmu"),
    ("anchmu", "anchgeom"),
    ("anchmu", "anchmtrue"),
    ("anchmtrue", "anchmobs"),
    ("anchsky", "anchmobs"),
    ("rho", "rdist"),
    ("bias", "rdist"),
    ("H0", "rdist"),
    ("rdist", "skydelta"),
    ("rdist", "rtrue"),
    ("rtrue", "mu"),
    ("rtrue", "zcos"),
    ("H0", "zcos"),
    ("Vfield", "vpec"),
    ("pv", "vpec"),
    ("skydelta", "vpec"),
    ("rtrue", "vpec"),
    ("Mtrgb", "mtrue"),
    ("cstar", "mtrue"),
    ("ctrue", "mtrue"),
    ("mu", "mtrue"),
    ("mtrue", "msamp"),
    ("sigint", "msamp"),
    ("msamp", "mobs"),
    ("zcos", "cztrue"),
    ("vpec", "cztrue"),
    ("cztrue", "czsamp"),
    ("sigv", "czsamp"),
    ("czsamp", "czobs"),
    ("selcuts", "selected"),
    ("mobs", "selected"),
    ("selcuts", "detfrac"),
]

level_labels = []


def node_style(name):
    style = node_styles[name]
    if name in {"msamp", "czsamp"}:
        style = f"{style}, text width=3.35cm"
    if name in {"cdist", "csamp"}:
        style = f"{style}, text width=2.15cm"
    if name in {"skydelta"}:
        style = f"{style}, text width=2.55cm"
    if name in {"anchprior", "anchgeom", "anchsky", "anchmobs", "rdist"}:
        style = f"{style}, text width=2.20cm"
    if name == "sigv":
        style = f"{style}, minimum width=1.05cm"
    if name == "sigint":
        style = f"{style}, minimum width=1.15cm"
    return style


# =========================================================================
# Generate TikZ
# =========================================================================
node_lines = []
for name in pos:
    x, y = pos[name]
    node_lines.append(
        f"\\node[{node_style(name)}] ({name}) at "
        f"({x:.2f}, {y:.2f}) {{{latex_labels[name]}}};"
    )

# Curved edges as (out angle, in angle, target anchor). Angle-based routing
# stays correct when nodes move; absolute control points had to be re-tuned by
# hand after every layout change.
edge_routing = {
    ("Mtrgb", "anchmtrue"): (-70, 110, "anchmtrue.north"),
    ("cpars", "cdist"): (-100, 60, "cdist.north east"),
    ("Mtrgb", "mtrue"): (-80, 160, "mtrue.north west"),
    ("cstar", "mtrue"): (-85, 120, "mtrue.north"),
    ("sigint", "msamp"): (-90, 75, "msamp.north east"),
    ("rho", "rdist"): (-90, 90, "rdist.north"),
    ("bias", "rdist"): (-115, 55, "rdist.north east"),
    ("H0", "rdist"): (-65, 125, "rdist.north west"),
    ("H0", "zcos"): (-80, 130, "zcos.north west"),
    ("Vfield", "vpec"): (-95, 120, "vpec.north west"),
    ("pv", "vpec"): (-100, 55, "vpec.north east"),
    ("sigv", "czsamp"): (-100, 60, "czsamp.north east", 0.6),
    ("selcuts", "selected"): (-70, 90, "selected.north", 0.7),
    ("selcuts", "detfrac"): (-55, 55, "detfrac.north east", 0.3),
    ("skydelta", "vpec"): (-25, 170, "vpec.west"),
    ("rtrue", "vpec"): (-40, 140, "vpec.north"),
    ("ctrue", "mtrue"): (-55, 165, "mtrue.north west"),
    ("mobs", "selected"): (-45, -135, "selected.south west", 0.45),
}
sel_edges = {("selcuts", "selected"), ("selcuts", "detfrac"),
             ("mobs", "selected")}

edge_lines = []
for a, b in edges:
    if (a, b) == ("rdist", "skydelta"):
        continue
    style = "sel edge" if (a, b) in sel_edges else "dag edge"
    route = edge_routing.get((a, b))
    if route is None:
        edge_lines.append(f"\\draw[{style}] ({a}) -- ({b});")
    else:
        out_angle, in_angle, target = route[:3]
        looseness = route[3] if len(route) > 3 else 1.0
        edge_lines.append(
            f"\\draw[{style}] ({a}) to[out={out_angle}, in={in_angle}, "
            f"looseness={looseness}] ({target});"
        )


level_lines = [
    (
        f"\\node[level label, anchor=east, align=right] "
        f"at ({x:.2f}, {y:.2f}) {{{label}}};"
    )
    for label, x, y in level_labels
]

nodes_block = "\n".join(node_lines)
edges_block = "\n".join(edge_lines)
levels_block = "\n".join(level_lines)

tex = rf"""
\documentclass[border=5pt]{{standalone}}
\usepackage{{tikz}}
\usetikzlibrary{{arrows.meta, backgrounds, decorations.pathreplacing,
                 fit, shapes.geometric}}
\usepackage{{amsmath, amssymb}}
\usepackage{{bm}}

\definecolor{{colglobal}}{{HTML}}{{4F6D7A}}
\definecolor{{colpop}}{{HTML}}{{8F5A83}}
\definecolor{{coldet}}{{HTML}}{{1F9D8A}}
\definecolor{{colsample}}{{HTML}}{{8A4F7D}}
\definecolor{{coldata}}{{HTML}}{{D8D8D8}}

\tikzset{{
    dag node/.style={{
        draw=black!70, semithick, align=center,
        font=\footnotesize, inner sep=2.3pt,
    }},
    global/.style={{dag node, rectangle, rounded corners=2pt,
        minimum width=1.45cm, minimum height=0.56cm, fill=colglobal!10}},
    input/.style={{dag node, rectangle, rounded corners=2pt,
        dashed, draw=black!60, minimum width=1.45cm,
        minimum height=0.56cm, fill=coldata!35}},
    selection/.style={{dag node, rectangle, dashed, draw=black!60,
        fill=white, minimum width=1.85cm, minimum height=0.56cm}},
    popdist/.style={{dag node, rectangle, draw=colpop!85!black,
        dashed, fill=colpop!8, minimum height=0.66cm,
        text width=1.95cm, inner sep=1.3pt}},
    latent/.style={{dag node, ellipse, fill=white,
        minimum width=1.45cm, minimum height=0.58cm}},
    det/.style={{dag node, diamond, aspect=2.35, draw=coldet!85!black,
        fill=coldet!8, inner xsep=1pt, inner ysep=1pt}},
    sample/.style={{dag node, rectangle, draw=colsample!85!black,
        fill=colsample!8, minimum height=0.62cm,
        text width=1.95cm, inner sep=1.3pt}},
    data/.style={{dag node, rectangle, very thick, fill=coldata,
        minimum width=1.5cm, minimum height=0.56cm}},
    conditioned/.style={{data, double, double distance=1pt}},
    plate/.style={{draw=black!55, rounded corners=5pt, dashed,
        inner xsep=9pt, inner ysep=10pt}},
    level label/.style={{font=\scriptsize, text=black!70}},
    dag edge/.style={{-{{Stealth[length=3pt, width=2.5pt]}}, thin,
        draw=black!55}},
    sel edge/.style={{dag edge, draw=black!38}},
}}

\begin{{document}}
\begin{{tikzpicture}}

% No manual bounding box: standalone crops to the ink, so the figure is not
% padded with dead margin that \includegraphics would then scale away.

% ===== LEGEND =====
\begin{{scope}}[on background layer]
    \fill[black!6, rounded corners=3pt] (-1.35, 10.70)
        rectangle (8.55, 12.75);
\end{{scope}}
\node[global, minimum width=1.35cm] at (-0.55, 12.25) {{Global}};
\node[input, minimum width=1.55cm] at (1.30, 12.25) {{Fixed\\input}};
\node[sample, text width=1.35cm] at (3.10, 12.25) {{Sampling}};
\node[latent, minimum width=1.25cm] at (4.85, 12.25) {{Latent}};
\node[det, minimum width=1.25cm] at (7.10, 12.25) {{Deterministic}};
\node[data, minimum width=1.35cm] at (-0.45, 11.20) {{Observed}};
\node[conditioned, minimum width=1.35cm] at (1.45, 11.20) {{Selection}};
\node[popdist, text width=1.50cm] at (3.50, 11.20) {{Population}};
\node[selection, minimum width=1.65cm] at (5.75, 11.20)
    {{Detection\\fraction}};

% ===== NODES =====
{nodes_block}

% ===== PLATES =====
\begin{{scope}}[on background layer]
    \node[plate, fit=(anchprior)(anchmu)(anchgeom)(anchmtrue)
        (anchsky)(anchmobs)] {{}};
    \node[plate, inner ysep=6pt, fit=(skydelta)(rdist)(rtrue)(mu)(zcos)
        (vpec)(cdist)(ctrue)(csamp)(cobs)(mtrue)(cztrue)(msamp)(czsamp)
        (mobs)(czobs)(selected)] {{}};
\end{{scope}}

% ===== EDGES =====
\begin{{scope}}[on background layer]
{edges_block}
\end{{scope}}
\draw[dag edge, draw=black!70]
    (rdist.west) -- (skydelta.east);

% ===== PLATE LABELS =====
\node[font=\scriptsize, fill=white, inner sep=1.2pt, anchor=west]
    at (-1.16, 7.56) {{Anchor $a\in\{{\rm LMC,N4258\}}$}};
\node[font=\scriptsize, fill=white, inner sep=1.2pt, anchor=east]
    at (13.30, 3.95) {{TRGB host $i=1,\ldots,N_{{\rm host}}$}};

\end{{tikzpicture}}
\end{{document}}
"""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEX_FILE.write_text(tex)

    result = subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-output-directory",
            str(OUTPUT_DIR),
            str(TEX_FILE),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("pdflatex FAILED:")
        print(result.stdout[-2000:])
        print(result.stderr[-500:])
    else:
        for ext in [".aux", ".log"]:
            p = OUTPUT_DIR / f"trgb_forward_model_dag{ext}"
            if p.exists():
                p.unlink()
        print(f"DAG rendered to {PDF_FILE}")


if __name__ == "__main__":
    main()
