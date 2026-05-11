from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "track_b_unrate_24m_mase_metrics.csv"
FIG_DIR = ROOT / "figures"
SLIDE_DIR = ROOT / "slides"
FIG_DIR.mkdir(exist_ok=True)
SLIDE_DIR.mkdir(exist_ok=True)

# Mandatory Nature-chart rcParams: editable SVG text.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 1.35,
    "legend.frameon": False,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PALETTE = {
    "baseline_dark": "#484878",
    "baseline_mid": "#7884B4",
    "baseline_soft": "#B4C0E4",
    "ours_base": "#E4CCD8",
    "ours_large": "#B64342",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "neutral_light": "#D8D8D8",
    "delta_up": "#2E9E44",
}

metrics = pd.read_csv(RESULTS)

baseline_names = ["Naive / Random Walk", "ARD(6)"]
ml = metrics[~metrics["model"].isin(baseline_names)].copy()
best_ml = (
    ml.sort_values("mase")
    .groupby("h", as_index=False)
    .first()[["h", "model", "mae", "mase"]]
    .rename(columns={"model": "best_ml_model", "mae": "best_ml_mae", "mase": "best_ml_mase"})
)

plot_df = (
    metrics[metrics["model"].isin(baseline_names)]
    .pivot(index="h", columns="model", values="mase")
    .reset_index()
    .merge(best_ml[["h", "best_ml_mase"]], on="h", how="left")
)
plot_df.to_csv(FIG_DIR / "track_b_unrate_mase_chart_source.csv", index=False)

best_overall = (
    metrics.sort_values("mase")
    .groupby("h", as_index=False)
    .first()[["h", "model", "mae", "mase"]]
    .sort_values("h")
)
best_overall.to_csv(FIG_DIR / "track_b_unrate_mase_best_by_horizon.csv", index=False)

# Nature-style single-panel chart.
fig, ax = plt.subplots(figsize=(5.9, 3.65))

x = plot_df["h"].to_numpy(dtype=float)
series = [
    ("Naive / random walk", plot_df["Naive / Random Walk"].to_numpy(float), PALETTE["baseline_dark"], "o"),
    ("ARD(6)", plot_df["ARD(6)"].to_numpy(float), PALETTE["baseline_mid"], "s"),
    ("Best ML model", plot_df["best_ml_mase"].to_numpy(float), PALETTE["ours_large"], "D"),
]

for label, y, color, marker in series:
    ax.plot(x, y, color=color, lw=2.1, marker=marker, ms=6.0, mec="white", mew=0.8, label=label)

# Direct value labels to reduce lookup cost on slide.
for _, row in plot_df.iterrows():
    h = float(row["h"])
    ax.text(h + 0.08, row["Naive / Random Walk"] + 0.03, f"{row['Naive / Random Walk']:.2f}", color=PALETTE["baseline_dark"], fontsize=9)
    ax.text(h + 0.08, row["ARD(6)"] - 0.19, f"{row['ARD(6)']:.2f}", color=PALETTE["baseline_mid"], fontsize=9)
    ax.text(h + 0.08, row["best_ml_mase"] + 0.08, f"{row['best_ml_mase']:.2f}", color=PALETTE["ours_large"], fontsize=9)

ax.set_xlabel("Forecast horizon (months)", labelpad=8)
ax.set_ylabel("Mean absolute scaled error (MASE)", labelpad=8)
ax.set_xticks([1, 3, 6])
ax.set_xticklabels(["1", "3", "6"])
ax.set_ylim(0.7, 4.45)
ax.set_yticks([1, 2, 3, 4])
ax.spines["bottom"].set_linewidth(1.35)
ax.spines["left"].set_linewidth(1.35)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=1, fontsize=9.2, handlelength=2.4)
ax.set_title("UNRATE level forecasts: simple baselines dominate", loc="left", fontsize=12.5, fontweight="bold", pad=12)
ax.text(0.01, -0.26, "Track B features use a trailing 24-month fixed input window; lower MASE is better.", transform=ax.transAxes, fontsize=8.8, color=PALETTE["neutral_mid"])

fig.tight_layout(pad=1.4)
for ext in ["svg", "pdf", "png"]:
    out = FIG_DIR / f"track_b_unrate_mase_nature.{ext}"
    if ext == "png":
        fig.savefig(out, dpi=300, bbox_inches="tight")
    else:
        fig.savefig(out, bbox_inches="tight")
plt.close(fig)

# LaTeX slide with one chart and one table.
def fmt_pp(x: float) -> str:
    return f"{x:.3f} pp"

def fmt_mase(x: float) -> str:
    return f"{x:.3f}"

def latex_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("_", r"\_")

rows = []
for _, r in best_overall.iterrows():
    rows.append(f"{int(r['h'])} mo & {latex_escape(r['model'])} & {fmt_pp(r['mae'])} & {fmt_mase(r['mase'])} \\\\")
rows_tex = "\n".join(rows)

tex = rf"""
\documentclass[aspectratio=169,10pt]{{beamer}}
\usetheme{{default}}
\usefonttheme{{professionalfonts}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{xcolor}}
\definecolor{{ink}}{{HTML}}{{0F172A}}
\definecolor{{muted}}{{HTML}}{{475569}}
\definecolor{{blue}}{{HTML}}{{484878}}
\definecolor{{red}}{{HTML}}{{B64342}}
\definecolor{{green}}{{HTML}}{{166534}}
\setbeamercolor{{normal text}}{{fg=ink,bg=white}}
\setbeamercolor{{frametitle}}{{fg=ink,bg=white}}
\setbeamertemplate{{navigation symbols}}{{}}
\setbeamertemplate{{footline}}{{}}
\setbeamertemplate{{headline}}{{}}
\setlength{{\tabcolsep}}{{4.8pt}}

\begin{{document}}
\begin{{frame}}[t]{{Track B UNRATE: persistence beats richer ML}}
\vspace{{-0.25em}}
{{\small\color{{muted}} 24-month fixed input window $\cdot$ look-ahead $h \in \{{1,3,6\}}$ months $\cdot$ metric: mean absolute scaled error (MASE)}}

\vspace{{0.8em}}
\begin{{columns}}[T,totalwidth=\textwidth]
  \begin{{column}}{{0.62\textwidth}}
    \includegraphics[width=\linewidth]{{../figures/track_b_unrate_mase_nature.pdf}}
  \end{{column}}
  \begin{{column}}{{0.35\textwidth}}
    \textbf{{Best model by horizon}}\\[-0.35em]
    {{\scriptsize Lower MASE is better. MAE is in percentage points of unemployment.}}
    \vspace{{0.45em}}

    \begin{{tabular}}{{@{{}}r l r r@{{}}}}
    \toprule
    $h$ & Model & MAE & MASE \\
    \midrule
    {rows_tex}
    \bottomrule
    \end{{tabular}}

    \vspace{{1.0em}}
    \textbf{{Interpretation.}}\\[-0.2em]
    {{\small UNRATE levels are highly persistent. With interpretable Track B inputs, richer ML models do not beat the naive/ARD baselines.}}

    \vspace{{0.8em}}
    {{\scriptsize\color{{muted}} Track B: UNRATE level/delta, building permits, S\&P 500, consumer sentiment, and 10Y--3M yield spread level/delta.}}
  \end{{column}}
\end{{columns}}

\vfill
{{\tiny\color{{muted}} Source: \texttt{{notebooks/03\_track\_b\_unrate\_fixed\_window\_mase.ipynb}}; results: \texttt{{results/track\_b\_unrate\_24m\_mase\_metrics.csv}}.}}
\end{{frame}}
\end{{document}}
"""
tex_path = SLIDE_DIR / "track_b_unrate_one_slide.tex"
tex_path.write_text(textwrap.dedent(tex).strip() + "\n")

# Compile if pdflatex is present.
try:
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=SLIDE_DIR,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
except FileNotFoundError:
    print("pdflatex not found; wrote TeX only")
except subprocess.CalledProcessError as e:
    print(e.stdout)
    raise

print("Wrote:")
for p in [
    FIG_DIR / "track_b_unrate_mase_nature.svg",
    FIG_DIR / "track_b_unrate_mase_nature.pdf",
    FIG_DIR / "track_b_unrate_mase_nature.png",
    FIG_DIR / "track_b_unrate_mase_chart_source.csv",
    FIG_DIR / "track_b_unrate_mase_best_by_horizon.csv",
    tex_path,
    SLIDE_DIR / "track_b_unrate_one_slide.pdf",
]:
    if p.exists():
        print(" -", p.relative_to(ROOT))
