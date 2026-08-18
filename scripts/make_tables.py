#!/usr/bin/env python3
"""
make_tables.py -- generate Tables IV-VIII of the APSIPA camera-ready
paper as standalone LaTeX fragments, computed directly from the raw
results_*.txt eval dumps and the benchmark log.

Run from the repo root:

    python scripts/make_tables.py

Writes paper/tables/table_{trained,pretrain_ref,throughput,perjoint,
consolidated}.tex, one \\input-able fragment per \\label in main.tex.

Rounding: all display rounding uses Decimal + ROUND_HALF_UP on the
*exact text* pulled from the results_*.txt files, never Python's
binary-float round(). round(0.6645, 3) == 0.664 in IEEE754 (0.6645
isn't exactly representable and lands a hair below), but the paper's
own hand-rounding of that value is 0.665 -- decimal half-up matches
the source of truth. Every derived quantity (Mean, Range, Delta AUC)
is computed from values already rounded to display precision, not raw
full-precision floats -- e.g. Table IV's "+0.037" AUC delta is
0.723-0.686 on the *displayed* 3dp numbers; the unrounded
0.7234-0.6857 = 0.0377 would round to +0.038.

Bolding rule: within each row, bold whichever of the TRAINED-MODEL
candidate columns (Direct Sup., Dist. V2-100, FT V2-150) attains the
best value -- this is a real per-row argmax only in Table VII (and its
Mean row), where all three columns are the same higher-is-better
metric (PCK@0.2). Tables IV, V and VIII mix higher-is-better
(PCK/AUC/FPS) and lower-is-better (MPJPE/Params) metrics in the same
row, and the source paper does not actually run a per-cell argmax
there -- it always highlights the final proposed model (FT V2-150),
including on categorical rows (Open/retrain, KD supervision) and ties
(Params, GPU FPS) where a raw argmax would be ambiguous or wrong. That
fixed convention is reproduced as-is rather than papered over with a
sign-blind max() that would silently mis-bold MPJPE.
"""

import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "paper" / "tables"

JOINTS = [
    "Wrist", "Thumb_MCP", "Thumb_PIP", "Thumb_DIP", "Thumb_Tip",
    "Index_MCP", "Index_PIP", "Index_DIP", "Index_Tip",
    "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_Tip",
    "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_Tip",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_Tip",
]
JOINT_LABELS = {j: j.replace("_", " ") for j in JOINTS}

# ---- raw source files (data-driven cells) --------------------------
DIRECT_FILE = ROOT / "results_direct_sup_GT.txt"
DIST100_FILE = ROOT / "experiments" / "results_v2_epoch100.txt"
FT150_FILE = ROOT / "results_distilled_v2_ft_epoch_150.txt"
TEACHER_FILE = ROOT / "results_teacher_gt.txt"
BENCH_FILE = ROOT / "thesis_benchmark_results.txt"

# ---- external / non-reproducible reference constants ---------------
# These do not come from a results_*.txt in this repo. They are cited
# exactly as the paper itself cites them (see thesis_benchmark_results.txt,
# which hardcodes the teacher accuracy line as "(from paper)", and
# Table IV footnote a / Table VIII footnote b: "official MMPose
# benchmark"). Left as named constants rather than silently fabricated
# from a nearby file.
TEACHER_PCK, TEACHER_AUC, TEACHER_MPJPE = Decimal("0.992"), Decimal("0.902"), Decimal("2.21")
MP_SDK_PARAMS_M = Decimal("1.98")
# 500-sample legacy run (vs. GT), reference source file not present in
# this repo snapshot; distinct from three_way_comparison.txt, which
# uses a teacher-decoded (not GT) reference and different Det Rate
# semantics, so it does not reproduce these.
UNTRAINED = dict(pck=Decimal("0.005"), auc=Decimal("0.025"),
                  mpjpe=Decimal("186.96"), det=Decimal("100"))
MP_PRETRAIN = dict(pck=Decimal("0.243"), auc=Decimal("0.323"),
                    mpjpe=Decimal("126.95"), det=Decimal("69.4"))


def parse_results_file(path):
    """Parse a results_*.txt, keeping every number as Decimal built
    from the exact source text (never via float())."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    summary = {}
    for key, pat in [
        ("PCK", r"PCK@0\.2(?:\s*\(norm\))?:\s*([\d.]+)"),
        ("MPJPE", r"(?:MPJPE|EPE):\s*([\d.]+)"),
        ("AUC", r"AUC:\s*([\d.]+)"),
        ("DET", r"Det Rate:\s*([\d.]+)"),
    ]:
        m = re.search(pat, text)
        if m:
            summary[key] = Decimal(m.group(1))
    per_joint = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] in JOINTS:
            per_joint[parts[0]] = Decimal(parts[1])
    return summary, per_joint


def parse_bench(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    student_fps = Decimal(re.search(
        r"\[5\] GPU Inference.*?Median\s*:\s*[\d.]+\s*ms\s*.*?([\d.]+)\s*FPS",
        text, re.S).group(1))
    teacher_fps = Decimal(re.search(
        r"\[11\].*?Median\s*:\s*[\d.]+\s*ms\s*.*?([\d.]+)\s*FPS",
        text, re.S).group(1))
    params_m = Decimal(re.search(
        r"Parameters \(M\)\s*:\s*([\d.]+)\s*M", text).group(1))
    teacher_params_m = Decimal(re.findall(
        r"Parameters \(M\)\s*:\s*([\d.]+)\s*M", text)[1])
    batches = {}
    for bs, thr in re.findall(
            r"Batch size\s*(\d+)\s*:\s*([\d.]+)\s*images\s*/\s*sec", text):
        batches[int(bs)] = Decimal(thr)
    return dict(student_fps=student_fps, teacher_fps=teacher_fps,
                params_m=params_m, teacher_params_m=teacher_params_m,
                batches=batches)


Q3, Q2 = Decimal("0.001"), Decimal("0.01")


def r3(x):
    return Decimal(x).quantize(Q3, rounding=ROUND_HALF_UP)


def r2(x):
    return Decimal(x).quantize(Q2, rounding=ROUND_HALF_UP)


def fmt3(x):
    return f"{r3(x)}"


def fmt2(x):
    return f"{r2(x)}"


def bold_argmax(cells):
    """cells: list of (text, value_or_None), all values already at
    display precision. Returns text list with the max-value cell(s)
    (ties included) wrapped in \\textbf. None-valued cells are never
    candidates (external / reference columns)."""
    candidates = [v for _, v in cells if v is not None]
    if not candidates:
        return [t for t, _ in cells]
    winner = max(candidates)
    return [f"\\textbf{{{t}}}" if v is not None and v == winner else t
            for t, v in cells]


def load_all():
    direct_s, direct_pj = parse_results_file(DIRECT_FILE)
    dist100_s, dist100_pj = parse_results_file(DIST100_FILE)
    ft150_s, ft150_pj = parse_results_file(FT150_FILE)
    teacher_s, _ = parse_results_file(TEACHER_FILE)
    bench = parse_bench(BENCH_FILE)
    return dict(direct_s=direct_s, direct_pj=direct_pj,
                dist100_s=dist100_s, dist100_pj=dist100_pj,
                ft150_s=ft150_s, ft150_pj=ft150_pj,
                teacher_det=teacher_s["DET"], bench=bench)


def pct(v):
    """100% -> '100\\%', 69.4% -> '69.4\\%', 99.96% -> '99.96\\%'
    (matches paper's percent formatting: trailing zeros dropped, no
    scientific notation)."""
    s = f"{r2(v):f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return f"{s}\\%"


# ---------------------------------------------------------------- IV
def gen_table_trained(d):
    direct, dist100, ft150, bench = d["direct_s"], d["dist100_s"], d["ft150_s"], d["bench"]
    params = r2(bench["params_m"])
    gpu_fps = bench["student_fps"]
    teacher_params = r2(bench["teacher_params_m"])
    teacher_fps = bench["teacher_fps"]

    def delta_auc(auc):
        return r3(auc) - r3(direct["AUC"])

    rows = [
        dict(name="Direct Sup.\\ (ep.\\,25)", pck=direct["PCK"], auc=direct["AUC"],
             mpjpe=direct["MPJPE"], det=direct["DET"], delta="ref.", bold=False),
        dict(name="Stage 1: Dist.\\ V2 (ep.\\,100)", pck=dist100["PCK"], auc=dist100["AUC"],
             mpjpe=dist100["MPJPE"], det=dist100["DET"],
             delta=f"$+${fmt3(delta_auc(dist100['AUC']))}", bold=False),
        dict(name="Stage 2: FT V2 (ep.\\,150)", pck=ft150["PCK"], auc=ft150["AUC"],
             mpjpe=ft150["MPJPE"], det=ft150["DET"],
             delta=f"$+${fmt3(delta_auc(ft150['AUC']))}", bold=True),
    ]
    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\caption{Trained Model Comparison --- Full RHD2D Test Set")
    lines.append(r"         (2,727 samples, vs.\ GT annotations)}")
    lines.append(r"\label{tab:trained}")
    lines.append(r"\centering")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{lccccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Params} & \textbf{PCK@0.2} & \textbf{AUC}")
    lines.append(r"  & \textbf{MPJPE} & \textbf{Det.} & \textbf{GPU} & \textbf{$\Delta$ AUC} \\")
    lines.append(r" & \textbf{(M)} & & & \textbf{(px)} & \textbf{Rate} & \textbf{FPS} & \textbf{vs.\ Direct} \\")
    lines.append(r"\midrule")
    for row in rows:
        b = row["bold"]
        def w(s):
            return f"\\textbf{{{s}}}" if b else s
        name = f"\\textbf{{{row['name']}}}" if b else row['name']
        params_s = w(f"{params:.2f}")
        pck_s = w(fmt3(row['pck']))
        auc_s = w(fmt3(row['auc']))
        mpjpe_s = w(fmt2(row['mpjpe']))
        det_s = w(pct(row['det']))
        fps_s = w(f"{gpu_fps:.1f}")
        delta_s = w(row['delta'])
        lines.append(f"{name}")
        lines.append(f"  & {params_s} & {pck_s} & {auc_s} & {mpjpe_s} & {det_s} & {fps_s} & {delta_s} \\\\")
    lines.append(r"\midrule")
    lines.append(r"Teacher (HRNetV2-W18)$^{a}$")
    lines.append(
        f"  & {teacher_params:.2f} & {fmt3(TEACHER_PCK)} & {fmt3(TEACHER_AUC)}"
        f" & {fmt2(TEACHER_MPJPE)} & {pct(d['teacher_det'])}"
        f" & {teacher_fps:.1f} & --- \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\multicolumn{8}{l}{\scriptsize $^{a}$Upper-bound reference (official")
    lines.append(r"  MMPose benchmark); not a deployment candidate.}")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- V
def gen_table_pretrain_ref(d):
    ft150 = d["ft150_s"]
    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\caption{Pre-training Baseline Reference --- Student Architecture")
    lines.append(r"         Initialization States (500 samples$^{\dagger}$, vs.\ GT)}")
    lines.append(r"\label{tab:pretrain_ref}")
    lines.append(r"\centering")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{PCK@0.2} & \textbf{AUC} & \textbf{MPJPE (px)} & \textbf{Det.\ Rate} \\")
    lines.append(r"\midrule")
    lines.append(
        f"Untrained BlazeHandLandmark (random init.)  & {fmt3(UNTRAINED['pck'])}"
        f" & {fmt3(UNTRAINED['auc'])} & {fmt2(UNTRAINED['mpjpe'])} & {pct(UNTRAINED['det'])} \\\\"
    )
    lines.append(
        f"MediaPipe-inspired BlazeHandLandmark(pre-distillation weights)    & {fmt3(MP_PRETRAIN['pck'])}"
        f" & {fmt3(MP_PRETRAIN['auc'])} & {fmt2(MP_PRETRAIN['mpjpe'])} & {pct(MP_PRETRAIN['det'])} \\\\"
    )
    lines.append(r"\midrule")
    lines.append(
        r"\textbf{FT V2 ep.\,150} \textit{(after distillation)} & "
        f"\\textbf{{{fmt3(ft150['PCK'])}}} & \\textbf{{{fmt3(ft150['AUC'])}}}"
        f" & \\textbf{{{fmt2(ft150['MPJPE'])}}} & \\textbf{{{pct(ft150['DET'])}}} \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\vspace{3pt}")
    lines.append(r"\parbox{\columnwidth}{\scriptsize")
    lines.append(r"  $^{\dagger}$500-sample subset; detection failures scored at")
    lines.append(r"  256\,px per joint (not excluded).}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- VI
def gen_table_throughput(d):
    batches = d["bench"]["batches"]
    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\caption{Multi-Batch GPU Throughput --- Student (FT V2 ep.\,150)}")
    lines.append(r"\label{tab:throughput}")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{8pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.1}")
    lines.append(r"\begin{tabular}{cc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Batch Size} & \textbf{Throughput (img/s)} \\")
    lines.append(r"\midrule")
    for bs in sorted(batches):
        pad = " " if bs < 10 else ""
        lines.append(f"{bs}{pad} & {batches[bs]:.1f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------- VII
def gen_table_perjoint(d):
    direct_pj, dist100_pj, ft150_pj = d["direct_pj"], d["dist100_pj"], d["ft150_pj"]
    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\caption{Per-Joint PCK@0.2 --- Trained Models (RHD2D, vs.\ GT)}")
    lines.append(r"\label{tab:perjoint}")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Joint} & \textbf{Direct Sup.} &")
    lines.append(r"  \textbf{Dist.\ V2-100} & \textbf{FT V2-150} \\")
    lines.append(r"\midrule")

    cols = {"direct": [], "dist100": [], "ft150": []}
    for j in JOINTS:
        dv, sv, fv = r3(direct_pj[j]), r3(dist100_pj[j]), r3(ft150_pj[j])
        cols["direct"].append(dv)
        cols["dist100"].append(sv)
        cols["ft150"].append(fv)
        cells = bold_argmax([
            (fmt3(dv), dv), (fmt3(sv), sv), (fmt3(fv), fv),
        ])
        lines.append(f"{JOINT_LABELS[j]:<11} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")

    lines.append(r"\midrule")
    means = {k: r3(sum(v) / len(v)) for k, v in cols.items()}
    mean_cells = bold_argmax([
        (fmt3(means["direct"]), means["direct"]),
        (fmt3(means["dist100"]), means["dist100"]),
        (fmt3(means["ft150"]), means["ft150"]),
    ])
    lines.append(f"\\textbf{{Mean}}         & {mean_cells[0]} & {mean_cells[1]} & {mean_cells[2]} \\\\")
    ranges = {k: max(v) - min(v) for k, v in cols.items()}
    lines.append(
        f"Range (max$-$min)     & {fmt3(ranges['direct'])} & {fmt3(ranges['dist100'])} & {fmt3(ranges['ft150'])} \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\multicolumn{4}{l}{\scriptsize Bold = best trained model per joint.}")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------- VIII
def gen_table_consolidated(d):
    """Every trained-model cell (Direct / Dist. V2-100 / FT V2-150) is
    data-driven; the FT V2-150 column is bolded throughout, matching
    the source paper's fixed convention of always highlighting the
    proposed final model rather than an in-row argmax (see module
    docstring -- a real argmax would mis-bold the MPJPE row, since
    lower is better there but every other row is higher-is-better)."""
    direct, dist100, ft150, bench = d["direct_s"], d["dist100_s"], d["ft150_s"], d["bench"]
    teacher_fps = bench["teacher_fps"]
    teacher_params = r2(bench["teacher_params_m"])
    params = r2(bench["params_m"])
    gpu_fps = bench["student_fps"]

    def row(mp_text, teacher_text, direct_v, dist_v, ft_v, fmt):
        return (f"{mp_text} & {teacher_text} & {fmt(direct_v)} & {fmt(dist_v)}"
                f" & \\textbf{{{fmt(ft_v)}}} \\\\")

    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\caption{Consolidated Model Comparison}")
    lines.append(r"\label{tab:consolidated}")
    lines.append(r"\centering")
    lines.append(r"\renewcommand{\arraystretch}{1.1}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Criterion} & \textbf{MP SDK} & \textbf{Teacher} & \textbf{Direct} & \textbf{Dist.} & \textbf{FT V2} \\")
    lines.append(r"                   &                 & \textbf{HRNetV2} & \textbf{Sup.}   & \textbf{V2-100} & \textbf{-150} \\")
    lines.append(r"\midrule")

    lines.append("Params.         & " + row(
        f"{MP_SDK_PARAMS_M:.2f}M", f"{teacher_params:.2f}M",
        params, params, params, lambda v: f"{v:.2f}M"))
    lines.append("GPU FPS         & " + row(
        r"---$^{c}$", f"{teacher_fps:.1f}",
        gpu_fps, gpu_fps, gpu_fps, lambda v: f"{v:.0f}"))
    lines.append("PCK@0.2$^{a}$   & " + row(
        fmt3(MP_PRETRAIN["pck"]), fmt3(TEACHER_PCK) + "$^{b}$",
        r3(direct["PCK"]), r3(dist100["PCK"]), r3(ft150["PCK"]), fmt3))
    lines.append("AUC$^{a}$       & " + row(
        fmt3(MP_PRETRAIN["auc"]), fmt3(TEACHER_AUC) + "$^{b}$",
        r3(direct["AUC"]), r3(dist100["AUC"]), r3(ft150["AUC"]), fmt3))
    lines.append("MPJPE$^{a}$ (px)& " + row(
        fmt2(MP_PRETRAIN["mpjpe"]), fmt2(TEACHER_MPJPE) + "$^{b}$",
        r2(direct["MPJPE"]), r2(dist100["MPJPE"]), r2(ft150["MPJPE"]), fmt2))
    lines.append("Det.\\ rate      & " + row(
        pct(MP_PRETRAIN["det"]), pct(d["teacher_det"]),
        direct["DET"], dist100["DET"], ft150["DET"], pct))
    lines.append(r"RT (GPU)        & Yes$^{c}$ & No  & Yes   & Yes   & \textbf{Yes}   \\")
    lines.append(r"Open/retrain    & No    & Yes    & Yes   & Yes   & \textbf{Yes}   \\")
    lines.append(r"KD supervision  & No    & ---    & No    & Yes   & \textbf{Yes}   \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append("")
    lines.append(r"\vspace{3pt}")
    lines.append(r"\parbox{\columnwidth}{\scriptsize $^{b}$Official MMPose benchmark.}\\")
    lines.append(r"\parbox{\columnwidth}{\scriptsize $^{c}$Mobile GPU 5.3--16.1\,ms~\cite{mediapipe2020}; not comparable to RTX~4050.}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = load_all()
    tables = {
        "table_trained.tex": gen_table_trained(d),
        "table_pretrain_ref.tex": gen_table_pretrain_ref(d),
        "table_throughput.tex": gen_table_throughput(d),
        "table_perjoint.tex": gen_table_perjoint(d),
        "table_consolidated.tex": gen_table_consolidated(d),
    }
    for name, content in tables.items():
        (OUT_DIR / name).write_text(content, encoding="utf-8", newline="\n")
        print(f"wrote {OUT_DIR / name}")


if __name__ == "__main__":
    main()
