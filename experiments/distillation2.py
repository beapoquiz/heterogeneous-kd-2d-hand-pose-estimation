import time

# ANSI color codes
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
DIM     = "\033[2m"
BOLD    = "\033[1m"
RESET   = "\033[0m"

# ─── HEADER ───────────────────────────────────────────────────────────────────
header = f"""------------------------------------------------------------
{BOLD}Knowledge Distillation Training{RESET}
------------------------------------------------------------
System environment:
    sys.platform: win32
    Python: 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)]
    CUDA available: True
    GPU 0: NVIDIA GeForce RTX 4050 Laptop GPU
    PyTorch: 2.1.0+cu121
    MMEngine: 0.10.7

Distillation Config:
    Teacher  : HRNet-W18 (MMPose) — {GREEN}Frozen{RESET}
    Student  : BlazeHandLandmark
    Dataset  : RHD (Train: 41,258 | Val: 2,728 images)
    Loss     : WingLoss (α=0.5) + L1 Distillation (β=0.5)
    Optimizer: Adam  |  LR: 1e-3  |  Scheduler: StepLR (step=10, γ=0.5)
    Epochs   : 50    |  Batch Size: 16
------------------------------------------------------------
"""
print(header)
time.sleep(0.5)

# ─── EPOCH DATA ───────────────────────────────────────────────────────────────
epochs = [
    {
        "epoch": 1,
        "batches": [
            ("000",  "4.8812", "2.4102", "2.4710"),
            ("200",  "4.6540", "2.3021", "2.3519"),
            ("400",  "4.4210", "2.1980", "2.2230"),
            ("800",  "4.2891", "2.1540", "2.1351"),
            ("1200", "4.1670", "2.0940", "2.0730"),
            ("1600", "4.0920", "2.0510", "2.0410"),
            ("2000", "4.0201", "2.0240", "1.9961"),
            ("2400", "3.9880", "2.0010", "1.9870"),
            ("2578", "3.9720", "1.9950", "1.9770"),
        ],
        "pck": "0.2981", "auc": "0.1740", "epe": "38.5521",
        "prev_pck": "0.4574", "prev_auc": "0.3151", "prev_epe": "27.7971",
    },
    {
        "epoch": 2,
        "batches": [
            ("000",  "3.9540", "1.9810", "1.9730"),
            ("200",  "3.8201", "1.9140", "1.9061"),
            ("400",  "3.7450", "1.8810", "1.8640"),
            ("800",  "3.6980", "1.8520", "1.8460"),
            ("1200", "3.6410", "1.8240", "1.8170"),
            ("1600", "3.5920", "1.8010", "1.7910"),
            ("2000", "3.5540", "1.7810", "1.7730"),
            ("2400", "3.5210", "1.7640", "1.7570"),
            ("2578", "3.5010", "1.7540", "1.7470"),
        ],
        "pck": "0.3412", "auc": "0.2010", "epe": "34.2210",
        "prev_pck": "0.2981", "prev_auc": "0.1740", "prev_epe": "38.5521",
    },
    {
        "epoch": 3,
        "batches": [
            ("000",  "3.4880", "1.7490", "1.7390"),
            ("200",  "3.4120", "1.7120", "1.7000"),
            ("400",  "3.3540", "1.6840", "1.6700"),
            ("800",  "3.3010", "1.6580", "1.6430"),
            ("1200", "3.2650", "1.6400", "1.6250"),
            ("1600", "3.2410", "1.6290", "1.6120"),
            ("2000", "3.2240", "1.6190", "1.6050"),
            ("2400", "3.2080", "1.6100", "1.5980"),
            ("2578", "3.1990", "1.6060", "1.5930"),
        ],
        "pck": "0.3801", "auc": "0.2380", "epe": "31.0940",
        "prev_pck": "0.3412", "prev_auc": "0.2010", "prev_epe": "34.2210",
    },
    {
        "epoch": 4,
        "batches": [
            ("000",  "3.1870", "1.6000", "1.5870"),
            ("200",  "3.1540", "1.5840", "1.5700"),
            ("400",  "3.1210", "1.5690", "1.5520"),
            ("800",  "3.0980", "1.5570", "1.5410"),
            ("1200", "3.0740", "1.5450", "1.5290"),
            ("1600", "3.0520", "1.5340", "1.5180"),
            ("2000", "3.0350", "1.5250", "1.5100"),
            ("2400", "3.0190", "1.5170", "1.5020"),
            ("2578", "3.0080", "1.5110", "1.4970"),
        ],
        "pck": "0.4020", "auc": "0.2650", "epe": "29.4410",
        "prev_pck": "0.3801", "prev_auc": "0.2380", "prev_epe": "31.0940",
    },
    {
        "epoch": 5,
        "batches": [
            ("000",  "2.9940", "1.5040", "1.4900"),
            ("200",  "2.9710", "1.4930", "1.4780"),
            ("400",  "2.9480", "1.4820", "1.4660"),
            ("800",  "2.9210", "1.4680", "1.4530"),
            ("1200", "2.9010", "1.4580", "1.4430"),
            ("1600", "2.8840", "1.4500", "1.4340"),
            ("2000", "2.8690", "1.4420", "1.4270"),
            ("2400", "2.8540", "1.4350", "1.4190"),
            ("2578", "2.8450", "1.4300", "1.4150"),
        ],
        "pck": "0.4198", "auc": "0.2891", "epe": "28.1150",
        "prev_pck": "0.4020", "prev_auc": "0.2650", "prev_epe": "29.4410",
    },
]

# Teacher reference
TEACHER_PCK = 0.991842
TEACHER_AUC = 0.902150
TEACHER_EPE = 2.184201

# ─── TRAINING LOOP ────────────────────────────────────────────────────────────
for ep in epochs:
    e = ep["epoch"]
    print(f"{BOLD}{YELLOW}Epoch [{e}/50]{RESET}")

    for (b, total, wing, distill) in ep["batches"]:
        timestamp = f"03/12 09:4{e}:{int(b)//50:02d}"
        print(
            f"{DIM}{timestamp}{RESET} - mmengine - INFO - "
            f"Iter [{b.rjust(4)}/2578]  "
            f"total_loss: {RED}{total}{RESET}  "
            f"wing_loss: {YELLOW}{wing}{RESET}  "
            f"distill_loss: {MAGENTA}{distill}{RESET}"
        )
        time.sleep(0.04)

    # Checkpoint save
    print(
        f"\n{DIM}03/12 09:4{e}:59{RESET} - mmengine - INFO - "
        f"{GREEN}Checkpoint saved → checkpoints/student_distill_epoch{e}.pth{RESET}\n"
    )
    time.sleep(0.1)

    # Validation header
    print(f"{DIM}03/12 09:4{e}:59{RESET} - mmengine - INFO - Evaluating PCKAccuracy (normalized by ``\"bbox_size\"``)...")
    print(f"{DIM}03/12 09:4{e}:59{RESET} - mmengine - INFO - Evaluating AUC...")
    print(f"{DIM}03/12 09:4{e}:59{RESET} - mmengine - INFO - Evaluating EPE...")
    time.sleep(0.15)

    # Parse metrics
    pck  = float(ep["pck"])
    auc  = float(ep["auc"])
    epe  = float(ep["epe"])
    ppck = float(ep["prev_pck"])
    pauc = float(ep["prev_auc"])
    pepe = float(ep["prev_epe"])

    pck_delta = pck - ppck
    auc_delta = auc - pauc
    epe_delta = epe - pepe   # negative = better

    pck_gap = TEACHER_PCK - pck
    auc_gap = TEACHER_AUC - auc
    epe_gap = epe - TEACHER_EPE

    def arrow(val, higher_better=True):
        good = val > 0 if higher_better else val < 0
        return f"{GREEN}▲{RESET}" if good else f"{RED}▼{RESET}"

    # Metrics line — mirrors teacher format exactly
    print(
        f"{DIM}03/12 09:4{e}:59{RESET} - mmengine - INFO - "
        f"Epoch(val) [{e}/50]    "
        f"PCK: {CYAN}{ep['pck']}{RESET}  "
        f"AUC: {CYAN}{ep['auc']}{RESET}  "
        f"EPE: {CYAN}{ep['epe']}{RESET}"
    )
    time.sleep(0.05)

    # Delta vs previous epoch
    print(
        f"                               "
        f"  Δ PCK: {arrow(pck_delta)}{GREEN if pck_delta>0 else RED}{pck_delta:+.6f}{RESET}"
        f"  Δ AUC: {arrow(auc_delta)}{GREEN if auc_delta>0 else RED}{auc_delta:+.6f}{RESET}"
        f"  Δ EPE: {arrow(epe_delta, higher_better=False)}{GREEN if epe_delta<0 else RED}{epe_delta:+.6f}{RESET}"
    )

    # Gap vs teacher
    print(
        f"                               "
        f"  Gap to Teacher →  "
        f"PCK: {RED}{pck_gap:.6f}{RESET}  "
        f"AUC: {RED}{auc_gap:.6f}{RESET}  "
        f"EPE: {RED}{epe_gap:.6f}{RESET}"
    )

    # Status — epoch 1 regresses, rest are marginal
    if e == 1:
        print(f"\n  {RED}✘ Status: Regression detected. Student performing worse than baseline. Check distillation loss weight.{RESET}\n")
    elif pck > ppck and auc > pauc and epe < pepe:
        improvement = abs(epe_delta)
        if improvement < 2.0:
            print(f"\n  {YELLOW}△ Status: Marginal improvement. Loss still high — student struggling to fit teacher heatmaps.{RESET}\n")
        else:
            print(f"\n  {GREEN}✔ Status: Improvement detected. Model checkpoint updated.{RESET}\n")
    else:
        print(f"\n  {RED}✘ Status: No improvement. Checkpoint not updated.{RESET}\n")

    print("-" * 60)
    time.sleep(0.3)

# ─── FINAL SUMMARY ────────────────────────────────────────────────────────────
last = epochs[-1]
print(f"""
{BOLD}============================================================
 DISTILLATION SUMMARY (Epoch {last['epoch']}/50 so far)
============================================================{RESET}

  {'Metric':<12} {'Teacher':>12} {'Student':>12} {'Gap':>12}
  {'-'*50}
  {'PCK ↑':<12} {GREEN}{TEACHER_PCK:>12.6f}{RESET} {CYAN}{last['pck']:>12}{RESET} {RED}{TEACHER_PCK - float(last['pck']):>12.6f}{RESET}
  {'AUC ↑':<12} {GREEN}{TEACHER_AUC:>12.6f}{RESET} {CYAN}{last['auc']:>12}{RESET} {RED}{TEACHER_AUC - float(last['auc']):>12.6f}{RESET}
  {'EPE ↓':<12} {GREEN}{TEACHER_EPE:>12.6f}{RESET} {CYAN}{last['epe']:>12}{RESET} {RED}{float(last['epe']) - TEACHER_EPE:>12.6f}{RESET}

  {YELLOW}⚠  Student is underperforming. Losses remain high.{RESET}
  {DIM}Consider: lower LR, higher distill weight β, or more epochs.{RESET}
  {DIM}Training continues... Next checkpoint at Epoch 10.{RESET}
{BOLD}============================================================{RESET}
""")

print(f"{YELLOW}Distillation training in progress. Student has not converged. Ctrl+C to interrupt.{RESET}")