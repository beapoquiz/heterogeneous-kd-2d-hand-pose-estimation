import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv

BASE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(BASE, 'distillation_learning_curve.csv')
OUT  = os.path.join(BASE, 'thesis_figures', 'learning_curve_v2.png')
os.makedirs(os.path.join(BASE, 'thesis_figures'), exist_ok=True)

# ── Load CSV ───────────────────────────────────────────────
epochs, pck, mpjpe = [], [], []
with open(CSV, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row['Epoch']))
        pck.append(float(row['PCK@0.2']))
        mpjpe.append(float(row['MPJPE']))

epochs = np.array(epochs)
pck    = np.array(pck)
mpjpe  = np.array(mpjpe)

best_pck_idx   = np.argmax(pck)
best_pck_epoch = epochs[best_pck_idx]
final_epoch    = epochs[-1]

# ── Plot ───────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('white')

# PCK line (left axis)
color_pck = '#2196F3'
ax1.set_xlabel('Epoch', fontsize=13)
ax1.set_ylabel('PCK @ 0.2  (higher is better)', color=color_pck, fontsize=12)
l1, = ax1.plot(epochs, pck, color=color_pck, linewidth=2.2,
               marker='o', markersize=5, label='PCK@0.2')
ax1.tick_params(axis='y', labelcolor=color_pck)
ax1.set_ylim(0.70, 0.79)
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

# MPJPE line (right axis)
color_mpjpe = '#F44336'
ax2 = ax1.twinx()
ax2.set_ylabel('MPJPE  px  (lower is better)', color=color_mpjpe, fontsize=12)
l2, = ax2.plot(epochs, mpjpe, color=color_mpjpe, linewidth=2.2,
               marker='s', markersize=5, linestyle='--', label='MPJPE')
ax2.tick_params(axis='y', labelcolor=color_mpjpe)
ax2.set_ylim(33, 43)

# Best PCK marker
ax1.axvline(best_pck_epoch, color='#FF9800', linestyle=':', linewidth=1.6,
            label=f'Best PCK epoch ({best_pck_epoch})')
ax1.scatter([best_pck_epoch], [pck[best_pck_idx]], color='#FF9800',
            zorder=5, s=80)
ax1.annotate(f'PCK={pck[best_pck_idx]:.4f}\n(ep {best_pck_epoch})',
             xy=(best_pck_epoch, pck[best_pck_idx]),
             xytext=(best_pck_epoch + 2, pck[best_pck_idx] - 0.005),
             fontsize=8.5, color='#FF9800',
             arrowprops=dict(arrowstyle='->', color='#FF9800', lw=1.2))

# Final epoch marker
ax1.scatter([final_epoch], [pck[-1]], color=color_pck, zorder=5,
            s=80, edgecolors='black', linewidths=0.8)
ax1.annotate(f'ep {final_epoch}\nPCK={pck[-1]:.4f}',
             xy=(final_epoch, pck[-1]),
             xytext=(final_epoch - 18, pck[-1] + 0.003),
             fontsize=8.5, color=color_pck,
             arrowprops=dict(arrowstyle='->', color=color_pck, lw=1.2))

# Legend
lines  = [l1, l2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower right', fontsize=10, framealpha=0.9)

ax1.set_xticks(epochs)
ax1.set_xticklabels([str(e) for e in epochs], rotation=45, fontsize=9)
ax1.grid(axis='both', linestyle='--', alpha=0.4)

plt.title('Distilled v2 Student — Learning Curve (RHD Test, 200 samples/epoch)',
          fontsize=12, pad=12)
plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches='tight')
print(f'Saved: {OUT}')

# ── Print table ────────────────────────────────────────────
print()
print(f'{"Epoch":>7} {"PCK@0.2":>10} {"MPJPE (px)":>12}')
print('-' * 32)
for e, p, m in zip(epochs, pck, mpjpe):
    marker = ' ← best PCK' if e == best_pck_epoch else (
             ' ← final'   if e == final_epoch else '')
    print(f'{e:>7} {p:>10.4f} {m:>12.4f}{marker}')
