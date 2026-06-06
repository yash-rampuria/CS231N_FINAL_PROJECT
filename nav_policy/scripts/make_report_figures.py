"""Generate all figures for the V-LEAD report."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('.')
FIGDIR = ROOT / 'report' / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 150,
})

# ---------------------------------------------------------------------------
# Figure 1: training/validation curves (15-epoch BC run)
# ---------------------------------------------------------------------------
log_path = ROOT / 'nav_policy' / 'data' / 'checkpoints_bc' / 'log.csv'
rows = []
with open(log_path) as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append({k: float(v) for k, v in row.items()})

ep = np.array([int(r['epoch']) for r in rows])
train_loss = np.array([r['train_loss_total'] for r in rows])
val_loss   = np.array([r['val_loss_total']   for r in rows])
val_lin    = np.array([r['val_mse_lin_vel']  for r in rows])
val_psi    = np.array([r['val_mse_psi_dot']  for r in rows])
val_vx     = np.array([r['val_mse_vx']       for r in rows])
val_vy     = np.array([r['val_mse_vy']       for r in rows])
val_vz     = np.array([r['val_mse_vz']       for r in rows])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))

ax1.plot(ep, train_loss, marker='o', ms=3, lw=1.2, label='train loss')
ax1.plot(ep, val_loss,   marker='s', ms=3, lw=1.2, label='val loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (z-scored MSE + smoothness)')
ax1.set_title('Training and validation loss')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

ax2.plot(ep, val_vx,  marker='o', ms=3, lw=1.0, label=r'$v_x$ (m$^2$/s$^2$)')
ax2.plot(ep, val_vy,  marker='s', ms=3, lw=1.0, label=r'$v_y$ (m$^2$/s$^2$)')
ax2.plot(ep, val_vz,  marker='^', ms=3, lw=1.0, label=r'$v_z$ (m$^2$/s$^2$)')
ax2.plot(ep, val_psi, marker='d', ms=3, lw=1.0, label=r'$\dot\psi$ (rad$^2$/s$^2$)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation MSE (physical units)')
ax2.set_title('Per-component validation MSE')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', ncol=2)

fig.tight_layout()
fig.savefig(FIGDIR / 'training_curves.pdf', bbox_inches='tight')
fig.savefig(FIGDIR / 'training_curves.png', bbox_inches='tight')
plt.close(fig)
print('wrote training_curves.{pdf,png}')

# ---------------------------------------------------------------------------
# Figure 2: per-horizon-step RMSE
# ---------------------------------------------------------------------------
ph_bc = list(csv.DictReader(open(ROOT / 'nav_policy/data/eval/bc_full/per_horizon.csv')))
ph_da = list(csv.DictReader(open(ROOT / 'nav_policy/data/eval/dagger_r2_mpc_offline/per_horizon.csv')))
hs = np.array([int(r['horizon_step']) for r in ph_bc])

fig, axes = plt.subplots(1, 4, figsize=(8.5, 2.1), sharex=True)
labels = [
    (r'$v_x$ RMSE (m/s)', 'rmse_vx'),
    (r'$v_y$ RMSE (m/s)', 'rmse_vy'),
    (r'$v_z$ RMSE (m/s)', 'rmse_vz'),
    (r'$\dot\psi$ RMSE (rad/s)', 'rmse_psi_dot'),
]
for ax, (title, key) in zip(axes, labels):
    bc  = np.array([float(r[key]) for r in ph_bc])
    da  = np.array([float(r[key]) for r in ph_da])
    ax.plot(hs, bc, marker='o', ms=3.5, lw=1.2, label='BC')
    ax.plot(hs, da, marker='s', ms=3.5, lw=1.2, label='BC + DAgger')
    ax.set_title(title)
    ax.set_xlabel('Horizon step $i$')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0, 2, 4, 6, 8])
axes[0].set_ylabel('Validation RMSE')
axes[0].legend(loc='upper right')
fig.tight_layout()
fig.savefig(FIGDIR / 'horizon_rmse.pdf', bbox_inches='tight')
fig.savefig(FIGDIR / 'horizon_rmse.png', bbox_inches='tight')
plt.close(fig)
print('wrote horizon_rmse.{pdf,png}')

# ---------------------------------------------------------------------------
# Figure 3: top-down trajectory overlays + per-step position error
# ---------------------------------------------------------------------------
roots = [
    ('backroom_00_sub0',  'Backroom sub0'),
    ('backroom_00_sub2',  'Backroom sub2'),
    ('packardpark_00_sub0', 'PackardPark sub0'),
]
fig, axes = plt.subplots(2, 3, figsize=(8.5, 4.6))
for j, (n, title) in enumerate(roots):
    base = ROOT / 'nav_policy' / 'data' / 'eval' / 'bc_v1' / f'rollout_{n}'
    pol = np.load(base / 'trajectory.npz')
    exp = np.load(base / 'expert_reference.npz')
    Ppol = pol['Xro'][0:3, :]
    Pexp = exp['Xro'][0:3, :]
    Tpol = pol['Tro']
    npts = min(Ppol.shape[1], Pexp.shape[1])
    Ppol = Ppol[:, :npts]
    Pexp = Pexp[:, :npts]
    Tpol = Tpol[:npts]

    ax = axes[0, j]
    ax.plot(Pexp[0], Pexp[1], 'k--', lw=1.4, label='expert')
    ax.plot(Ppol[0], Ppol[1], 'C0-', lw=1.4, label='policy')
    ax.scatter([Pexp[0, 0]], [Pexp[1, 0]], c='g', s=22, zorder=5, label='start')
    ax.scatter([Pexp[0, -1]], [Pexp[1, -1]], c='r', s=22, marker='*', zorder=5, label='goal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    if j == 0:
        ax.legend(loc='best', fontsize=7)

    err = np.linalg.norm(Ppol - Pexp, axis=0)
    ax2 = axes[1, j]
    ax2.plot(Tpol - Tpol[0], err, 'C3-', lw=1.2)
    ax2.axhline(0.5, color='k', ls=':', lw=0.8, label='success thresh.')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('||p_pol - p_exp|| (m)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(0.6, err.max() * 1.1))
    if j == 0:
        ax2.legend(loc='upper left', fontsize=7)

fig.tight_layout()
fig.savefig(FIGDIR / 'trajectory_overlay.pdf', bbox_inches='tight')
fig.savefig(FIGDIR / 'trajectory_overlay.png', bbox_inches='tight')
plt.close(fig)
print('wrote trajectory_overlay.{pdf,png}')

# ---------------------------------------------------------------------------
# Figure 4: expert vs policy speed profile (single panel per rollout)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.4), sharey=True)
for j, (n, title) in enumerate(roots):
    base = ROOT / 'nav_policy' / 'data' / 'eval' / 'bc_v1' / f'rollout_{n}'
    pol = np.load(base / 'trajectory.npz')
    exp = np.load(base / 'expert_reference.npz')
    Vpol = pol['Xro'][3:6, :]
    Vexp = exp['Xro'][3:6, :]
    Tpol = pol['Tro']
    npts = min(Vpol.shape[1], Vexp.shape[1])
    spd_pol = np.linalg.norm(Vpol[:, :npts], axis=0)
    spd_exp = np.linalg.norm(Vexp[:, :npts], axis=0)
    t = Tpol[:npts] - Tpol[0]

    ax = axes[j]
    ax.plot(t, spd_exp, 'k--', lw=1.4, label='expert')
    ax.plot(t, spd_pol, 'C0-', lw=1.4, label='policy')
    ax.set_title(title)
    ax.set_xlabel('time (s)')
    ax.grid(True, alpha=0.3)
    if j == 0:
        ax.set_ylabel('speed (m/s)')
        ax.legend(loc='best', fontsize=7)
fig.tight_layout()
fig.savefig(FIGDIR / 'speed_profile.pdf', bbox_inches='tight')
fig.savefig(FIGDIR / 'speed_profile.png', bbox_inches='tight')
plt.close(fig)
print('wrote speed_profile.{pdf,png}')

print('all figures written to', FIGDIR)
