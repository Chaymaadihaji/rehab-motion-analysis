import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider
import csv
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
BASE       = Path(r"C:\Users\PC\rehab-motion-analysis\classification\KINECAL_clf_bn_GGFV\fold0\run0")
REP_CSV    = BASE / "class_0" / "representatives.csv"
EXERCISE   = "KINECAL_CLF_BN_GGFV"
FPS        = 8
N_SNAPSHOTS = 6   # nombre de snapshots dans la grille (2 lignes x 3 colonnes)

DARK_BG   = "#0d0d1a"
PANEL_BG  = "#12122a"
ACCENT    = "#4ecdc4"
JOINT_COL = "#ff4444"

# ══════════════════════════════════════════════════════════════════════
#  SQUELETTE — 25 joints NTU RGB+D
# ══════════════════════════════════════════════════════════════════════
CHAINS = [
    [3, 2, 1, 0],
    [4, 2, 8],
    [4, 5, 7],
    [8, 9, 10],
    [0, 12, 13, 14, 15],
    [0, 16, 17, 18, 19],
]
CHAIN_COLORS = [
    "#ffffff",
    "#ffffff",
    "#4ecdc4",
    "#f7b731",
    "#a29bfe",
    "#fd79a8",
]
USED_JOINTS = sorted(set(sum(CHAINS, [])))

# ══════════════════════════════════════════════════════════════════════
#  CHARGEMENT
# ══════════════════════════════════════════════════════════════════════
X = np.concatenate([
    np.load(BASE / "x_train_fold0.npy"),
    np.load(BASE / "x_test_fold0.npy"),
], axis=0)
print(f"  → {X.shape[0]} séquences  |  shape = {X.shape}")

clusters_reps = {}
with open(REP_CSV) as f:
    for row in csv.DictReader(f):
        k   = int(row["cluster"])
        idx = int(row["sequence_index"])
        clusters_reps.setdefault(k, []).append(idx)

N_CLUSTERS = len(clusters_reps)
reps = [clusters_reps[k][0] for k in sorted(clusters_reps)]
print(f"  → {N_CLUSTERS} clusters  |  représentants : {reps}")

COLORS = [matplotlib.colormaps['tab10'](i) for i in range(N_CLUSTERS)]

# ══════════════════════════════════════════════════════════════════════
#  SQUELETTE — extraction + interpolation + centrage pelvis
# ══════════════════════════════════════════════════════════════════════
def get_skeleton(seq_idx):
    raw  = X[seq_idx]
    n_fr = raw.shape[1]
    data = raw.reshape(25, 3, n_fr)
    skel = data.transpose(2, 0, 1)
    xy   = skel[:, :, :2].copy()

    for j in range(xy.shape[1]):
        traj  = xy[:, j, :]
        vel   = np.abs(np.diff(traj, axis=0))
        std   = np.std(vel)
        seuil = max(0.05, min(0.3, 3 * std))
        fi = 1
        while fi < n_fr - 1:
            if vel[fi-1].max() > seuil:
                fj = fi + 1
                while fj < n_fr - 1 and vel[fj-1].max() > seuil:
                    fj += 1
                for fk in range(fi, fj):
                    t = (fk - fi + 1) / (fj - fi + 1)
                    xy[fk, j, :] = (1-t) * xy[fi-1, j, :] + t * xy[fj, j, :]
                fi = fj
            else:
                fi += 1

    pelvis_median = np.median(xy[:, 0:1, :], axis=0, keepdims=True)
    xy = xy - pelvis_median
    return xy

def pose_to_plot(pose):
    return pose[:, 0], pose[:, 1]

def compute_bounds(skel):
    pad    = 0.15
    x_vals = skel[:, :, 0].flatten()
    y_vals = skel[:, :, 1].flatten()
    return (
        np.percentile(x_vals,  1) - pad,
        np.percentile(x_vals, 99) + pad,
        np.percentile(y_vals,  1) - pad,
        np.percentile(y_vals, 99) + pad,
    )

# ══════════════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════════════
state = dict(
    cluster=0, frame=0, playing=False,
    skel=get_skeleton(reps[0]), rep_idx=reps[0],
    _cid_ready=None,
)
state['bounds'] = compute_bounds(state['skel'])

# ══════════════════════════════════════════════════════════════════════
#  FIGURE — layout : animation gauche | grille snapshots droite
# ══════════════════════════════════════════════════════════════════════
plt.rcParams.update({'figure.facecolor': DARK_BG, 'axes.facecolor': PANEL_BG,
                     'text.color': 'white'})
fig = plt.figure(figsize=(16, 8))
fig.patch.set_facecolor(DARK_BG)

# Colonne gauche : animation | Colonne droite : grille 2x3
outer_gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25,
                              left=0.03, right=0.97, bottom=0.15, top=0.91)
ax_anim = fig.add_subplot(outer_gs[0])
ax_anim.set_facecolor(PANEL_BG)
for sp in ax_anim.spines.values(): sp.set_edgecolor('#2a2a4a')
ax_anim.tick_params(colors='#666688', labelsize=7)
ax_anim.axis('off')
ax_anim.set_title('Squelette animé', fontsize=10, color='#ccccee', pad=6)

# Grille 2 lignes x 3 colonnes pour les snapshots
inner_gs = gridspec.GridSpecFromSubplotSpec(
    2, 3, subplot_spec=outer_gs[1], hspace=0.35, wspace=0.15)
ax_snaps = [fig.add_subplot(inner_gs[r, c])
            for r in range(2) for c in range(3)]
for ax in ax_snaps:
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values(): sp.set_edgecolor('#2a2a4a')
    ax.set_xticks([]); ax.set_yticks([])

title_obj = fig.suptitle('', fontsize=12, fontweight='bold', color='white', y=0.97)

# ── Widgets ───────────────────────────────────────────────────────────
ax_sl  = fig.add_axes([0.04, 0.065, 0.87, 0.020])
slider = Slider(ax_sl, 'Frame', 0, state['skel'].shape[0]-1,
                valinit=0, valstep=1, color=ACCENT)
slider.label.set_color('white'); slider.valtext.set_color(ACCENT)

ax_pp = fig.add_axes([0.73, 0.01, 0.08, 0.042])
btn_pp = Button(ax_pp, '▶  Lecture', color='#1e3a5f', hovercolor='#2a5080')
btn_pp.label.set_color('white'); btn_pp.label.set_fontsize(10)

ax_rs = fig.add_axes([0.82, 0.01, 0.08, 0.042])
btn_rs = Button(ax_rs, '⏮  Début', color='#2a2a4a', hovercolor='#3a3a6a')
btn_rs.label.set_color('white'); btn_rs.label.set_fontsize(10)

btn_cls = []
bw, bh = 0.09, 0.042
for k in range(N_CLUSTERS):
    bax = fig.add_axes([0.04 + k*(bw+0.005), 0.01, bw, bh])
    b   = Button(bax, f'Cluster {k}',
                 color=matplotlib.colors.to_hex(COLORS[k]), hovercolor='#eeeeee')
    b.label.set_fontsize(8); b.label.set_color('white')
    btn_cls.append(b)

# ══════════════════════════════════════════════════════════════════════
#  FONCTIONS
# ══════════════════════════════════════════════════════════════════════
def _silent_slider(val):
    slider.eventson = False
    slider.set_val(val)
    slider.eventson = True

def update_title():
    k, idx, n = state['cluster'], state['rep_idx'], state['skel'].shape[0]
    title_obj.set_text(
        f"Cluster {k}  —  Séquence #{idx}  |  {EXERCISE}  |  {n} frames")

def draw_skeleton_on(ax, skel, fi, bounds, highlight=False):
    """Dessine le squelette de la frame fi sur l'axe ax."""
    x, y = pose_to_plot(skel[fi])
    for ci, chain in enumerate(CHAINS):
        lw = 2.0 if highlight else 1.4
        ax.plot(x[chain], y[chain],
                color=CHAIN_COLORS[ci], lw=lw,
                solid_capstyle='round', solid_joinstyle='round', zorder=3)
    s = 12 if highlight else 8
    ax.scatter(x[USED_JOINTS], y[USED_JOINTS], color=JOINT_COL, s=s, zorder=5)
    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')

def draw_frame(fi):
    skel = state['skel']
    x, y = pose_to_plot(skel[fi])

    ax_anim.clear()
    ax_anim.set_facecolor(PANEL_BG)
    ax_anim.set_xticks([]); ax_anim.set_yticks([])
    for sp in ax_anim.spines.values(): sp.set_visible(False)
    ax_anim.autoscale(False)

    for ci, chain in enumerate(CHAINS):
        ax_anim.plot(x[chain], y[chain],
                     color=CHAIN_COLORS[ci], lw=2.5,
                     solid_capstyle='round', solid_joinstyle='round', zorder=3)
    ax_anim.scatter(x[USED_JOINTS], y[USED_JOINTS],
                    color=JOINT_COL, s=40, zorder=5)
    ax_anim.text(0.02, 0.97, f'Frame {fi+1} / {skel.shape[0]}',
                 transform=ax_anim.transAxes, fontsize=9,
                 color='#aaaaee', va='top')
    ax_anim.set_title('Squelette animé', fontsize=10, color='#ccccee', pad=6)

    xmin, xmax, ymin, ymax = state['bounds']
    ax_anim.set_xlim(xmin, xmax)
    ax_anim.set_ylim(ymin, ymax)

    _silent_slider(fi)
    fig.canvas.draw_idle()

def redraw_snapshots():
    """Remplit la grille avec N_SNAPSHOTS frames clés de la séquence."""
    skel   = state['skel']
    n_fr   = skel.shape[0]
    bounds = state['bounds']
    cmap   = matplotlib.colormaps['Oranges']

    # Frames clés : on cherche la frame la plus STABLE autour de chaque instant cible
    # "stable" = faible vélocité = posture nette, pas de transition bruitée
    vel = np.sqrt(((np.diff(skel, axis=0))**2).sum(axis=(1, 2)))  # (n_fr-1,)
    stability = np.convolve(vel, np.ones(5)/5, mode='same')       # lissage fenêtre 5

    snap_indices = []
    for i in range(N_SNAPSHOTS):
        center = int(round(i * (n_fr - 1) / (N_SNAPSHOTS - 1)))
        win = max(5, n_fr // 15)          # fenêtre ±10% autour de l'instant cible
        lo  = max(0, center - win)
        hi  = min(n_fr - 1, center + win)
        best = lo + int(np.argmin(stability[lo:hi+1]))
        snap_indices.append(best)

    for i, ax in enumerate(ax_snaps):
        ax.clear()
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values(): sp.set_edgecolor('#2a2a4a')
        ax.set_xticks([]); ax.set_yticks([])

        fi = snap_indices[i]
        t  = fi / max(n_fr - 1, 1)

        # Bordure colorée selon la progression temporelle
        border_color = cmap(0.35 + 0.65 * t)
        for sp in ax.spines.values():
            sp.set_edgecolor(border_color)
            sp.set_linewidth(2.0)

        draw_skeleton_on(ax, skel, fi, bounds, highlight=True)

        # Label frame + pourcentage
        pct = int(round(100 * t))
        ax.set_title(f'Frame {fi+1}  ({pct}%)',
                     fontsize=7, color=matplotlib.colors.to_hex(border_color),
                     pad=3, fontweight='bold')

    fig.canvas.draw_idle()

# ══════════════════════════════════════════════════════════════════════
#  INTERACTIONS
# ══════════════════════════════════════════════════════════════════════
def switch_cluster(k):
    state['cluster'] = k
    state['rep_idx'] = reps[k]
    state['skel']    = get_skeleton(reps[k])
    state['frame']   = 0
    state['playing'] = True
    state['bounds']  = compute_bounds(state['skel'])
    btn_pp.label.set_text('⏸  Pause'); btn_pp.ax.set_facecolor('#1e3a5f')
    n_fr = state['skel'].shape[0]
    slider.valmax = n_fr - 1; slider.ax.set_xlim(0, n_fr - 1)
    _silent_slider(0)
    redraw_snapshots(); update_title(); draw_frame(0)

def on_slider(val):
    state['frame'] = int(slider.val); draw_frame(state['frame'])
slider.on_changed(on_slider)

def toggle_play(event):
    state['playing'] = not state['playing']
    if state['playing']:
        btn_pp.label.set_text('⏸  Pause'); btn_pp.ax.set_facecolor('#1e3a5f')
    else:
        btn_pp.label.set_text('▶  Lecture'); btn_pp.ax.set_facecolor('#2a2a4a')
    fig.canvas.draw_idle()
btn_pp.on_clicked(toggle_play)

def restart(event):
    state['frame'] = 0; state['playing'] = True
    btn_pp.label.set_text('⏸  Pause'); btn_pp.ax.set_facecolor('#1e3a5f')
    draw_frame(0)
btn_rs.on_clicked(restart)

for k, b in enumerate(btn_cls):
    def _cb(event, ki=k): switch_cluster(ki)
    b.on_clicked(_cb)

def tick():
    if state['playing']:
        n_fr  = state['skel'].shape[0]
        xmin, xmax, ymin, ymax = state['bounds']
        for _ in range(n_fr):
            state['frame'] = (state['frame'] + 1) % n_fr
            x, y = pose_to_plot(state['skel'][state['frame']])
            if not (x.min() < xmin - 0.5 or x.max() > xmax + 0.5 or
                    y.min() < ymin - 0.5 or y.max() > ymax + 0.5):
                break
        draw_frame(state['frame'])

timer = fig.canvas.new_timer(interval=int(1000 / FPS))
timer.add_callback(tick)

def on_figure_ready(event):
    fig.canvas.mpl_disconnect(state['_cid_ready'])
    redraw_snapshots(); update_title(); draw_frame(0)
    timer.start()

state['_cid_ready'] = fig.canvas.mpl_connect('draw_event', on_figure_ready)
plt.show()