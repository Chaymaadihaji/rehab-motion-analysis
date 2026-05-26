import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider
import csv
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────────────
BASE     = Path(r"C:\Users\PC\rehab-motion-analysis\classification\KERAAL_clf_mc_CTK")
REP_CSV  = BASE / "fold0" / "run0" / "class_C" / "representatives.csv"
EXERCISE = "KERAAL_CLF_MC_CTK"
FPS      = 8

DARK_BG  = "#0d0d1a"
PANEL_BG = "#12122a"
ACCENT   = "#4ecdc4"
JOINT_COL= "#ff4444"

CHAINS = [
    [0, 1, 10, 2, 3],
    [10, 4, 5, 6],
    [10, 7, 8, 9],
]
CHAIN_COLORS = ["#ffffff", "#4ecdc4", "#f7b731"]

TRACE_STEP = 10   # 1 squelette toutes les 10 frames

# ── CHARGEMENT ─────────────────────────────────────────────────────────────────
X = np.concatenate([
    np.load(BASE / "fold0" / "run0" / "x_train_fold0.npy"),
    np.load(BASE / "fold0" / "run0" / "x_test_fold0.npy")
], axis=0)
print(f"  → {X.shape[0]} séquences | shape = {X.shape}")

clusters_reps = {}
with open(REP_CSV) as f:
    for row in csv.DictReader(f):
        k   = int(row["cluster"])
        idx = int(row["sequence_index"])
        clusters_reps.setdefault(k, []).append(idx)

N_CLUSTERS = len(clusters_reps)
reps = [clusters_reps[k][0] for k in sorted(clusters_reps)]
print(f"  → {N_CLUSTERS} clusters | représentants : {reps}")

COLORS = [matplotlib.colormaps['tab10'](i) for i in range(N_CLUSTERS)]

# ── SKELETON ───────────────────────────────────────────────────────────────────
def get_skeleton(seq_idx):
    raw  = X[seq_idx]
    n_fr = raw.shape[1]
    data = raw.reshape(11, 7, n_fr)
    skel = data.transpose(2, 0, 1)
    return skel[:, :, :2]

def pose_to_plot(pose):
    return pose[:, 0], pose[:, 1]

def compute_bounds(skel):
    pad = 0.08
    return (skel[:,:,0].min()-pad, skel[:,:,0].max()+pad,
            skel[:,:,1].min()-pad, skel[:,:,1].max()+pad)

# ── STATE ──────────────────────────────────────────────────────────────────────
state = dict(
    cluster=0, frame=0, playing=False,
    skel=get_skeleton(reps[0]), rep_idx=reps[0], cb_trace=None,
    _cid_ready=None,
)
state['bounds'] = compute_bounds(state['skel'])

# ── FIGURE ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({'figure.facecolor': DARK_BG, 'axes.facecolor': PANEL_BG,
                     'text.color': 'white'})
fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor(DARK_BG)

gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30,
                       left=0.04, right=0.97, bottom=0.15, top=0.91)
ax_anim  = fig.add_subplot(gs[0])
ax_trace = fig.add_subplot(gs[1])

for ax in (ax_anim, ax_trace):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values(): sp.set_edgecolor('#2a2a4a')
    ax.tick_params(colors='#666688', labelsize=7)

ax_anim.axis('off')
ax_anim.set_title('Squelette animé', fontsize=10, color='#ccccee', pad=6)
title_obj = fig.suptitle('', fontsize=12, fontweight='bold', color='white', y=0.97)

# ── WIDGETS ────────────────────────────────────────────────────────────────────
ax_sl = fig.add_axes([0.04, 0.065, 0.87, 0.020])
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
    b = Button(bax, f'Cluster {k}',
               color=matplotlib.colors.to_hex(COLORS[k]), hovercolor='#eeeeee')
    b.label.set_fontsize(8); b.label.set_color('white')
    btn_cls.append(b)

# ── FONCTIONS ──────────────────────────────────────────────────────────────────
def _silent_slider(val):
    slider.eventson = False
    slider.set_val(val)
    slider.eventson = True

def update_title():
    k, idx, n = state['cluster'], state['rep_idx'], state['skel'].shape[0]
    title_obj.set_text(f"Cluster {k}  —  Séquence #{idx}  |  {EXERCISE}  |  {n} frames")

def draw_frame(fi):
    skel = state['skel']
    x, y = pose_to_plot(skel[fi])

    ax_anim.clear()
    ax_anim.set_facecolor(PANEL_BG)
    ax_anim.set_xticks([]); ax_anim.set_yticks([])
    for sp in ax_anim.spines.values(): sp.set_visible(False)

    for ci, chain in enumerate(CHAINS):
        ax_anim.plot(x[chain], y[chain], color=CHAIN_COLORS[ci],
                     lw=2.5, solid_capstyle='round', zorder=3)

    used = sorted(set(sum(CHAINS, [])))
    ax_anim.scatter(x[used], y[used], color=JOINT_COL, s=40, zorder=5)

    ax_anim.text(0.02, 0.97, f'Frame {fi+1} / {skel.shape[0]}',
                 transform=ax_anim.transAxes, fontsize=9, color='#aaaaee', va='top')
    ax_anim.set_title('Squelette animé', fontsize=10, color='#ccccee', pad=6)
    ax_anim.set_xlim(state['bounds'][0], state['bounds'][1])
    ax_anim.set_ylim(state['bounds'][2], state['bounds'][3])
    # pas de invert_yaxis — orientation naturelle des données est correcte

    _silent_slider(fi)
    fig.canvas.draw_idle()

def redraw_trace():
    if state['cb_trace'] is not None:
        try: state['cb_trace'].remove()
        except: pass
        state['cb_trace'] = None

    ax_trace.cla()
    ax_trace.set_facecolor(PANEL_BG)
    for sp in ax_trace.spines.values(): sp.set_edgecolor('#2a2a4a')

    skel  = state['skel']
    n_fr  = skel.shape[0]
    cmap  = matplotlib.colormaps['Oranges']

    # Squelettes avec dégradé Oranges, espacés de TRACE_STEP frames
    for fi in range(0, n_fr, TRACE_STEP):
        t     = fi / max(n_fr - 1, 1)
        color = cmap(0.3 + 0.7 * t)   # clair au début → foncé à la fin
        alpha = 0.3 + 0.5 * t
        x, y  = pose_to_plot(skel[fi])
        for chain in CHAINS:
            ax_trace.plot(x[chain], y[chain],
                          color=color, lw=0.9, alpha=alpha, zorder=3)

    # Limites — même orientation que l'animation (pas de invert)
    xmin, xmax, ymin, ymax = state['bounds']
    ax_trace.set_xlim(xmin, xmax)
    ax_trace.set_ylim(ymin, ymax)
    ax_trace.set_aspect('equal', adjustable='box')

    ax_trace.set_title('Trace du mouvement', fontsize=10, color='#ccccee', pad=5)
    ax_trace.tick_params(colors='#666688', labelsize=7)

    sm = plt.cm.ScalarMappable(cmap='Oranges', norm=plt.Normalize(0, n_fr - 1))
    cb = fig.colorbar(sm, ax=ax_trace, fraction=0.035, pad=0.02)
    cb.ax.tick_params(colors='#aaaaaa', labelsize=7)
    cb.set_label('Frame', color='#aaaaaa', fontsize=8)
    state['cb_trace'] = cb

def switch_cluster(k):
    state['cluster'] = k
    state['rep_idx'] = reps[k]
    state['skel']    = get_skeleton(reps[k])
    state['frame']   = 0
    state['playing'] = True
    state['bounds']  = compute_bounds(state['skel'])
    btn_pp.label.set_text('⏸  Pause'); btn_pp.ax.set_facecolor('#1e3a5f')
    n_fr = state['skel'].shape[0]
    slider.valmax = n_fr-1; slider.ax.set_xlim(0, n_fr-1)
    _silent_slider(0)
    redraw_trace(); update_title(); draw_frame(0)

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
        n_fr = state['skel'].shape[0]
        state['frame'] = (state['frame']+1) % n_fr
        draw_frame(state['frame'])

timer = fig.canvas.new_timer(interval=int(1000/FPS))
timer.add_callback(tick)

def on_figure_ready(event):
    fig.canvas.mpl_disconnect(state['_cid_ready'])
    redraw_trace()
    update_title()
    draw_frame(0)
    timer.start()

state['_cid_ready'] = fig.canvas.mpl_connect('draw_event', on_figure_ready)

plt.show()