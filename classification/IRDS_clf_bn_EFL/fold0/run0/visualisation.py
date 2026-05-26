import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter
import csv
from pathlib import Path

# ============================================================
# CONFIG — IRDS_clf_bn_EFL
# ============================================================
BASE          = Path(r"C:\Users\PC\rehab-motion-analysis\classification\IRDS_clf_bn_EFL\fold0\run0")
EXERCISE_NAME = "IRDS_clf_bn_EFL"
FPS           = 22
VIDEO_DURATION = 10
TRACE_STEP    = 3   # 1 squelette toutes les 3 frames

DARK_BG   = "#0d0d1a"
PANEL_BG  = "#12122a"
JOINT_COL = "#ff5555"
CHAIN_COLORS = ["#ffffff", "#4ecdc4", "#f7b731", "#a29bfe", "#fd79a8"]

CHAINS = [
    [0, 1, 20, 2, 3],
    [20, 4, 5, 6, 7],
    [20, 8, 9, 10, 11],
    [0, 12, 13, 14, 15],
    [0, 16, 17, 18, 19],
]

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor':   PANEL_BG,
    'text.color':       'white'
})

fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor(DARK_BG)

gs = gridspec.GridSpec(1, 2, figure=fig,
                       width_ratios=[3.0, 1.1], wspace=0.15,
                       left=0.03, right=0.98, bottom=0.05, top=0.92)
ax_anim  = fig.add_subplot(gs[0])
ax_trace = fig.add_subplot(gs[1])

state = dict(skel=None, bounds=None, cb_trace=None,
             class_id="", cluster=0, rep_idx=0)

title_obj = fig.suptitle("", fontsize=20, fontweight='bold', color='white', y=0.97)

def get_skeleton(X, seq_idx):
    raw  = X[seq_idx]
    n_fr = raw.shape[1]
    skel = raw.reshape(25, 3, n_fr).transpose(2, 0, 1)
    return skel[:, :, :2]

def pose_to_plot(pose):
    return pose[:, 0], pose[:, 1]

def compute_bounds(skel):
    pad = 0.08
    return (skel[:,:,0].min()-pad, skel[:,:,0].max()+pad,
            skel[:,:,1].min()-pad, skel[:,:,1].max()+pad)

def draw_frame(fi):
    skel = state["skel"]
    x, y = pose_to_plot(skel[fi])

    ax_anim.clear()
    ax_anim.set_facecolor(PANEL_BG)
    ax_anim.set_xticks([]); ax_anim.set_yticks([])
    for sp in ax_anim.spines.values(): sp.set_visible(False)

    for ci, chain in enumerate(CHAINS):
        ax_anim.plot(x[chain], y[chain], color=CHAIN_COLORS[ci],
                     lw=3.5, solid_capstyle='round')
    used = sorted(set(sum(CHAINS, [])))
    ax_anim.scatter(x[used], y[used], color=JOINT_COL, s=75)

    ax_anim.text(0.02, 0.97, f"Frame {fi+1} / {skel.shape[0]}",
                 transform=ax_anim.transAxes, fontsize=12,
                 color="#ccccff", va='top')
    ax_anim.set_title("Squelette animé", fontsize=14, color="#ddddff")
    ax_anim.set_xlim(state["bounds"][0], state["bounds"][1])
    ax_anim.set_ylim(state["bounds"][2], state["bounds"][3])
    return []

def redraw_trace():
    if state["cb_trace"] is not None:
        try: state["cb_trace"].remove()
        except Exception: pass
        state["cb_trace"] = None

    for ax in fig.axes:
        if ax is not ax_anim and ax is not ax_trace:
            try: ax.remove()
            except Exception: pass

    ax_trace.clear()
    ax_trace.set_facecolor(PANEL_BG)
    for sp in ax_trace.spines.values(): sp.set_edgecolor('#2a2a4a')

    skel  = state["skel"]
    n_fr  = skel.shape[0]
    cmap  = matplotlib.colormaps["Oranges"]

    # Squelette complet avec dégradé — step=3
    for fi in range(0, n_fr, TRACE_STEP):
        t     = fi / max(n_fr - 1, 1)
        color = cmap(0.3 + 0.7 * t)
        alpha = 0.3 + 0.5 * t
        x, y  = pose_to_plot(skel[fi])
        for chain in CHAINS:
            ax_trace.plot(x[chain], y[chain],
                          color=color, lw=0.9, alpha=alpha)

    # Mêmes bounds que l'animation
    xmin, xmax, ymin, ymax = state["bounds"]
    ax_trace.set_xlim(xmin, xmax)
    ax_trace.set_ylim(ymin, ymax)
    ax_trace.set_aspect('equal', adjustable='box')
    ax_trace.set_title("Trace du mouvement", fontsize=14, color="#ddddff")
    ax_trace.tick_params(colors='#666688', labelsize=7)

    sm = plt.cm.ScalarMappable(cmap='Oranges', norm=plt.Normalize(0, n_fr-1))
    cb = fig.colorbar(sm, ax=ax_trace, fraction=0.035, pad=0.02,
                      use_gridspec=False)
    cb.ax.tick_params(colors='#aaaaaa', labelsize=7)
    cb.set_label('Frame', color='#aaaaaa', fontsize=8)
    state["cb_trace"] = cb

def generate_video(X, class_id, cluster_id, rep_number, rep_idx, output_dir):
    print(f"    🎥 classe {class_id} | cluster {cluster_id} | rep {rep_number} | seq #{rep_idx}")

    state["class_id"]  = class_id
    state["cluster"]   = cluster_id
    state["rep_idx"]   = rep_idx
    state["skel"]      = get_skeleton(X, rep_idx)
    state["bounds"]    = compute_bounds(state["skel"])
    state["cb_trace"]  = None

    title_obj.set_text(
        f"Séquence #{rep_idx}  |  {EXERCISE_NAME}"
        f"  |  Classe {class_id}  |  Cluster {cluster_id}")

    redraw_trace()

    original_frames = state["skel"].shape[0]
    n_frames = FPS * VIDEO_DURATION

    def update(vfi):
        real_idx = min(int(vfi * original_frames / n_frames), original_frames - 1)
        return draw_frame(real_idx)

    ani = FuncAnimation(fig, update,
                        frames=n_frames, interval=1000/FPS, repeat=False)

    out = output_dir / f"{EXERCISE_NAME}_{class_id}_{cluster_id}_{rep_number}_{rep_idx}.mp4"
    writer = FFMpegWriter(fps=FPS, bitrate=3000)
    ani.save(out, writer=writer, dpi=180)
    print(f"    ✅ {out.name}")

# ============================================================
# MAIN
# ============================================================
X = np.concatenate([
    np.load(BASE / "x_train_fold0.npy"),
    np.load(BASE / "x_test_fold0.npy"),
], axis=0)
print(f"  → {X.shape[0]} séquences | shape = {X.shape}")

output_dir = BASE / "videos"
output_dir.mkdir(parents=True, exist_ok=True)

class_dirs = sorted(BASE.glob("class_*"))
print(f"  → {len(class_dirs)} classe(s)\n")

for class_dir in class_dirs:
    class_id = class_dir.name.split("_")[-1]
    rep_csv  = class_dir / "representatives.csv"
    if not rep_csv.exists():
        print(f"  ❌ representatives.csv absent")
        continue

    clusters_reps = {}
    with open(rep_csv) as f:
        for row in csv.DictReader(f):
            k   = int(row["cluster"])
            idx = int(row["sequence_index"])
            clusters_reps.setdefault(k, []).append(idx)

    n_total = sum(len(v) for v in clusters_reps.values())
    print(f"  Classe {class_id} : {len(clusters_reps)} clusters | {n_total} séquences")

    for cluster_id in sorted(clusters_reps):
        rep_list = clusters_reps[cluster_id]
        print(f"    Cluster {cluster_id} : {len(rep_list)} séquence(s)")
        for rep_number, rep_idx in enumerate(rep_list):
            generate_video(X, class_id, cluster_id, rep_number, rep_idx, output_dir)

print("\n✅ Toutes les vidéos EFL générées.")