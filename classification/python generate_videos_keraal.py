import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter
import csv
from pathlib import Path

# ============================================================
# GLOBAL CONFIG
# ============================================================
ROOT = Path(__file__).resolve().parent
CLASSIFICATION_DIR = ROOT
FPS = 22
VIDEO_DURATION = 10

# ============================================================
# COLORS
# ============================================================
DARK_BG   = "#0d0d1a"
PANEL_BG  = "#12122a"
JOINT_COL = "#ff5555"

# ============================================================
# KERAAL CHAINS — 11 joints
# 0=bassin, 1=torse, 10=épaules
# 2=épaule G, 4=coude G, 5=poignet G, 6=main G
# 3=épaule D, 7=coude D, 8=poignet D, 9=main D  (à vérifier)
# ============================================================
CHAINS = [
    [0, 1, 10, 2, 3],   # colonne + épaules
    [10, 4, 5, 6],      # bras gauche
    [10, 7, 8, 9],      # bras droit
]
CHAIN_COLORS = ["#ffffff", "#4ecdc4", "#f7b731"]
TRACE_STEP   = 15

# ============================================================
# FIGURE
# ============================================================
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

# ============================================================
# STATE
# ============================================================
state = dict(cluster=0, rep_idx=0, skel=None, bounds=None,
             cb_trace=None, exercise_name="", class_id="")

title_obj = fig.suptitle("", fontsize=18, fontweight='bold', color='white', y=0.97)

def update_title():
    title_obj.set_text(
        f"Séquence #{state['rep_idx']}  |  {state['exercise_name']}"
        f"  |  Classe {state['class_id']}  |  Cluster {state['cluster']}")

# ============================================================
# SKELETON — 11 joints x 7 dims
# ============================================================
def get_skeleton(X, seq_idx):
    raw  = X[seq_idx]          # (77, n_frames)
    n_fr = raw.shape[1]
    data = raw.reshape(11, 7, n_fr)
    skel = data.transpose(2, 0, 1)
    xy   = skel[:, :, :2].copy()

    # Interpolation frames aberrantes
    SEUIL = 0.15
    fi = 1
    while fi < n_fr - 1:
        diff = np.abs(xy[fi] - xy[fi-1]).max()
        if diff > SEUIL:
            fj = fi + 1
            while fj < n_fr - 1:
                if np.abs(xy[fj] - xy[fi-1]).max() <= SEUIL:
                    break
                fj += 1
            for fk in range(fi, fj):
                t = (fk - fi + 1) / (fj - fi + 1)
                xy[fk] = (1-t) * xy[fi-1] + t * xy[fj]
            fi = fj
        else:
            fi += 1
    return xy

def pose_to_plot(pose):
    return pose[:, 0], pose[:, 1]

def compute_bounds(skel):
    pad    = 0.1
    x_vals = skel[:, :, 0].flatten()
    y_vals = skel[:, :, 1].flatten()
    return (
        np.percentile(x_vals,  1) - pad,
        np.percentile(x_vals, 99) + pad,
        np.percentile(y_vals,  1) - pad,
        np.percentile(y_vals, 99) + pad,
    )

# ============================================================
# DRAW FRAME
# ============================================================
def draw_frame(fi):
    skel = state["skel"]
    x, y = pose_to_plot(skel[fi])

    ax_anim.clear()
    ax_anim.set_facecolor(PANEL_BG)
    ax_anim.set_xticks([]); ax_anim.set_yticks([])
    for sp in ax_anim.spines.values(): sp.set_visible(False)

    used = sorted(set(sum(CHAINS, [])))
    for ci, chain in enumerate(CHAINS):
        ax_anim.plot(x[chain], y[chain], color=CHAIN_COLORS[ci],
                     lw=3.5, solid_capstyle='round')
    ax_anim.scatter(x[used], y[used], color=JOINT_COL, s=75)

    ax_anim.text(0.02, 0.97, f"Frame {fi+1} / {skel.shape[0]}",
                 transform=ax_anim.transAxes, fontsize=12,
                 color="#ccccff", va='top')
    ax_anim.set_title("Squelette animé", fontsize=14, color="#ddddff")
    ax_anim.set_xlim(state["bounds"][0], state["bounds"][1])
    ax_anim.set_ylim(state["bounds"][2], state["bounds"][3])
    return []

# ============================================================
# TRACE
# ============================================================
def redraw_trace():
    # Supprimer ancienne colorbar + orphelines
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

    skel  = state["skel"]
    n_fr  = skel.shape[0]
    cmap  = matplotlib.colormaps["Oranges"]
    xmin, xmax, ymin, ymax = state["bounds"]

    # Squelettes fantômes
    for fi in range(0, n_fr, TRACE_STEP):
        t     = fi / max(n_fr - 1, 1)
        color = cmap(0.3 + 0.7 * t)
        alpha = 0.2 + 0.5 * t
        x, y  = pose_to_plot(skel[fi])
        if (np.any(x < xmin) or np.any(x > xmax) or
            np.any(y < ymin) or np.any(y > ymax)):
            continue
        for chain in CHAINS:
            ax_trace.plot(x[chain], y[chain],
                          color=color, lw=1.2, alpha=alpha)

    ax_trace.set_xlim(xmin, xmax)
    ax_trace.set_ylim(ymin, ymax)
    ax_trace.set_aspect('equal', adjustable='box')
    ax_trace.set_title("Trace du mouvement", fontsize=14, color="#ddddff")

    sm = plt.cm.ScalarMappable(cmap='Oranges', norm=plt.Normalize(0, n_fr-1))
    cb = fig.colorbar(sm, ax=ax_trace, fraction=0.035, pad=0.02,
                      use_gridspec=False)
    cb.ax.tick_params(colors='#aaaaaa', labelsize=7)
    cb.set_label('Frame', color='#aaaaaa', fontsize=8)
    state["cb_trace"] = cb

# ============================================================
# GENERATE VIDEO
# ============================================================
def generate_video(X, exercise_name, class_id, cluster_id,
                   rep_number, rep_idx, output_dir):
    print(f"  🎥 classe {class_id} | cluster {cluster_id} | rep {rep_number} | seq #{rep_idx}")

    state["cluster"]       = cluster_id
    state["rep_idx"]       = rep_idx
    state["exercise_name"] = exercise_name
    state["class_id"]      = class_id
    state["skel"]          = get_skeleton(X, rep_idx)
    state["bounds"]        = compute_bounds(state["skel"])
    state["cb_trace"]      = None

    update_title()
    redraw_trace()

    original_frames = state["skel"].shape[0]
    n_frames = FPS * VIDEO_DURATION

    def update_video_frame(vfi):
        real_idx = min(int(vfi * original_frames / n_frames), original_frames - 1)
        return draw_frame(real_idx)

    ani = FuncAnimation(fig, update_video_frame,
                        frames=n_frames, interval=1000/FPS, repeat=False)
    output_path = output_dir / f"{exercise_name}_{class_id}_{cluster_id}_{rep_number}_{rep_idx}.mp4"
    writer = FFMpegWriter(fps=FPS, bitrate=3000)
    ani.save(output_path, writer=writer, dpi=180)
    print(f"  ✅ {output_path.name}")


exercise_dirs = sorted([d for d in CLASSIFICATION_DIR.iterdir()
                        if d.is_dir() and d.name.startswith("KERAAL")])

print(f"\n{len(exercise_dirs)} exercices KERAAL trouvés\n")

for exercise_dir in exercise_dirs:

    # Chercher x_train / x_test
    train_file = exercise_dir / "fold0" / "run0" / "x_train_fold0.npy"
    test_file  = exercise_dir / "fold0" / "run0" / "x_test_fold0.npy"
    if not train_file.exists():
        print(f"⚠️  Pas de données pour {exercise_dir.name}, ignoré")
        continue

    print(f"{'='*60}")
    print(f"EXERCICE : {exercise_dir.name}")
    print(f"{'='*60}")

    X = np.concatenate([np.load(train_file), np.load(test_file)], axis=0)
    print(f"  → {X.shape[0]} séquences | shape = {X.shape}")

    # Chercher class_C, class_E1, class_E2, class_E3
    class_dirs = sorted((exercise_dir / "fold0" / "run0").glob("class_*"))

    if not class_dirs:
        print(f"   Pas de dossiers class_* trouvés")
        continue

    print(f"  → {len(class_dirs)} classe(s) : {[d.name for d in class_dirs]}")

    output_dir = exercise_dir / "fold0" / "run0" / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_dir in class_dirs:
        class_id = class_dir.name.replace("class_", "")  # C, E1, E2, E3
        rep_csv  = class_dir / "representatives.csv"

        if not rep_csv.exists():
            print(f"   representatives.csv absent : {class_dir}")
            continue

        clusters_reps = {}
        with open(rep_csv) as f:
            for row in csv.DictReader(f):
                k   = int(row["cluster"])
                idx = int(row["sequence_index"])
                clusters_reps.setdefault(k, []).append(idx)

        n_total = sum(len(v) for v in clusters_reps.values())
        print(f"\n  Classe {class_id} : {len(clusters_reps)} clusters | {n_total} séquences")

        for cluster_id in sorted(clusters_reps):
            rep_list = clusters_reps[cluster_id]
            print(f"    Cluster {cluster_id} : {len(rep_list)} séquences")
            for rep_number, rep_idx in enumerate(rep_list):
                generate_video(X, exercise_dir.name, class_id,
                               cluster_id, rep_number, rep_idx, output_dir)

print("\n✅ Toutes les vidéos KERAAL ont été générées.")