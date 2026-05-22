import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter

import csv
import os

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILE = os.path.join(HERE, "x_train_fold0.npy")
TEST_FILE  = os.path.join(HERE, "x_test_fold0.npy")

REP_CSV = os.path.join(HERE, "..", "class_1", "representatives.csv")

EXERCISE = "IRDS_CLF_BN_EFL — Elbow Flexion Left"
CLASS_ID = 1

FPS = 22

OUTPUT_DIR = os.path.join(HERE, "generated_videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────────────────────

DARK_BG   = "#0d0d1a"
PANEL_BG  = "#12122a"

JOINT_COL = "#ff5555"

CHAIN_COLORS = [
    "#ffffff",
    "#4ecdc4",
    "#f7b731",
    "#a29bfe",
    "#fd79a8",
]

# ─────────────────────────────────────────────────────────────
# NTU CHAINS
# ─────────────────────────────────────────────────────────────

CHAINS = [

    [0, 1, 20, 2, 3],

    [20, 4, 5, 6, 7],

    [20, 8, 9, 10, 11],

    [0, 12, 13, 14, 15],

    [0, 16, 17, 18, 19],
]

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

print("Chargement des données...")

X = np.concatenate([
    np.load(TRAIN_FILE),
    np.load(TEST_FILE)
], axis=0)

print(f"→ {X.shape[0]} séquences")

# ─────────────────────────────────────────────────────────────
# LOAD REPRESENTATIVES
# ─────────────────────────────────────────────────────────────

clusters_reps = {}

with open(REP_CSV) as f:

    for row in csv.DictReader(f):

        k   = int(row["cluster"])
        idx = int(row["sequence_index"])

        clusters_reps.setdefault(k, []).append(idx)

N_CLUSTERS = len(clusters_reps)

reps = [clusters_reps[k][0] for k in sorted(clusters_reps)]

print(f"→ {N_CLUSTERS} clusters")

# ─────────────────────────────────────────────────────────────
# SKELETON FUNCTIONS
# ─────────────────────────────────────────────────────────────

def get_skeleton(seq_idx):

    raw = X[seq_idx]

    n_fr = raw.shape[1]

    data = raw.reshape(25, 3, n_fr)

    skel = data.transpose(2, 0, 1)

    return skel[:, :, :2]

def pose_to_plot(pose):

    return pose[:, 0], pose[:, 1]

def compute_bounds(skel):

    pad = 0.08

    return (

        skel[:,:,0].min()-pad,
        skel[:,:,0].max()+pad,

        skel[:,:,1].min()-pad,
        skel[:,:,1].max()+pad
    )

# ─────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────

plt.rcParams.update({

    'figure.facecolor': DARK_BG,
    'axes.facecolor': PANEL_BG,
    'text.color': 'white'
})

fig = plt.figure(figsize=(14, 8))

fig.patch.set_facecolor(DARK_BG)

gs = gridspec.GridSpec(

    1,
    2,

    figure=fig,

    width_ratios=[3.0, 1.1],

    wspace=0.15,

    left=0.03,
    right=0.98,
    bottom=0.05,
    top=0.92
)

ax_anim  = fig.add_subplot(gs[0])
ax_trace = fig.add_subplot(gs[1])

for ax in (ax_anim, ax_trace):

    ax.set_facecolor(PANEL_BG)

    for sp in ax.spines.values():

        sp.set_edgecolor("#2a2a4a")

# ─────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────

state = dict(

    cluster = 0,

    rep_idx = reps[0],

    skel = get_skeleton(reps[0]),

    bounds = None,

    cb_trace = None
)

state["bounds"] = compute_bounds(state["skel"])

# ─────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────

title_obj = fig.suptitle(

    "",

    fontsize=18,
    fontweight='bold',
    color='white',
    y=0.97
)

def update_title():

    k = state["cluster"]

    idx = state["rep_idx"]

    n = state["skel"].shape[0]

    title_obj.set_text(

    f"Séquence #{idx}  |  {EXERCISE}"
)

# ─────────────────────────────────────────────────────────────
# DRAW FRAME
# ─────────────────────────────────────────────────────────────

def draw_frame(fi):

    skel = state["skel"]

    x, y = pose_to_plot(skel[fi])

    ax_anim.clear()

    ax_anim.set_facecolor(PANEL_BG)

    ax_anim.set_xticks([])
    ax_anim.set_yticks([])

    for sp in ax_anim.spines.values():

        sp.set_visible(False)

    # skeleton lines
    for ci, chain in enumerate(CHAINS):

        ax_anim.plot(

            x[chain],
            y[chain],

            color=CHAIN_COLORS[ci],

            lw=3.5,

            solid_capstyle='round',

            zorder=3
        )

    # joints
    used_joints = sorted(set(sum(CHAINS, [])))

    ax_anim.scatter(

        x[used_joints],
        y[used_joints],

        color=JOINT_COL,

        s=75,

        zorder=5
    )

    # frame text
    ax_anim.text(

        0.02,
        0.97,

        f"Frame {fi+1} / {skel.shape[0]}",

        transform=ax_anim.transAxes,

        fontsize=12,

        color="#ccccff",

        va='top'
    )

    ax_anim.set_title(

        "Squelette animé",

        fontsize=13,

        color="#ddddff"
    )

    ax_anim.set_xlim(state["bounds"][0], state["bounds"][1])

    ax_anim.set_ylim(state["bounds"][2], state["bounds"][3])

    return []

# ─────────────────────────────────────────────────────────────
# TRACE
# ─────────────────────────────────────────────────────────────

def redraw_trace():

    if state["cb_trace"] is not None:

        try:
            state["cb_trace"].remove()
        except:
            pass

        state["cb_trace"] = None

    ax_trace.clear()

    ax_trace.set_facecolor(PANEL_BG)

    for sp in ax_trace.spines.values():

        sp.set_edgecolor("#2a2a4a")

    skel = state["skel"]

    n_fr = skel.shape[0]

    cmap = matplotlib.colormaps["Oranges"]

    RIGHT_ARM = [20, 8, 9, 10, 11]

    # movement trace
    for fi in range(n_fr):

        t = fi / max(n_fr - 1, 1)

        color = cmap(t)

        x, y = pose_to_plot(skel[fi])

        ax_trace.plot(

            x[RIGHT_ARM],
            y[RIGHT_ARM],

            color=color,

            lw=2.5,

            alpha=0.35 + 0.65*t,

            solid_capstyle='round',

            zorder=3
        )

        ax_trace.scatter(

            x[RIGHT_ARM],
            y[RIGHT_ARM],

            color=color,

            s=14,

            alpha=0.35 + 0.65*t,

            zorder=4
        )

    # final skeleton
    xf, yf = pose_to_plot(skel[-1])

    for ci, chain in enumerate(CHAINS):

        ax_trace.plot(

            xf[chain],
            yf[chain],

            color=CHAIN_COLORS[ci],

            lw=3,

            zorder=10
        )

    used_joints = sorted(set(sum(CHAINS, [])))

    ax_trace.scatter(

        xf[used_joints],
        yf[used_joints],

        color=JOINT_COL,

        s=50,

        zorder=11
    )

    ax_trace.set_aspect('equal')

    ax_trace.set_title(

        "Trace du mouvement",

        fontsize=13,

        color="#ddddff"
    )

    ax_trace.tick_params(

        colors="#777799",

        labelsize=8
    )

    # colorbar
    sm = plt.cm.ScalarMappable(

        cmap="Oranges",

        norm=plt.Normalize(0, n_fr-1)
    )

    cb = fig.colorbar(

        sm,

        ax=ax_trace,

        fraction=0.05,

        pad=0.03
    )

    cb.ax.tick_params(

        colors="#cccccc",

        labelsize=8
    )

    cb.set_label(

        "Frame",

        color="#cccccc",

        fontsize=10
    )

    state["cb_trace"] = cb

# ─────────────────────────────────────────────────────────────
# GENERATE VIDEO
# ─────────────────────────────────────────────────────────────

def generate_video(cluster_id):

    print(f"\n🎥 Génération cluster {cluster_id}")

    state["cluster"] = cluster_id

    state["rep_idx"] = reps[cluster_id]

    state["skel"] = get_skeleton(reps[cluster_id])

    state["bounds"] = compute_bounds(state["skel"])

    update_title()

    redraw_trace()

    # =====================================================
    # VIDEO SETTINGS
    # =====================================================

    video_duration = 10  # secondes

    # vraies frames de la séquence
    original_frames = state["skel"].shape[0]

    # frames finales de la vidéo
    n_frames = FPS * video_duration

    # =====================================================
    # REMAPPING TEMPOREL
    # =====================================================

    def update_video_frame(video_frame_idx):

        real_idx = int(
            video_frame_idx * original_frames / n_frames
        )

        real_idx = min(real_idx, original_frames - 1)

        return draw_frame(real_idx)

    # =====================================================
    # ANIMATION
    # =====================================================

    ani = FuncAnimation(

        fig,

        update_video_frame,

        frames=n_frames,

        interval=1000/FPS,

        blit=False,

        repeat=False
    )

    # =====================================================
    # OUTPUT PATH
    # =====================================================

    exercise_name = EXERCISE.split("—")[0].strip()

    class_name = str(CLASS_ID)

    output_path = os.path.join(

    OUTPUT_DIR,

    f"{exercise_name}_{class_name}_{cluster_id}.mp4"
)

    # =====================================================
    # VIDEO WRITER
    # =====================================================

    writer = FFMpegWriter(

        fps=FPS,

        bitrate=3000   
    )

    # =====================================================
    # SAVE VIDEO
    # =====================================================

    ani.save(

        output_path,

        writer=writer,

        dpi=180
    )

    print(f"✅ Sauvegardé : {output_path}")

# ─────────────────────────────────────────────────────────────
# GENERATE ALL VIDEOS
# ─────────────────────────────────────────────────────────────

for k in range(N_CLUSTERS):

    generate_video(k)

print("\n✅ Toutes les vidéos ont été générées.")