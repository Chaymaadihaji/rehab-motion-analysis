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

CHAIN_COLORS = [
    "#ffffff",
    "#4ecdc4",
    "#f7b731",
    "#a29bfe",
    "#fd79a8",
]

# ============================================================
# NTU CHAINS
# ============================================================

CHAINS = [

    [0, 1, 20, 2, 3],

    [20, 4, 5, 6, 7],

    [20, 8, 9, 10, 11],

    [0, 12, 13, 14, 15],

    [0, 16, 17, 18, 19],
]

RIGHT_ARM = [20, 8, 9, 10, 11]

# ============================================================
# FIGURE
# ============================================================

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

# ============================================================
# STATE
# ============================================================

state = dict(

    cluster = 0,
    rep_idx = 0,

    skel = None,

    bounds = None,

    cb_trace = None,

    exercise_name = "",

    class_id = ""
)

# ============================================================
# TITLE
# ============================================================

title_obj = fig.suptitle(

    "",

    fontsize=22,

    fontweight='bold',

    color='white',

    y=0.97
)

def update_title():

    idx = state["rep_idx"]

    title_obj.set_text(

        f"Séquence #{idx}  |  {state['exercise_name']}"
    )

# ============================================================
# SKELETON
# ============================================================

def get_skeleton(X, seq_idx):

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

# ============================================================
# DRAW FRAME
# ============================================================

def draw_frame(fi):

    skel = state["skel"]

    x, y = pose_to_plot(skel[fi])

    ax_anim.clear()

    ax_anim.set_facecolor(PANEL_BG)

    ax_anim.set_xticks([])
    ax_anim.set_yticks([])

    for sp in ax_anim.spines.values():

        sp.set_visible(False)

    for ci, chain in enumerate(CHAINS):

        ax_anim.plot(

            x[chain],
            y[chain],

            color=CHAIN_COLORS[ci],

            lw=3.5,

            solid_capstyle='round'
        )

    used_joints = sorted(set(sum(CHAINS, [])))

    ax_anim.scatter(

        x[used_joints],
        y[used_joints],

        color=JOINT_COL,

        s=75
    )

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

        fontsize=14,

        color="#ddddff"
    )

    ax_anim.set_xlim(state["bounds"][0], state["bounds"][1])

    ax_anim.set_ylim(state["bounds"][2], state["bounds"][3])

    return []

# ============================================================
# TRACE
# ============================================================

def redraw_trace():

    if state["cb_trace"] is not None:

        try:
            state["cb_trace"].remove()
        except:
            pass

    ax_trace.clear()

    ax_trace.set_facecolor(PANEL_BG)

    skel = state["skel"]

    n_fr = skel.shape[0]

    cmap = matplotlib.colormaps["Oranges"]

    # --------------------------------------------------------
    # RIGHT ARM TRAJECTORY
    # --------------------------------------------------------

    for fi in range(n_fr):

        t = fi / max(n_fr - 1, 1)

        color = cmap(t)

        x, y = pose_to_plot(skel[fi])

        ax_trace.plot(

            x[RIGHT_ARM],
            y[RIGHT_ARM],

            color=color,

            lw=2.2,

            alpha=0.55 + 0.45*t
        )

        ax_trace.scatter(

            x[RIGHT_ARM],
            y[RIGHT_ARM],

            color=color,

            s=12,

            alpha=0.55 + 0.45*t
        )

    # --------------------------------------------------------
    # FINAL SKELETON
    # --------------------------------------------------------

    xf, yf = pose_to_plot(skel[-1])

    for ci, chain in enumerate(CHAINS):

        ax_trace.plot(

            xf[chain],
            yf[chain],

            color=CHAIN_COLORS[ci],

            lw=3
        )

    used_joints = sorted(set(sum(CHAINS, [])))

    ax_trace.scatter(

        xf[used_joints],
        yf[used_joints],

        color=JOINT_COL,

        s=55
    )

    ax_trace.set_aspect('equal')

    ax_trace.set_title(

        "Trace du mouvement",

        fontsize=14,

        color="#ddddff"
    )

# ============================================================
# GENERATE VIDEO
# ============================================================

def generate_video(

    X,
    exercise_name,
    class_id,
    cluster_id,
    rep_idx,
    output_dir
):

    print(
        f"🎥 {exercise_name} | classe {class_id} | cluster {cluster_id}"
    )

    state["cluster"] = cluster_id

    state["rep_idx"] = rep_idx

    state["exercise_name"] = exercise_name

    state["class_id"] = class_id

    state["skel"] = get_skeleton(X, rep_idx)

    state["bounds"] = compute_bounds(state["skel"])

    update_title()

    redraw_trace()

    original_frames = state["skel"].shape[0]

    n_frames = FPS * VIDEO_DURATION

    def update_video_frame(video_frame_idx):

        real_idx = int(
            video_frame_idx * original_frames / n_frames
        )

        real_idx = min(real_idx, original_frames - 1)

        return draw_frame(real_idx)

    ani = FuncAnimation(

        fig,

        update_video_frame,

        frames=n_frames,

        interval=1000/FPS,

        repeat=False
    )

    output_path = output_dir / (

        f"{exercise_name}_{class_id}_{cluster_id}.mp4"
    )

    writer = FFMpegWriter(

        fps=FPS,

        bitrate=3000
    )

    ani.save(

        output_path,

        writer=writer,

        dpi=180
    )

    print(f"✅ {output_path.name}")

# ============================================================
# MAIN LOOP
# ============================================================

exercise_dirs = [

    d for d in CLASSIFICATION_DIR.iterdir()

    if d.is_dir()
]

print("\nEXERCISES TROUVÉS :")

for d in exercise_dirs:
    print(d)

for exercise_dir in exercise_dirs:

    # ========================================================
    # FIND TRAIN/TEST FILES
    # ========================================================

    train_file = exercise_dir / "fold0/run0/x_train_fold0.npy"

    test_file  = exercise_dir / "fold0/run0/x_test_fold0.npy"

    if not train_file.exists():

        train_file = exercise_dir / "x_train_fold0.npy"

        test_file  = exercise_dir / "x_test_fold0.npy"

    if not train_file.exists():
        continue

    print("\n===================================================")
    print(f"EXERCICE : {exercise_dir.name}")
    print("===================================================")

    X = np.concatenate([

        np.load(train_file),

        np.load(test_file)

    ], axis=0)

    # ========================================================
    # FIND CLASS DIRECTORIES
    # ========================================================

    class_dirs = list(
        exercise_dir.glob("fold0/run0/class_*")
    )

    if len(class_dirs) == 0:

        class_dirs = list(
            exercise_dir.glob("class_*")
        )

    print("\nCLASSES TROUVÉES :")

    for c in class_dirs:
        print(c)

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = exercise_dir / "fold0/run0/videos"

    if not output_dir.parent.exists():

        output_dir = exercise_dir / "videos"

    output_dir.mkdir(exist_ok=True)

    # ========================================================
    # PROCESS EACH CLASS
    # ========================================================

    for class_dir in class_dirs:

        class_id = class_dir.name.split("_")[-1]

        rep_csv = class_dir / "representatives.csv"

        if not rep_csv.exists():

            print(f"❌ representatives.csv absent : {class_dir}")

            continue

        clusters_reps = {}

        with open(rep_csv) as f:

            for row in csv.DictReader(f):

                k = int(row["cluster"])

                idx = int(row["sequence_index"])

                clusters_reps.setdefault(k, []).append(idx)

        reps = [

            clusters_reps[k][0]

            for k in sorted(clusters_reps)
        ]

        # ====================================================
        # GENERATE VIDEOS
        # ====================================================

        for cluster_id in sorted(clusters_reps):

         rep_list = clusters_reps[cluster_id]

    print(f"Clusters trouvés : {cluster_id}")
    print(f"Nombre représentants : {len(rep_list)}")

    for rep_number, rep_idx in enumerate(rep_list):

        generate_video(

            X,

            exercise_dir.name,

            class_id,

            f"{cluster_id}_{rep_number}",

            rep_idx,

            output_dir
        )

print("\n✅ Toutes les vidéos ont été générées.")