from __future__ import print_function
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import binascii
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import json
import os
import matplotlib.markers as mmarkers
import matplotlib.colors as mcolors
import matplotlib.patches as mpatch
import sys
from skimage.color import rgb2lab, lab2rgb

# https://stackoverflow.com/questions/9018016/how-to-compare-two-colors-for-similarity-difference

# TODO: DMCS_RGB HAS DUPLICATE VALUES, HOW IS THAT POSSIBLE???????????
# TODO: make the code work with pngs
# TODO: Write somewhere which thread numbers are needed for the project
# TODO: Write how many stiches are necessary
# TODO: Calculate how much thread is needed of each type
# TODO: find the closest color using a different method (maybe hsv)
# TODO: make the color comparison plot and color-to-symbol plot scale to the number of colors chosen

# Globals
NUM_CLUSTERS = 30
DIR = "./"
FILEPATH = DIR + "im_igen.jpg"
JSONPATH = DIR + "DMC_colors.json"
pre, _ = os.path.splitext(FILEPATH)
SAVEIMGS = True
PLOTIMGS = True
IMGSAVEPATH = DIR + f"{pre}_{NUM_CLUSTERS}_colors.png"
GRIDSAVEPATH = DIR + f"{pre}_{NUM_CLUSTERS}_grid.png"
IMAGECOMPARISONPATH = DIR + f"{pre}_{NUM_CLUSTERS}_imagecompare.png"
COLORCOMPARISONPATH = DIR + f"{pre}_{NUM_CLUSTERS}_colorcompare.png"
IMG_SHAPE = (40, 50)
USE_ASPECT_RATIO = True
REDUCTION = 1


def hex2rgb(h):
    """Convert hexadecimal string to numpy array of r, g and b values.
    input:
        h: 7-character long string containing one hashtag and 6 values between 0 and F.
    output:
        rgb: np.array([r, g, b])."""
    h = h.lstrip("#")
    return np.array(list(int(h[i : i + 2], 16) for i in (0, 2, 4)))


def rgb2hex(r, g, b):
    """Convert  generator of r, g and b values to hexadecimal string.
    input:
        rgb: iterable of length 3.
    output:
        h: 7-character long string containing one hashtag and 6 values between 0 and F."""
    return "#{0:02x}{1:02x}{2:02x}".format(r, g, b).upper()


def closest_color(rgb, colors):
    """Given 3-length array and 3xn array, find the color that is closest to matching.
    input:
        rgb:  np.array([r, g, b]).
        colors: np.array([r0, g0, b0])
                        ([.,  .,  .])
                        ([rn, gn, bn])
    output:
        rgb_min: The 3-length rgb array with the smallest different to rgb."""
    r, g, b = rgb
    col_diffs = []
    for color in colors:
        cr, cg, cb = color
        color_diff = np.sqrt((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2)
        col_diffs.append(color_diff)

    min_col_idx = np.argmin(col_diffs)
    min_col = colors[min_col_idx]

    return min_col, min_col_idx


def mscatter(x, y, ax=None, m=None, **kw):
    if not ax:
        ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in tqdm(m):
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def color_comparison(codes_hex, new_cols_hex):
    fig = plt.figure(figsize=[3, 15])
    ax = fig.add_axes([0, 0, 1, 1])

    n_groups = 1
    n_rows = len(new_cols_hex) // n_groups

    print(f"{len(new_cols_hex) = }")
    sort_idxs = np.array(codes_hex).argsort()
    sorted_new_cols_hex = np.array(new_cols_hex)[sort_idxs[::-1]]
    sorted_codes_hex = np.array(codes_hex)[sort_idxs[::-1]]

    for j, (color_name1, color_name2) in enumerate(
        zip(sorted_new_cols_hex, sorted_codes_hex)
    ):
        # Pick text colour based on perceived luminance.
        rgba = mcolors.to_rgba_array([color_name2, color_name1])
        luma = 0.299 * rgba[:, 0] + 0.587 * rgba[:, 1] + 0.114 * rgba[:, 2]
        color_name2_text_color = "k" if luma[0] > 0.5 else "w"
        color_name1_text_color = "k" if luma[1] > 0.5 else "w"

        col_shift = (j // n_rows) * 2
        y_pos = j % n_rows
        text_args = dict(fontsize=10)
        ax.add_patch(mpatch.Rectangle((0 + col_shift, y_pos), 2, 6, color=color_name2))
        ax.add_patch(mpatch.Rectangle((1 + col_shift, y_pos), 2, 6, color=color_name1))
        ax.text(
            0.5 + col_shift,
            y_pos + 0.7,
            color_name2,
            color=color_name2_text_color,
            ha="center",
            **text_args,
        )
        ax.text(
            1.5 + col_shift,
            y_pos + 0.7,
            color_name1,
            color=color_name1_text_color,
            ha="center",
            **text_args,
        )

    for g in range(n_groups):
        ax.hlines(range(n_rows), 3 * g, 3 * g + 2.8, color="0.7", linewidth=1)
        ax.text(0.5 + 3 * g, -0.3, "True Color", ha="center")
        ax.text(1.5 + 3 * g, -0.3, "New Color", ha="center")

    ax.set_xlim(0, 2 * n_groups)
    ax.set_ylim(n_rows, -1)
    ax.axis("off")
    if SAVEIMGS:
        plt.savefig(COLORCOMPARISONPATH, dpi=200)
    plt.show()


def color_marker_relationship(color_marker_dict):
    fig = plt.figure(figsize=[3, 15])
    ax = fig.add_axes([0, 0, 1, 1])

    colors, attrs = list(color_marker_dict.keys()), list(color_marker_dict.values())
    markers = []
    markercolors = []
    names = []
    for attr in attrs:
        markers.append(attr[0])
        markercolors.append(attr[1])
        names.append(attr[2])

    n_groups = 1
    n_rows = len(colors) // n_groups

    sort_idxs = np.array(colors).argsort()
    sorted_markers = np.array(markers)[sort_idxs[::-1]]
    sorted_colors = np.array(colors)[sort_idxs[::-1]]
    sorted_markercolors = np.array(markercolors)[sort_idxs[::-1]]
    sorted_names = np.array(names)[sort_idxs[::-1]]
    print(f"{sorted_markers = }")
    print(f"{sorted_markercolors = }")

    for j, (color, marker, markercolor, name) in enumerate(
        zip(sorted_colors, sorted_markers, sorted_markercolors, sorted_names)
    ):
        # Pick text colour based on perceived luminance.
        rgba = mcolors.to_rgba_array([color])
        luma = 0.299 * rgba[0, 0] + 0.587 * rgba[0, 1] + 0.114 * rgba[0, 2]
        text_color = "k" if luma > 0.5 else "w"

        col_shift = (j // n_rows) * 2
        y_pos = j % n_rows
        text_args = dict(fontsize=10)
        ax.add_patch(mpatch.Rectangle((0 + col_shift, y_pos), 2, 6, color=color))
        ax.add_patch(mpatch.Rectangle((1 + col_shift, y_pos), 2, 6, color="#ffffff"))
        ax.text(
            0.5 + col_shift,
            y_pos + 0.7,
            name,
            color=text_color,
            ha="center",
            **text_args,
        )
        ax.scatter(1.5 + col_shift, y_pos + 0.5, marker=marker, c=markercolor, s=300)
        # ax.text(
        #     1.5 + col_shift,
        #     y_pos + 0.7,
        #     color,
        #     color=text_color,
        #     ha="center",
        #     **text_args,
        # )

    for g in range(n_groups):
        ax.hlines(range(n_rows), 3 * g, 3 * g + 2.8, color="0.7", linewidth=1)
        ax.text(0.5 + 3 * g, -0.3, "Color", ha="center")
        ax.text(1.5 + 3 * g, -0.3, "Symbol", ha="center")

    ax.set_xlim(0, 2 * n_groups)
    ax.set_ylim(n_rows, -1)
    ax.axis("off")
    # if SAVEIMGS:
    plt.savefig(COLORCOMPARISONPATH, dpi=200)
    plt.show()


# Load and size the image
print("Loading image...")
im = Image.open(FILEPATH)
if USE_ASPECT_RATIO:
    rows, cols, channels = np.shape(np.array(im))
    print(f"Image size: ({rows}, {cols})")

    rows = rows // REDUCTION
    cols = cols // REDUCTION
    max_size = (rows, cols)
    im.thumbnail(max_size)
else:
    im.thumbnail(IMG_SHAPE)

ar = np.asarray(im)
IMG_SHAPE = ar.shape
print(f"{IMG_SHAPE = }")

# Do clustering to find the NUM_CLUSTERS most important colors
ar = ar.reshape(scipy.product(IMG_SHAPE[:2]), IMG_SHAPE[2]).astype(float)
print("finding clusters...")
codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)

vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
counts, bins = scipy.histogram(vecs, len(codes))  # count occurrences

index_max = scipy.argmax(counts)  # find most frequent
peak = codes[index_max]
colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode("ascii")


# Insert the found colors in the image (not yet DMC)
c = ar.copy()
for i, code in enumerate(codes):
    c[scipy.r_[np.where(vecs == i)], :] = code
codes = codes.astype(int)
im_cluster = c.reshape(*IMG_SHAPE).astype(np.uint8)

# Load the file full of DMC colors
with open(JSONPATH, "r") as f:
    dmcs_hex = json.load(f)


dmcs_rgb = np.array(
    [hex2rgb(hex) for hex in dmcs_hex.keys()]
)  # Convert all of them to rgb values

# np.set_printoptions(threshold=sys.maxsize)
# dmcs_rgb.sort(axis=0)
# print(f"{dmcs_rgb = }")

dmcs_names = np.array([name for name in dmcs_hex.values()])
dmcs_hsv = dmcs_rgb
codes_hsv = codes
dmcs_hsv = rgb2lab(dmcs_rgb / 256)
codes_hsv = rgb2lab(codes / 256)

# Get list of DMC colors equivalent to the cluster colors

new_cols_rgb = np.zeros_like(codes)
new_cols_hex = []
new_cols_names = []
for i, code in enumerate(tqdm(codes_hsv)):
    best_col_lab, best_col_idx = closest_color(
        code, dmcs_hsv
    )  # Find the closest color in the DMC dataset
    best_col_rgb = 256 * lab2rgb(best_col_lab)
    best_col_rgb = best_col_rgb.astype(int)
    best_col_hex = rgb2hex(*best_col_rgb)
    # dmcs_rgb = np.delete(dmcs_rgb, (best_col_idx), axis=0)
    # dmcs_hsv = np.delete(dmcs_hsv, (best_col_idx), axis=0)
    # print(f"{best_col_hex = }")
    # dmcs_hex.pop(best_col_hex)

    col_name = dmcs_names[best_col_idx]  # Find the dmc name of the color for plotting
    new_cols_rgb[i] = best_col_rgb
    new_cols_hex.append(best_col_hex)
    new_cols_names.append(col_name)


codes_hex = [rgb2hex(*cod) for cod in codes]

if PLOTIMGS:
    color_comparison(
        codes_hex, new_cols_hex
    )  # Make a plot comparing the chosen colors to the real ones

# Create DMC image
markers = [
    ".",
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "s",
    "p",
    "*",
    "+",
    "x",
    "X",
    "D",
    "d",
    "_",
    ".",
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "s",
    "p",
    "*",
    "+",
    "x",
    "X",
    "D",
    "d",
    "_",
]
colors = [
    "k",
    "k",
    "k",
    "k",
    "k",
    "k",
    "k",
    "k",
    "k",
    "k",
    "k",
    "k",
    "k",
    "k",
    "k",
    "k",
    "k",
    "k",
    "k",
    "cornflowerblue",
    "cornflowerblue",
    "cornflowerblue",
    "cornflowerblue",
    "cornflowerblue",
    "cornflowerblue",
    "cornflowerblue",
    "cornflowerblue",
    "cornflowerblue",
    "cornflowerblue",
    "cornflowerblue",
    "cornflowerblue",
    "cornflowerblue",
    "cornflowerblue",
    "cornflowerblue",
    "cornflowerblue",
    "cornflowerblue",
    "cornflowerblue",
    "cornflowerblue",
]

assert len(new_cols_hex) < len(markers) and len(new_cols_hex) < len(
    colors
), f"You have chosen too many colors. {NUM_CLUSTERS} should be below {min(len(markers), len(colors))}."

hex2marker = {
    hex: (marker, color, name)
    for hex, (marker, color, name) in zip(
        new_cols_hex, zip(markers, colors, new_cols_names)
    )
}
hex2color = {hex: color for hex, color in zip(new_cols_hex, colors)}

c = ar.copy()
for i, code in enumerate(codes):
    c[scipy.r_[np.where(vecs == i)], :] = new_cols_rgb[i]

new_im = c.reshape(*IMG_SHAPE).astype(np.uint8)

# Show the image after DMC colors have been inserted
if PLOTIMGS:
    _, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(im)
    ax2.imshow(im_cluster)
    ax3.imshow(new_im)
    if SAVEIMGS:
        plt.savefig(IMAGECOMPARISONPATH)
    plt.show()

# Create plot containing symbols for each distinct type of pixel

if PLOTIMGS:
    fig_size_inch = max(IMG_SHAPE) // 10
    print(f"{fig_size_inch = }")

    fig_dpi = 500
    n_markers = max(IMG_SHAPE)
    plt.figure(figsize=(fig_size_inch, fig_size_inch))
    ax = plt.gca()
    x_major_ticks = np.arange(0, IMG_SHAPE[1], 10)
    x_minor_ticks = np.arange(0, IMG_SHAPE[1], 1)
    y_major_ticks = np.arange(0, IMG_SHAPE[0], 10)
    y_minor_ticks = np.arange(0, IMG_SHAPE[0], 1)
    ax.set_xticks(x_major_ticks)
    ax.set_xticks(x_minor_ticks, minor=True)
    ax.set_yticks(y_major_ticks)
    ax.set_yticks(y_minor_ticks, minor=True)
    ax.grid(which="major")
    ax.grid(which="minor", alpha=0.7)
    ax.set_aspect("equal")

    # Insert the correct marker for each color
    marker_size = 0.8 * (fig_size_inch * np.sqrt(fig_dpi) / n_markers) ** 2
    print(f"{marker_size = }")

    ies = []
    js = []
    marker_vals = []
    c_vals = []
    for i in range(IMG_SHAPE[0] - 1, -1, -1):
        for j in range(IMG_SHAPE[1] - 1, -1, -1):
            marker = hex2marker[rgb2hex(*new_im[i, j])][0]
            color = hex2color[rgb2hex(*new_im[i, j])]
            marker_vals.append(marker)
            c_vals.append(color)
            ies.append(i)
            js.append(j)

    ies = np.array(ies)
    js = np.array(js)
    marker_vals = np.array(marker_vals)
    c_vals = np.array(c_vals)
    scatter = mscatter(
        js + 0.5,
        IMG_SHAPE[0] - ies + 0.5,
        c=c_vals,
        s=marker_size,
        m=marker_vals,
        ax=ax,
    )

    # plt.scatter(js + 0.5, IMG_SHAPE[0] - ies + 0.5, c=c_vals, marker=marker_vals, s=10)

    ax.set_xlim(0, IMG_SHAPE[1])
    ax.set_ylim(1.0, IMG_SHAPE[0] + 1.0)
    if SAVEIMGS:
        plt.savefig(GRIDSAVEPATH, dpi=fig_dpi)
    plt.show()

color_marker_relationship(hex2marker)  # Show which markers corrospond to which colors
