"""
sheet_calibration.py  —  GP sheet generator + camera calibration pipeline
Hardware: Intel RealSense (any D-series)

Usage:
    python sheet_and_calibration.py --generate A1/A2/A3/A4  # produce sheet.pdf to print
    python sheet_and_calibration.py --calibrate             # robot at HOME_OBS, compute H
    python sheet_and_calibration.py --check                 # robot at HOME_OBS, live overlay
    python sheet_and_calibration.py --check --recalib       # force recompute H then show
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch

# ──────────────────────────────────────────────────────────────────────────────
# SHEET CONFIGURATION  (single source of truth)
# All XY coordinates are in mm, measured from the bottom-left corner of the
# printed sheet.  X = rightward,  Y = upward  (standard math convention).
# The robot base-frame XY must be aligned to these by jogging to each GP.
# ──────────────────────────────────────────────────────────────────────────────

SHEET_W_MM = 600   # 30 cm shared + 30 cm remote
SHEET_H_MM = 400   # approx from your sketch

# Shared area spans x = 0..300 mm, Remote area spans x = 300..600 mm
SHARED_X_MAX = 300

# ArUco reference markers — one per corner of the full sheet.
# IDs must be unique within DICT_4X4_50. Position = center of the printed square.
ARUCO_REFS = [
    {"id": 0, "xy_mm": ( 25,  25)},   # bottom-left
    {"id": 1, "xy_mm": (575,  25)},   # bottom-right
    {"id": 2, "xy_mm": ( 25, 375)},   # top-left
    {"id": 3, "xy_mm": (575, 375)},   # top-right
]
ARUCO_SIZE_MM = 35   # printed square side length

# Grasp points.
# type "standard"          → robot grasps at the circle center
# type "screws_container"  → robot grasps at the circle EDGE (radius offset); reserved for screws
# type "spring_container"  → robot grasps at the circle EDGE (radius offset); reserved for spring
#   edge_direction: angle in degrees (0 = right, 90 = up, 180 = left, 270 = down)
#   The actual grasp offset = GP_CONTAINER_RADIUS_MM in that direction.
#
# From your sketch (approximate mm positions):
GRASP_POINTS = [
    # ── SHARED AREA ──────────────────────────────────────────────────────────
    {
        "id": "GP_S1",
        "xy_mm": (90, 345),
        "zone": "shared",
        "type": "standard",
        "orientation_hint": 90,
        "label": "GP_S1",
    },
    {
        "id": "GP_SCREWDRIVER_S",
        "xy_mm": (20, 210),
        "zone": "shared",
        "type": "screwdriver",
        "orientation_hint": 90,
        "label": "GP_SCREWDRIVER_S",
        "content": "screwdriver",      # YOLO class name
    },
    {
        "id": "GP_S2",
        "xy_mm": (125, 195),
        "zone": "shared",
        "type": "standard",
        "orientation_hint": 180,
        "label": "GP_S2",
    },
    {
        "id": "GP_S3",
        "xy_mm": (230, 195),
        "zone": "shared",
        "type": "standard",
        "orientation_hint": 180,
        "label": "GP_S3",
    },
    # Screws container — robot grasps at edge, moves as a unit.
    {
        "id": "CTR_SCREW_S",
        "xy_mm": (85, 60),
        "zone": "shared",
        "type": "screws_container",
        "edge_direction": 90,
        "orientation_hint": 180,
        "label": "CTR_SCREW_S",
        "content": "screw",
    },
    # Spring container — same pattern as screws container.
    {
        "id": "CTR_GENERAL_S2",
        "xy_mm": (215, 60),
        "zone": "shared",
        "type": "general_container",
        "edge_direction": 90,
        "orientation_hint": 180,
        "label": "CTR_GENERAL_S2",
        "content": None,
    },

    # General container — for objects that can't be grasped individually.
    # Participates in normal GP routing but isn't reserved for a specific type.
    {
        "id": "CTR_GENERAL_S1",
        "xy_mm": (240, 340),
        "zone": "shared",
        "type": "general_container",
        "orientation_hint": 180,
        "label": "CTR_GENERAL_S1",
        "content": None,
    },

    # ── REMOTE AREA ───────────────────────────────────────────────────────────
    {
        "id": "GP_SCREWDRIVER_R",
        "xy_mm": (315, 240),
        "zone": "remote",
        "type": "screwdriver",
        "orientation_hint": 90,
        "label": "GP_SCREWDRIVER_R",
        "content": "screwdriver",      # YOLO class name
    },
    # General container — remote area, mirror of CTR_GENERAL_S.
    {
        "id": "CTR_GENERAL_R1",
        "xy_mm": (400, 345),
        "zone": "remote",
        "type": "general_container",
        "orientation_hint": 180,
        "label": "CTR_GENERAL_R1",
        "content": None,
    },
    {
        "id": "GP_R1",
        "xy_mm": (520, 345),
        "zone": "remote",
        "type": "standard",
        "orientation_hint": 90,
        "label": "GP_R1",
    },
    {
        "id": "GP_R2",
        "xy_mm": (385, 210),
        "zone": "remote",
        "type": "standard",
        "orientation_hint": 180,
        "label": "GP_R2",
    },
    {
        "id": "GP_R3",
        "xy_mm": (475, 210),
        "zone": "remote",
        "type": "standard",
        "orientation_hint": 180,
        "label": "GP_R3",
    },
    {
        "id": "GP_R4",
        "xy_mm": (560, 210),
        "zone": "remote",
        "type": "standard",
        "orientation_hint":180,
        "label": "GP_R4",
    },
    # Screws container — remote area (home when not in use).
    {
        "id": "CTR_SCREW_R",
        "xy_mm": (370, 60),
        "zone": "remote",
        "type": "screws_container",
        "edge_direction": 90,
        "orientation_hint": 180,
        "label": "CTR_SCREW_R",
        "content": "screw",
    },
    # Spring container — remote area (home when not in use).
    {
        "id": "CTR_GENERAL_R2",
        "xy_mm": (530, 60),
        "zone": "remote",
        "type": "general_container",
        "edge_direction": 90,
        "orientation_hint": 180,
        "label": "CTR_GENERAL_R2",
        "content": None,
    },
]

GP_STANDARD_RADIUS_MM  = 25   # reference radius (used for label offset / arrow length)
GP_CONTAINER_RADIUS_MM = 46.25   # container circles are slightly larger
GP_JAW_HALF_GAP_MM    = 3.5   # half the gripper jaw separation (total gap = 7 mm)
GP_JAW_HALF_LENGTH_MM = 18    # half-length of each jaw line (→ 44 mm total span)

CALIB_FILE   = Path("homography.json")
ARUCO_DICT   = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def gp_grasp_xy(gp: dict) -> tuple[float, float]:
    """
    Return the actual XY (mm) where the robot TCP should go.
    - "standard" and "screwdriver": center of circle.
    - "spring_container" and "screws_container": center offset by radius
      in edge_direction — robot grasps the rim, not the interior.
    """
    x, y = gp["xy_mm"]
    if gp["type"] in ("spring_container", "screws_container"):
        angle_rad = np.deg2rad(gp["edge_direction"])
        r = GP_CONTAINER_RADIUS_MM
        x += r * np.cos(angle_rad)
        y += r * np.sin(angle_rad)
    return x, y


# ──────────────────────────────────────────────────────────────────────────────
# 1. SHEET GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

def _draw_sheet_content(ax, W_TOTAL, H, LEGEND_W_MM, W_WORK):
    WX = LEGEND_W_MM   # shorthand: x-offset for everything in the work area
    ALPHA = 0.80        # global transparency — keeps colours faint to avoid YOLO false positives

    # ── Legend strip (left margin) ────────────────────────────────────────────
    # White background for the strip
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), LEGEND_W_MM, H,
        boxstyle="square,pad=0",
        facecolor="white", edgecolor="#CCCCCC", linewidth=0.5, zorder=1, alpha=ALPHA))

    # "OPERATOR SIDE" vertical label with orange background bar
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 10, H,
        boxstyle="square,pad=0",
        facecolor="#F39C12", edgecolor="none", zorder=2, alpha=ALPHA))
    ax.text(5, H / 2, "OPERATOR\nSIDE",
            ha="center", va="center", fontsize=5,
            color="white", fontweight="bold", fontfamily="monospace",
            rotation=-90, zorder=3, linespacing=1.4, alpha=ALPHA)

    # Legend title
    ax.text(LEGEND_W_MM / 2 + 5, H - 14, "LEGEND",
            ha="center", va="center", fontsize=6,
            color="#333333", fontweight="bold", fontfamily="monospace", zorder=3, alpha=ALPHA)

    # Legend entries — symbol above, name below (stacked).
    # "jaw" entries show two parallel lines; "circle" entries show a ring.
    legend_items = [
        ("#1E8449", "Standard GP",   "jaw"),
        ("#B7950B", "Screwdriver GP","jaw"),
        ("#E67E22", "Screws Cont.",  "circle"),
        ("#2980B9", "Spring Cont.",  "circle"),
        ("#7D3C98", "General Cont.", "circle"),
    ]
    legend_cx = LEGEND_W_MM / 2 + 5   # horizontal center of legend strip
    for i, (ec, label, shape) in enumerate(legend_items):
        sym_cy = H - 38 - i * 42
        if shape == "jaw":
            # Two short horizontal jaw lines + center dot
            for sign in (+1, -1):
                ax.plot([legend_cx - 10, legend_cx + 10],
                        [sym_cy + sign * 3.5, sym_cy + sign * 3.5],
                        color=ec, linewidth=1.5, zorder=3, alpha=ALPHA)
        else:
            # Circle (unfilled), consistent with container work-area drawing
            circle = plt.Circle((legend_cx, sym_cy), 10,
                                 facecolor="none", edgecolor=ec, linewidth=1.5, zorder=3, alpha=ALPHA)
            ax.add_patch(circle)
        # Center dot (all types)
        ax.plot(legend_cx, sym_cy, "o", markersize=2, color=ec, zorder=4, alpha=ALPHA)
        # Label to the right of the symbol, rotated -90° so the operator on the
        # left reads it left→right (text flows top→bottom on the page).
        ax.text(legend_cx + 14, sym_cy, label,
                ha="center", va="center", fontsize=9,
                rotation=-90,
                color=ec, fontweight="bold", fontfamily="monospace", zorder=4, alpha=ALPHA)

    # Dimensions inside legend strip (bottom)
    ax.text(LEGEND_W_MM / 2 + 5, 55, "Each zone:",
            ha="center", va="center", fontsize=4.5,
            color="#555555", fontfamily="monospace", zorder=3, alpha=ALPHA)
    ax.text(LEGEND_W_MM / 2 + 5, 42, "30 × 40 cm",
            ha="center", va="center", fontsize=5,
            color="#333333", fontweight="bold", fontfamily="monospace", zorder=3, alpha=ALPHA)
    ax.text(LEGEND_W_MM / 2 + 5, 28, "Total sheet:",
            ha="center", va="center", fontsize=4.5,
            color="#555555", fontfamily="monospace", zorder=3, alpha=ALPHA)
    ax.text(LEGEND_W_MM / 2 + 5, 15, "60 × 40 cm",
            ha="center", va="center", fontsize=5,
            color="#333333", fontweight="bold", fontfamily="monospace", zorder=3, alpha=ALPHA)

    # ── Work area backgrounds ─────────────────────────────────────────────────
    shared_x = WX
    remote_x = WX + SHARED_X_MAX

    ax.add_patch(mpatches.FancyBboxPatch(
        (shared_x, 0), SHARED_X_MAX, H,
        boxstyle="square,pad=0",
        facecolor="white", edgecolor="#AAAAAA", linewidth=1.5, zorder=1, alpha=ALPHA))
    ax.add_patch(mpatches.FancyBboxPatch(
        (remote_x, 0), W_WORK - SHARED_X_MAX, H,
        boxstyle="square,pad=0",
        facecolor="white", edgecolor="#AAAAAA", linewidth=1.5, zorder=1, alpha=ALPHA))

    # Zone labels
    ax.text(shared_x + SHARED_X_MAX / 2, H - 12, "SHARED AREA",
            ha="center", va="center", fontsize=8,
            color="#333333", fontweight="bold", fontfamily="monospace", alpha=ALPHA)
    ax.text(remote_x + (W_WORK - SHARED_X_MAX) / 2, H - 12, "REMOTE AREA",
            ha="center", va="center", fontsize=8,
            color="#333333", fontweight="bold", fontfamily="monospace", alpha=ALPHA)

    # Centre dividing line
    ax.plot([remote_x, remote_x], [0, H],
            color="#888888", lw=1.5, ls="--", alpha=ALPHA)

    # ── Grasp points ──────────────────────────────────────────────────────────
    for gp in GRASP_POINTS:
        cx = gp["xy_mm"][0] + WX
        cy = gp["xy_mm"][1]
        gp_type               = gp["type"]
        is_screws_container   = gp_type == "screws_container"
        is_spring_container   = gp_type == "spring_container"
        is_general_container  = gp_type == "general_container"
        is_screwdriver        = gp_type == "screwdriver"
        is_any_container      = is_screws_container or is_spring_container or is_general_container
        r = GP_CONTAINER_RADIUS_MM if is_any_container else GP_STANDARD_RADIUS_MM

        if is_screws_container:
            edge = "#E67E22"   # orange — screws container
        elif is_spring_container:
            edge = "#2980B9"   # blue — spring container
        elif is_general_container:
            edge = "#7D3C98"   # purple — general container
        elif is_screwdriver:
            edge = "#B7950B"   # yellow-brown — screwdriver
        else:
            edge = "#1E8449"   # green — standard

        if is_any_container:
            # Containers keep the circumference ring — YOLO-safe (no fill).
            circle = plt.Circle((cx, cy), r, facecolor="none",
                                 edgecolor=edge, linewidth=2, zorder=4, alpha=ALPHA)
            ax.add_patch(circle)
            ax.plot(cx, cy, "o", markersize=3, color=edge, zorder=6, alpha=ALPHA)
            if is_screws_container:
                inner_text = "SCREWS"
            elif is_spring_container:
                inner_text = "SPRING"
            else:
                inner_text = "GEN"
            # Inner type label — rotated -90° (reads top→bottom on page = left→right
            # for operator on the left), centred inside the circle.
            ax.text(cx, cy, inner_text,
                    ha="center", va="center", fontsize=9,
                    rotation=-90,
                    color=edge, fontweight="bold", fontfamily="monospace", zorder=8, alpha=ALPHA)
            # GP id label — to the LEFT of the circle (toward the operator).
            ax.text(cx - r - 8, cy, gp["label"],
                    ha="center", va="center", fontsize=11,
                    rotation=-90,
                    color=edge, fontweight="bold", fontfamily="monospace", alpha=ALPHA)

        else:
            # Standard / screwdriver GPs: two parallel jaw lines instead of a circle.
            angle_rad = np.deg2rad(gp["orientation_hint"])
            dx = np.cos(angle_rad)
            dy = np.sin(angle_rad)
            px = -np.sin(angle_rad)
            py =  np.cos(angle_rad)

            half_len = GP_JAW_HALF_LENGTH_MM
            half_gap = GP_JAW_HALF_GAP_MM

            # Two jaw lines, one on each side of center
            for sign in (+1, -1):
                jx = cx + sign * half_gap * px
                jy = cy + sign * half_gap * py
                ax.plot([jx - half_len * dx, jx + half_len * dx],
                        [jy - half_len * dy, jy + half_len * dy],
                        color=edge, linewidth=2, zorder=4, alpha=ALPHA)

            # Center dot
            ax.plot(cx, cy, "o", markersize=3, color=edge, zorder=6, alpha=ALPHA)

            # GP id label — screwdriver GPs go to the RIGHT (GP_SCREWDRIVER_R sits at
            # the far-right edge of remote; GP_SCREWDRIVER_S sits at the far-left edge
            # of shared, close to the legend strip); all others go to the LEFT.
            if is_screwdriver:
                label_x = cx + GP_STANDARD_RADIUS_MM - 8
            else:
                label_x = cx - GP_STANDARD_RADIUS_MM
            ax.text(label_x, cy, gp["label"],
                    ha="center", va="center", fontsize=11,
                    rotation=-90,
                    color=edge, fontweight="bold", fontfamily="monospace", alpha=ALPHA)

    # ── ArUco markers at corners of the WORK AREA ─────────────────────────────
    # Positions in ARUCO_REFS are in work-area mm — shift by WX for canvas
    sz = ARUCO_SIZE_MM
    quiet_zone_mm = 4  # 4mm white border around the marker
    for ref in ARUCO_REFS:
        # Generate at a higher resolution (400) for crisper PDF printing
        img    = aruco.generateImageMarker(ARUCO_DICT, ref["id"], 800)
        rx     = ref["xy_mm"][0] + WX
        ry     = ref["xy_mm"][1]
        
        # Draw a white background square for the ArUco quiet zone
        bg_rect = mpatches.Rectangle(
            (rx - sz/2 - quiet_zone_mm, ry - sz/2 - quiet_zone_mm),
            sz + quiet_zone_mm*2, sz + quiet_zone_mm*2,
            facecolor="white", edgecolor="none", zorder=7)
        ax.add_patch(bg_rect)

        ax.imshow(img,
                  extent=[rx - sz/2, rx + sz/2, ry - sz/2, ry + sz/2],
                  cmap="gray", vmin=0, vmax=255, zorder=8, origin="upper")
        
        ax.text(rx, ry - sz/2 - quiet_zone_mm - 2, f"REF-{ref['id']}",
                ha="center", va="top", fontsize=4,
                color="#555555", fontfamily="monospace", alpha=ALPHA)


def generate_sheet(output_path: str = "sheet.pdf", page_format: str = "FULL"):
    LEGEND_W_MM = 50          # width of the left legend strip in mm
    W_WORK      = SHEET_W_MM  # 600 mm — the actual work surface
    W_TOTAL     = LEGEND_W_MM + W_WORK
    H           = SHEET_H_MM  # 400 mm
    dpi         = 300

    # Standard landscape sizes in mm
    FORMATS = {
        "A4": (297, 210),
        "A3": (420, 297),
        "A2": (594, 420),
        "A1": (841, 594)
    }

    page_format = page_format.upper()
    if page_format in FORMATS:
        page_w, page_h = FORMATS[page_format]
        overlap_mm = 15  # Include 1.5 cm overlap to allow taping without losing borders
        margin_mm = 10   # Unprintable hardware margin for standard printers
    else:
        page_w, page_h = W_TOTAL, H
        overlap_mm = 0
        margin_mm = 0

    # The actual printable area on the page
    print_w = page_w - 2 * margin_mm
    print_h = page_h - 2 * margin_mm

    step_w = print_w - overlap_mm
    step_h = print_h - overlap_mm
    
    cols = int(np.ceil((W_TOTAL - overlap_mm) / step_w)) if page_format != "FULL" else 1
    rows = int(np.ceil((H - overlap_mm) / step_h)) if page_format != "FULL" else 1

    with PdfPages(output_path) as pdf:
        # Render from top to bottom (visually) so page 1 is top-left
        for r in range(rows - 1, -1, -1):
            for c in range(cols):
                fig, ax = plt.subplots(figsize=(page_w / 25.4, page_h / 25.4), dpi=dpi)
                
                _draw_sheet_content(ax, W_TOTAL, H, LEGEND_W_MM, W_WORK)
                
                x_start = c * step_w
                x_end   = x_start + print_w
                y_start = r * step_h
                y_end   = y_start + print_h
                
                # Setting limits specifically equal to the page size maintains the 1:1 scale
                if cols == 1 and rows == 1 and page_format != "FULL":
                    # Center content on the page (e.g. A1 single-page print)
                    x_pad = (page_w - W_TOTAL) / 2
                    y_pad = (page_h - H) / 2
                    ax.set_xlim(-x_pad, W_TOTAL + x_pad)
                    ax.set_ylim(-y_pad, H + y_pad)
                else:
                    # Expand limits by margin_mm to create the hardware margin
                    ax.set_xlim(x_start - margin_mm, x_end + margin_mm)
                    ax.set_ylim(y_start - margin_mm, y_end + margin_mm)
                
                ax.set_aspect("equal")
                ax.axis("off")
                ax.set_clip_on(False)
                fig.patch.set_facecolor("white")
                
                # Save — exact figure size, no extra whitespace
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                
                if page_format != "FULL" and not (cols == 1 and rows == 1):
                    # Draw a light dashed boundary inside the margin to show where to align/cut
                    rect = mpatches.Rectangle((x_start + overlap_mm/2, y_start + overlap_mm/2),
                                              print_w - overlap_mm, print_h - overlap_mm,
                                              fill=False, edgecolor="gray", linestyle=":", lw=0.5, zorder=10)
                    ax.add_patch(rect)

                    # Add a small page identifier at the bottom center of the tile
                    ax.text(x_start + print_w/2, y_start + overlap_mm/2 + 2, f"Page {c+1}x{rows-r}",
                            ha="center", va="bottom", fontsize=6, color="gray", zorder=10)

                pdf.savefig(fig, pad_inches=0)
                plt.close(fig)

    print(f"Sheet saved → {output_path}")
    if cols * rows > 1:
        print(f"Tiled onto {cols * rows} pages of size {page_format} ({page_w}x{page_h} mm).")
        print("15mm overlap included. Print at EXACTLY 100% scale (Actual Size) and tape them together along the dotted lines.")
    else:
        print(f"Physical size: {W_WORK/10:.0f} cm work area + {LEGEND_W_MM/10:.0f} cm legend strip")
        print(f"Total printed size: {W_TOTAL/10:.0f} cm × {H/10:.0f} cm")
        print("IMPORTANT: print at exactly 1:1 scale (do not scale to fit page).")


# ──────────────────────────────────────────────────────────────────────────────
# 2. REALSENSE CAMERA
# ──────────────────────────────────────────────────────────────────────────────

def get_realsense_frame(robot_serial: str = None) -> np.ndarray:
    """
    Grab one color frame from the first connected RealSense device.
    Returns BGR numpy array (OpenCV convention).
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        sys.exit("pyrealsense2 not installed.  Run: pip install pyrealsense2")

    pipeline = rs.pipeline()
    config   = rs.config()
    if robot_serial:
        config.enable_device(robot_serial)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    pipeline.start(config)
    try:
        # discard first few frames while auto-exposure settles
        for _ in range(10):
            pipeline.wait_for_frames()
        frames      = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            sys.exit("RealSense: could not get a color frame.")
        return np.asanyarray(color_frame.get_data())
    finally:
        pipeline.stop()


def get_realsense_stream(robot_serial: str = None):
    """
    Returns a generator that yields BGR frames continuously.
    Use in a loop; call .close() when done.
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        sys.exit("pyrealsense2 not installed.")

    pipeline = rs.pipeline()
    config   = rs.config()
    if robot_serial:
        config.enable_device(robot_serial)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        while True:
            frames      = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                yield np.asanyarray(color_frame.get_data())
    finally:
        pipeline.stop()


# ──────────────────────────────────────────────────────────────────────────────
# 3. HOMOGRAPHY: compute and load
# ──────────────────────────────────────────────────────────────────────────────

def _detect_aruco(frame: np.ndarray):
    detector = aruco.ArucoDetector(ARUCO_DICT, aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(frame)
    return corners, ids


def compute_and_save_homography(frame: np.ndarray) -> np.ndarray:
    """
    Detects the 4 ArUco corner markers in the HOME_OBS frame and computes H.
    world (mm) → image (px).
    """
    corners, ids = _detect_aruco(frame)

    if ids is None or len(ids) < 4:
        n = 0 if ids is None else len(ids)
        raise RuntimeError(
            f"Detected {n}/4 ArUco markers.\n"
            "Check: lighting, camera focus, sheet fully visible, not occluded."
        )

    ref_lookup  = {r["id"]: r["xy_mm"] for r in ARUCO_REFS}
    world_pts, image_pts = [], []

    for i, mid in enumerate(ids.flatten()):
        if mid in ref_lookup:
            cx = float(corners[i][0][:, 0].mean())
            cy = float(corners[i][0][:, 1].mean())
            image_pts.append([cx, cy])
            world_pts.append(list(ref_lookup[mid]))

    if len(world_pts) < 4:
        raise RuntimeError(
            f"Only {len(world_pts)} markers matched known IDs {list(ref_lookup)}. "
            "Make sure the correct ArUco dictionary (DICT_4X4_50) is printed."
        )

    H, mask = cv2.findHomography(
        np.array(world_pts, dtype=np.float32),
        np.array(image_pts, dtype=np.float32),
        cv2.RANSAC, 5.0,
    )
    print(f"Homography OK — {int(mask.sum())}/{len(world_pts)} inliers")

    CALIB_FILE.write_text(json.dumps({"H": H.tolist()}))
    print(f"Saved → {CALIB_FILE}")
    return H


def load_homography() -> np.ndarray:
    if not CALIB_FILE.exists():
        raise FileNotFoundError(
            f"{CALIB_FILE} not found — run --calibrate first."
        )
    return np.array(json.loads(CALIB_FILE.read_text())["H"], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# 4. PROJECT GPs INTO PIXEL SPACE
# ──────────────────────────────────────────────────────────────────────────────

def project_grasp_points(H: np.ndarray) -> dict:
    """Apply homography to all GRASP_POINTS and return a dict keyed by GP id with px_center, px_grasp, zone, type, content, label."""
    result = {}
    for gp in GRASP_POINTS:
        cx, cy = gp["xy_mm"]
        pt_c   = np.array([[[cx, cy]]], dtype=np.float32)
        px_c   = cv2.perspectiveTransform(pt_c, H)[0][0]

        gx, gy = gp_grasp_xy(gp)
        pt_g   = np.array([[[gx, gy]]], dtype=np.float32)
        px_g   = cv2.perspectiveTransform(pt_g, H)[0][0]

        entry = {
            "px_center": (int(px_c[0]), int(px_c[1])),
            "px_grasp":  (int(px_g[0]), int(px_g[1])),
            "zone":      gp["zone"],
            "type":      gp["type"],
            "content":   gp.get("content", None),
            "label":     gp["label"],
        }

        # For non-container GPs project the jaw line endpoints so the overlay
        # can draw the actual jaw geometry through the perspective transform.
        gp_type = gp["type"]
        if gp_type not in ("screws_container", "spring_container", "general_container"):
            angle_rad = np.deg2rad(gp["orientation_hint"])
            dx, dy = np.cos(angle_rad), np.sin(angle_rad)   # along jaw
            px_vec, py_vec = -np.sin(angle_rad), np.cos(angle_rad)  # across gap
            hl, hg = GP_JAW_HALF_LENGTH_MM, GP_JAW_HALF_GAP_MM
            # 4 jaw corner points in world mm
            jaw_world = np.array([[
                [cx + hg * px_vec - hl * dx,  cy + hg * py_vec - hl * dy],
                [cx + hg * px_vec + hl * dx,  cy + hg * py_vec + hl * dy],
                [cx - hg * px_vec - hl * dx,  cy - hg * py_vec - hl * dy],
                [cx - hg * px_vec + hl * dx,  cy - hg * py_vec + hl * dy],
            ]], dtype=np.float32)
            jaw_px = cv2.perspectiveTransform(jaw_world, H)[0]
            entry["px_jaw_lines"] = [
                ((int(jaw_px[0][0]), int(jaw_px[0][1])),
                 (int(jaw_px[1][0]), int(jaw_px[1][1]))),
                ((int(jaw_px[2][0]), int(jaw_px[2][1])),
                 (int(jaw_px[3][0]), int(jaw_px[3][1]))),
            ]

        result[gp["id"]] = entry
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 5. DRAW OVERLAY
# ──────────────────────────────────────────────────────────────────────────────

# BGR colors
C_GREEN     = ( 40, 180,  90)   # standard GPs
C_ORANGE    = ( 30, 130, 230)   # screws container (BGR: B=30, G=130, R=230)
C_BLUE      = (210, 130,  40)   # spring container (BGR: B=210, G=130, R=40)
C_YELLOW    = (  0, 200, 220)   # screwdriver GP
C_PURPLE    = (152,  60, 125)   # general container
C_TEXT_SH   = ( 40, 120,  40)
C_TEXT_RM   = (140,  60, 180)


def draw_overlay(frame: np.ndarray,
                 gp_data: dict,
                 show_aruco_markers: bool = True) -> np.ndarray:
    out = frame.copy()

    # Fixed pixel radii — independent of the homography scale so the overlay
    # stays readable across different zoom levels at HOME_OBS.
    SCREEN_R_STANDARD  = 22
    SCREEN_R_CONTAINER = 28

    for gp_id, info in gp_data.items():
        uc, vc = info["px_center"]
        gp_type               = info["type"]
        is_screws_container   = gp_type == "screws_container"
        is_spring_container   = gp_type == "spring_container"
        is_general_container  = gp_type == "general_container"
        is_screwdriver        = gp_type == "screwdriver"

        if is_screws_container:
            color = C_ORANGE
            r     = SCREEN_R_CONTAINER
        elif is_spring_container:
            color = C_BLUE
            r     = SCREEN_R_CONTAINER
        elif is_general_container:
            color = C_PURPLE
            r     = SCREEN_R_CONTAINER
        elif is_screwdriver:
            color = C_YELLOW
            r     = SCREEN_R_STANDARD
        else:
            color = C_GREEN
            r     = SCREEN_R_STANDARD

        is_any_container = is_screws_container or is_spring_container or is_general_container

        if is_any_container:
            # Containers: outer ring only (no fill)
            cv2.circle(out, (uc, vc), r, color, 2, cv2.LINE_AA)
            if is_screws_container:
                label_text = "SCREWS"
            elif is_spring_container:
                label_text = "SPRING"
            else:
                label_text = "GEN"
            cv2.putText(out, label_text, (uc - 20, vc + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)
        else:
            # Standard / screwdriver: two parallel jaw lines projected from world space
            jaw_lines = info.get("px_jaw_lines")
            if jaw_lines:
                for (p1, p2) in jaw_lines:
                    cv2.line(out, p1, p2, color, 2, cv2.LINE_AA)
            else:
                # Fallback if projection data is missing
                cv2.circle(out, (uc, vc), r, color, 2, cv2.LINE_AA)
            # Center dot
            cv2.circle(out, (uc, vc), 3, color, -1, cv2.LINE_AA)

        # ID label with background for readability
        label = gp_id
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        lx, ly = uc + r + 4, vc + 5
        cv2.rectangle(out, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2),
                      (20, 20, 20), -1)
        cv2.putText(out, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    # Zone separator line using the midpoint between left and right ArUco markers
    corners, ids = _detect_aruco(out)
    sep_u = None
    
    if ids is not None:
        left_x = [corners[i][0][:, 0].mean() for i, id_val in enumerate(ids.flatten()) if id_val in (0, 2)]
        right_x = [corners[i][0][:, 0].mean() for i, id_val in enumerate(ids.flatten()) if id_val in (1, 3)]
        if left_x and right_x:
            sep_u = int((sum(left_x) / len(left_x) + sum(right_x) / len(right_x)) / 2)

    # Fallback to GP data (with corrected X-coordinate index) if ArUco markers are occluded
    if sep_u is None:
        shared_us = [v["px_center"][0] for v in gp_data.values() if v["zone"] == "shared"]
        remote_us = [v["px_center"][0] for v in gp_data.values() if v["zone"] == "remote"]
        if shared_us and remote_us:
            sep_u = (max(shared_us) + min(remote_us)) // 2

    if sep_u is not None:
        h_img = out.shape[0]
        cv2.line(out, (sep_u, 0), (sep_u, h_img), (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(out, "SHARED", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEXT_SH, 1, cv2.LINE_AA)
        cv2.putText(out, "REMOTE", (sep_u + 10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEXT_RM, 1, cv2.LINE_AA)

    # Optionally draw detected ArUco outlines
    if show_aruco_markers:
        if ids is not None:
            aruco.drawDetectedMarkers(out, corners, ids)

    return out


def _draw_dashed_line(img, p1, p2, color, dash=6, gap=4, thickness=1):
    """Draw a dashed line between p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dist  = np.hypot(x2 - x1, y2 - y1)
    if dist < 1:
        return
    dx, dy   = (x2 - x1) / dist, (y2 - y1) / dist
    step     = dash + gap
    pos      = 0.0
    while pos < dist:
        end = min(pos + dash, dist)
        sx, sy = int(x1 + dx * pos), int(y1 + dy * pos)
        ex, ey = int(x1 + dx * end), int(y1 + dy * end)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
        pos += step


# ──────────────────────────────────────────────────────────────────────────────
# 6. CALIBRATION WORKFLOW
# ──────────────────────────────────────────────────────────────────────────────

def run_calibrate(robot_serial: str = None):
    """
    Robot must be at HOME_OBS.
    Grabs one frame, detects ArUco markers, computes and saves homography,
    then shows the GP overlay for visual confirmation.
    """
    print("Grabbing frame from RealSense...")
    frame = get_realsense_frame(robot_serial)

    print("Detecting ArUco markers and computing homography...")
    H = compute_and_save_homography(frame)

    gp_data  = project_grasp_points(H)
    display  = draw_overlay(frame, gp_data, show_aruco_markers=True)

    # Also draw ArUco detection result
    corners, ids = _detect_aruco(frame)
    if ids is not None:
        aruco.drawDetectedMarkers(display, corners, ids)

    cv2.imshow("Calibration result. Press Q to confirm, R to retry", display)
    print("Review the overlay. GP circles should align with printed sheet.")
    print("Press Q to confirm and save.  Press R to retry.")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            frame   = get_realsense_frame(robot_serial)
            H       = compute_and_save_homography(frame)
            gp_data = project_grasp_points(H)
            display = draw_overlay(frame, gp_data, show_aruco_markers=True)
            cv2.imshow("Calibration result. Press Q to confirm, R to retry", display)

    cv2.destroyAllWindows()
    print("Calibration complete.")
    return gp_data


def run_check(robot_serial: str = None, force_recalibrate: bool = False):
    """
    Robot must be at HOME_OBS.
    Shows a live feed with GP overlay from the saved homography.
    Returns the gp_data dict for use in the pick-place runtime.
    """
    if force_recalibrate or not CALIB_FILE.exists():
        return run_calibrate(robot_serial)

    H       = load_homography()
    gp_data = project_grasp_points(H)

    print("Live check — press Q to exit, R to recompute homography.")
    for frame in get_realsense_stream(robot_serial):
        display = draw_overlay(frame, gp_data, show_aruco_markers=False)
        cv2.imshow("GP overlay check, robot at HOME_OBS. Press Q to exit, R to recalibrate.", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            cv2.destroyAllWindows()
            return run_calibrate(robot_serial)

    cv2.destroyAllWindows()
    return gp_data


# ──────────────────────────────────────────────────────────────────────────────
# 7. RUNTIME HELPER (used by pick-place loop)
# ──────────────────────────────────────────────────────────────────────────────

def get_gp_data() -> dict:
    """
    Load saved homography and return projected GP data.
    Call this at the start of each session (robot at HOME_OBS).
    """
    H = load_homography()
    return project_grasp_points(H)


def snap_with_overlay(gp_data: dict, robot_serial: str = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Grab one frame from HOME_OBS and return (raw_frame, annotated_frame).
    Useful for occupancy detection in the pick-place loop.
    """
    frame   = get_realsense_frame(robot_serial)
    display = draw_overlay(frame, gp_data, show_aruco_markers=False)
    return frame, display


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GP sheet generator and RealSense calibration pipeline")
    parser.add_argument("--generate", nargs="?", const="FULL", choices=["FULL", "A4", "A3", "A2", "A1"],
                        help="Generate the printable sheet PDF. Optionally specify format (e.g., A4, A3) to tile.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Compute homography (robot at HOME_OBS)")
    parser.add_argument("--check",     action="store_true",
                        help="Show live GP overlay (robot at HOME_OBS)")
    parser.add_argument("--recalib",   action="store_true",
                        help="Force recompute homography before --check")
    args = parser.parse_args()

    if args.generate:
        generate_sheet("sheet.pdf", page_format=args.generate)

    elif args.calibrate:
        run_calibrate()

    elif args.check:
        run_check(force_recalibrate=args.recalib)

    else:
        parser.print_help()