"""
Top-down parallel-jaw grasp estimator for tabletop objects — 6-DOF rewrite.

Physical model
──────────────
• Robot has 6 DOF: the wrist can rotate freely.
• The end-effector always arrives FROM ABOVE — the robot descends to the object.
  Convention: `approach` is the unit vector pointing FROM the robot TOWARD the
  object, i.e. it points downward.  Hard constraint: approach·Ẑ ≤ −min_descent
  (negative Z component; default −0.5, meaning at least 30° below horizontal).
  Purely horizontal or upward approaches are physically impossible because the
  robot cannot reach below the table surface.
• Jaws close along a single axis (close_dir) that is orthogonal to approach.
• Each jaw inner face is a flat rectangle:
      width  = jaw_half_width * 2   (along the lateral axis, Y in jaw frame)
      height = jaw_depth            (along the approach axis, Z in jaw frame)
• Contact model:
      Flat surface  → area contact  (spread in Y and Z) → both spans large
      Horizontal cylinder (axis ‖ Y) → line contact → Y-span large
      Vertical cylinder  (axis ‖ Z) → line contact → Z-span large
      Sphere / point → tiny blob → rejected by min_contact
  Acceptance: (Y-span ≥ min_line) OR (Z-span ≥ min_line) on BOTH jaw faces.

Grasp frame (local coordinates)
────────────────────────────────
  X = close_dir   jaw opening axis; jaw inner faces at X = ±hw
  Y = lateral     jaw width axis   (= approach × close_dir)
  Z = approach    robot descent direction (Z < 0 is inside/behind the jaw)
  Z = 0           jaw tip plane (where jaws first touch the object)

Score (lower = better)
──────────────────────
  descent  : reward steeper descent (approach·Ẑ more negative = more vertical)
  com      : small penalty for distance from object centre of mass
  contact  : reward more contact points
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GripperGeometry:
    """Physical dimensions of a parallel-jaw gripper (metres)."""
    max_aperture:   float = 0.040   # max jaw separation
    jaw_depth:      float = 0.030   # jaw height along approach axis
    jaw_thickness:  float = 0.005   # jaw body thickness (outside of contact face)
    palm_thickness: float = 0.010   # palm/body above jaw root
    jaw_half_width: float = 0.010   # half-width of jaw face along lateral axis

    @property
    def jaw_full_width(self) -> float:
        return 2.0 * self.jaw_half_width


@dataclass
class GraspCandidate:
    """A single evaluated grasp pose."""
    center:    np.ndarray           # grasp centre in world frame
    close_dir: np.ndarray           # unit vector: jaw closing axis
    approach:  np.ndarray           # unit vector: approach direction (into object)
    width:     float                # measured jaw separation (m)
    contact:   int                  # total contact points (both jaws)
    score:     float = field(default=0.0)

    @property
    def lateral(self) -> np.ndarray:
        """Third axis of grasp frame (orthogonal to close_dir and approach)."""
        return np.cross(self.approach, self.close_dir)

    @property
    def rotation(self) -> np.ndarray:
        """3×3 rotation matrix: columns are [close_dir, lateral, approach]."""
        return np.column_stack([self.close_dir, self.lateral, self.approach])


# ─────────────────────────────────────────────────────────────────────────────
# Core planner
# ─────────────────────────────────────────────────────────────────────────────

class GraspPlanner:
    """
    Finds the best parallel-jaw grasp on a tabletop point cloud.

    Parameters
    ──────────
    gripper         : GripperGeometry
    voxel_size      : downsampling voxel size (m)
    min_z           : table surface guard — ignore points below this height (m)
    num_seeds       : max seeds sampled per run
    n_approaches    : approach directions sampled per closing axis
    min_contact     : minimum total contact points (both jaws)
    min_line        : minimum span along Y *or* Z on each jaw face (m)
                      — satisfying either axis accepts the grasp (line contact)
    squeeze_tol     : half-thickness of the contact band at each jaw face (m)
    normal_dot_max  : antipodal threshold on dot(n1, n2); < 0 → facing away
    min_descent     : minimum magnitude of downward component; approach·Ẑ ≤ −min_descent
                      (0.5 = 30° below horizontal minimum; 1.0 = only vertical allowed)
    score_weights   : dict with keys 'descent', 'com', 'contact'
    """

    def __init__(
        self,
        gripper: GripperGeometry | None = None,
        *,
        voxel_size:      float = 0.003,
        min_z:           float = 0.005,
        num_seeds:       int   = 500,
        n_approaches:    int   = 4,
        min_contact:     int   = 4,
        min_line:        float = 0.008,
        squeeze_tol:     float = 0.0015,
        normal_dot_max:  float = -0.35,
        min_descent:     float = 0.50,      # approach·Ẑ ≤ −min_descent (robot descends from above)
        score_weights:   dict | None = None,
    ) -> None:
        self.gripper         = gripper or GripperGeometry()
        self.voxel_size      = voxel_size
        self.min_z           = min_z
        self.num_seeds       = num_seeds
        self.n_approaches    = n_approaches
        self.min_contact     = min_contact
        self.min_line        = min_line
        self.squeeze_tol     = squeeze_tol
        self.normal_dot_max  = normal_dot_max
        self.min_descent     = min_descent
        # Score: lower = better.
        # descent: penalise shallow approaches (approach·Ẑ close to 0 = horizontal)
        # The more negative approach·Ẑ is, the steeper the descent → reward it.
        self.weights = {"descent": 2.0, "com": 0.1, "contact": -0.01}
        if score_weights:
            self.weights.update(score_weights)

    # ── public API ──────────────────────────────────────────────────────────

    def plan(self, pcd: o3d.geometry.PointCloud) -> GraspCandidate | None:
        """Return the best grasp, or None if no valid grasp is found."""
        bar = "─" * 62
        print(f"\n{bar}")
        t_wall = time.perf_counter()

        # ── preprocessing ────────────────────────────────────────────────────
        t0  = time.perf_counter()
        pcd = self._ensure_normals(pcd)
        seed_cloud = pcd.voxel_down_sample(self.voxel_size)

        seed_pts   = np.asarray(seed_cloud.points,  dtype=np.float32)
        seed_norms = np.asarray(seed_cloud.normals, dtype=np.float32)
        obj_pts    = np.asarray(pcd.points,         dtype=np.float32)
        obj_pts    = obj_pts[obj_pts[:, 2] > self.min_z]

        if len(obj_pts) == 0:
            print("  ✗ No object points above min_z.")
            print(bar); return None

        com  = obj_pts.mean(axis=0)
        tree = o3d.geometry.KDTreeFlann(seed_cloud)
        t_pre = time.perf_counter() - t0

        # ── Level 0: seed pre-filter ─────────────────────────────────────────
        # Keep seeds above table with a meaningfully horizontal normal
        # (|nz| < 0.95).  Fully vertical normals (top/bottom faces) cannot
        # be part of an antipodal lateral-closure pair.
        above   = seed_pts[:, 2] > self.min_z
        horiz   = np.sqrt(np.maximum(0.0, 1.0 - seed_norms[:, 2] ** 2)) > 0.30
        valid   = np.where(above & horiz)[0]
        rng     = np.random.default_rng(0)
        seeds   = rng.choice(valid, min(self.num_seeds, len(valid)), replace=False)

        print(f"  Preprocessing  {(time.perf_counter()-t0)*1000:6.1f} ms  │  "
              f"{len(seed_pts):,} seed pts · {len(obj_pts):,} obj pts · "
              f"{len(seeds):,}/{len(valid):,} seeds")

        # ── candidate search ─────────────────────────────────────────────────
        t0         = time.perf_counter()
        candidates: list[GraspCandidate] = []
        n_batch = n_eval = 0
        g = self.gripper

        for i in seeds:
            p1 = seed_pts[i]
            n1 = seed_norms[i]

            _, nbr_raw, _ = tree.search_radius_vector_3d(p1, g.max_aperture)
            nbr_idx = np.asarray(nbr_raw[1:], dtype=np.int32)
            if len(nbr_idx) == 0:
                continue

            P2 = seed_pts[nbr_idx]
            N2 = seed_norms[nbr_idx]

            alive = P2[:, 2] > self.min_z
            if not np.any(alive):
                continue
            P2, N2 = P2[alive], N2[alive]

            # ── Level 1: vectorised batch filter ─────────────────────────────
            vecs  = P2 - p1
            dists = np.linalg.norm(vecs, axis=1)
            ok    = dists > 1e-3
            vecs, dists, P2, N2 = vecs[ok], dists[ok], P2[ok], N2[ok]
            if len(vecs) == 0:
                continue

            close_dirs = vecs / dists[:, None]      # (M, 3) unit closing dirs

            # Antipodal: normals must face generally opposite each other
            ok  = (N2 @ n1) < self.normal_dot_max
            # Closing axis must be roughly aligned with seed normal
            ok &= np.abs(close_dirs @ n1) >= 0.40

            n_batch += int(np.sum(ok))
            if not np.any(ok):
                continue

            close_dirs_f = close_dirs[ok]
            P2_f         = P2[ok]

            # ── Level 2: dedup closing directions ────────────────────────────
            close_dirs_f, P2_f = _dedup_dirs(close_dirs_f, P2_f)

            # ── Levels 3+4: approach sampling + full evaluation ───────────────
            for cd, p2 in zip(close_dirs_f, P2_f):
                approaches = self._sample_approaches(cd)
                if len(approaches) == 0:
                    continue
                for app in approaches:
                    n_eval += 1
                    g_result = self._evaluate_grasp(p1, p2, cd, app, obj_pts, com)
                    if g_result is not None:
                        candidates.append(g_result)

        t_search = time.perf_counter() - t0
        print(f"  Search         {t_search*1000:6.1f} ms  │  "
              f"{n_batch:,} batch-filtered · {n_eval:,} evaluated · "
              f"{len(candidates):,} valid")

        if not candidates:
            print(f"  ✗ No valid grasp found  (total {(time.perf_counter()-t_wall)*1000:.1f} ms)")
            print(bar); return None

        candidates.sort(key=lambda c: c.score)
        best = candidates[0]
        descent_deg = float(np.degrees(np.arcsin(-float(best.approach[2]))))
        print(f"  Best  │  width {best.width*100:.2f} cm  │  "
              f"contact {best.contact} pts  │  "
              f"descent {descent_deg:.1f}°  │  "
              f"score {best.score:.4f}")
        print(f"  Total          {(time.perf_counter()-t_wall)*1000:6.1f} ms")
        print(bar)
        return best

    # ── approach sampling ────────────────────────────────────────────────────

    def _sample_approaches(self, close_dir: np.ndarray) -> list[np.ndarray]:
        """
        Sample candidate approach vectors for this closing axis.

        Physical constraint: the robot descends FROM ABOVE, so `approach`
        points FROM the robot TOWARD the object — downward in world space.
        Hard filter: approach·Ẑ ≤ −min_descent
          (e.g. min_descent=0.5 → at least 30° below horizontal).

        Always includes the steepest-descent candidate (−Ẑ projected onto the
        plane perpendicular to close_dir) so purely vertical grasps are never
        missed even when n_approaches is small.
        """
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(close_dir, ref))) > 0.95:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        u = np.cross(close_dir, ref).astype(np.float32)
        u /= np.linalg.norm(u)
        v = np.cross(u, close_dir).astype(np.float32)

        results = []
        angles = np.linspace(0.0, 2 * np.pi, self.n_approaches, endpoint=False)
        for a in angles:
            app = (float(np.cos(a)) * u + float(np.sin(a)) * v).astype(np.float32)
            app /= np.linalg.norm(app)
            if float(app[2]) > 0:   # must point downward
                app = -app
            if float(app[2]) <= -self.min_descent:
                results.append(app)

        # Guarantee the best vertical candidate: project −Ẑ onto the plane
        # perp to close_dir.  For a horizontal close_dir this equals −Ẑ exactly.
        down = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        proj = down - float(np.dot(down, close_dir)) * close_dir
        mag  = float(np.linalg.norm(proj))
        if mag > 1e-4:
            best_down = (proj / mag).astype(np.float32)
            if float(best_down[2]) <= -self.min_descent:
                results.append(best_down)

        return results

    # ── full grasp evaluation ────────────────────────────────────────────────

    def _evaluate_grasp(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        close_dir: np.ndarray,
        approach:  np.ndarray,
        obj_pts:   np.ndarray,
        com:       np.ndarray,
    ) -> GraspCandidate | None:
        """
        Project object points into the grasp frame and check:
          1. Aperture fits within max_aperture.
          2. No collision with jaw bodies or palm.
          3. Sufficient contact on both jaw inner faces.
          4. Score the grasp.

        Grasp frame (local coordinates):
          X = close_dir  (jaw opening axis; jaw faces at ±hw)
          Y = lateral    (jaw width axis)
          Z = approach   (Z=0 at jaw tip plane; Z < 0 is inside the jaw)
        """
        g       = self.gripper
        center  = (p1 + p2) * 0.5
        lateral = np.cross(approach, close_dir).astype(np.float32)
        lateral /= np.linalg.norm(lateral)

        # Build rotation matrix R such that local = (world - center) @ R.T
        R     = np.vstack([close_dir, lateral, approach])   # (3, 3)
        local = ((obj_pts - center) @ R.T).astype(np.float32)  # (N, 3)

        lx, ly, lz = local[:, 0], local[:, 1], local[:, 2]

        # ── 1. Aperture ───────────────────────────────────────────────────────
        # Points inside the jaw opening volume
        in_open = (
            (lz > -g.jaw_depth) & (lz < 0.002)
            & (np.abs(ly) < g.jaw_half_width)
        )
        if not np.any(in_open):
            return None

        x_open       = lx[in_open]
        x_min, x_max = float(x_open.min()), float(x_open.max())
        true_width   = x_max - x_min
        if true_width > g.max_aperture:
            return None

        # Re-centre so jaw faces are symmetric around X=0
        shift  = (x_max + x_min) * 0.5
        center = center + close_dir * shift
        lx     = lx - shift          # cheap: just shift the X column
        hw     = true_width * 0.5

        # ── 2. Collision check ────────────────────────────────────────────────
        if self._collides(lx, ly, lz, hw, g):
            return None

        # ── 3. Contact quality ────────────────────────────────────────────────
        contact, ok = self._check_contact(lx, ly, lz, hw, g)
        if contact < self.min_contact or not ok:
            return None

        # ── 4. Score (lower = better) ─────────────────────────────────────────
        # approach[2] is negative (robot descends); more negative = more vertical.
        # Penalise shallow approaches by adding (descent_weight * (1 + approach[2])).
        # When approach[2] = -1 (perfectly vertical) this term = 0 (no penalty).
        # When approach[2] = -0.5 (45° tilt) this term = 0.5 * weight.
        w     = self.weights
        score = (
            w["descent"] * (1.0 + float(approach[2]))   # 0 = vertical, >0 = tilted
            + w["com"]   * float(np.linalg.norm(center - com))
            + w["contact"] * contact
        )
        return GraspCandidate(
            center=center, close_dir=close_dir, approach=approach,
            width=true_width, contact=contact, score=score,
        )

    # ── contact check ────────────────────────────────────────────────────────

    def _check_contact(
        self,
        lx: np.ndarray, ly: np.ndarray, lz: np.ndarray,
        hw: float, g: GripperGeometry,
    ) -> tuple[int, bool]:
        """
        Check contact on both jaw inner faces.

        Each jaw inner face is the rectangle:
            |Y| < jaw_half_width  (jaw width)
            Z ∈ [-jaw_depth, 0]   (jaw height along approach)

        A point "touches" a jaw face if its X coordinate is within squeeze_tol
        of the face position (±hw).

        Contact geometry:
            Flat surface → points spread in Y and Z → Y-span and Z-span both large
            Horizontal cylinder (axis ‖ Y) → points spread along Y, narrow in Z
            Vertical cylinder (axis ‖ Z) → points spread along Z, narrow in Y
            Sphere / point contact → small blob → rejected by min_contact

        Acceptance criterion per jaw:
            (Y-span ≥ min_line) OR (Z-span ≥ min_line)
        This captures both area and line-contact cases with two cheap max-min ops.
        """
        stol = self.squeeze_tol
        ml   = self.min_line

        in_y = np.abs(ly) < g.jaw_half_width
        in_z = (lz > -g.jaw_depth) & (lz < 0.002)

        def _jaw_stats(face_x: float) -> tuple[int, bool]:
            mask = in_y & in_z & (np.abs(lx - face_x) < stol)
            n = int(np.sum(mask))
            if n < 2:
                return n, False
            y_span = float(ly[mask].max() - ly[mask].min())
            z_span = float(lz[mask].max() - lz[mask].min())
            ok = (y_span >= ml) or (z_span >= ml)
            return n, ok

        n_right, ok_right = _jaw_stats(+hw)
        n_left,  ok_left  = _jaw_stats(-hw)

        return n_right + n_left, ok_right and ok_left

    # ── collision check ──────────────────────────────────────────────────────

    @staticmethod
    def _collides(
        lx: np.ndarray, ly: np.ndarray, lz: np.ndarray,
        hw: float, g: GripperGeometry,
    ) -> bool:
        """
        True if any point is inside a solid gripper part.

        Jaw bodies (left and right) — outside the opening:
            |X| ∈ [hw, hw + jaw_thickness]
            |Y| < jaw_half_width
            Z  ∈ [-jaw_depth, 0]

        Palm — behind the jaw root:
            |X| < hw + jaw_thickness
            |Y| < jaw_half_width
            Z  ∈ [-jaw_depth - palm_thickness, -jaw_depth]
        """
        in_y = np.abs(ly) < g.jaw_half_width
        in_jaw_z = (lz > -g.jaw_depth) & (lz < 0.0)
        abs_x = np.abs(lx)

        # Jaw body collision
        if np.any(in_y & in_jaw_z & (abs_x > hw) & (abs_x < hw + g.jaw_thickness)):
            return True

        # Palm collision
        in_palm_z = (lz > -g.jaw_depth - g.palm_thickness) & (lz < -g.jaw_depth)
        if np.any(in_y & in_palm_z & (abs_x < hw + g.jaw_thickness)):
            return True

        return False

    # ── utilities ────────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_normals(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        if not pcd.has_normals():
            print("  (estimating normals…)")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
            )
            pcd.orient_normals_towards_camera_location(
                camera_location=np.array([0.0, 0.0, 1.0])
            )
        return pcd

    # ── visualisation ────────────────────────────────────────────────────────

    def visualise(self, grasp: GraspCandidate) -> list[o3d.geometry.TriangleMesh]:
        """Return solid TriangleMesh parts for the gripper at the given pose."""
        g   = self.gripper
        R   = np.column_stack([grasp.close_dir, grasp.lateral, grasp.approach])
        c   = grasp.center
        hw  = grasp.width * 0.5
        t   = g.jaw_thickness

        def _place(mesh, local_offset):
            mesh.rotate(R, center=(0, 0, 0))
            mesh.translate(c + R @ local_offset)
            return mesh

        parts = []
        for sign in (+1.0, -1.0):
            jaw = o3d.geometry.TriangleMesh.create_box(t, g.jaw_full_width, g.jaw_depth)
            jaw.translate((-t * 0.5, -g.jaw_half_width, -g.jaw_depth))
            jaw = _place(jaw, np.array([sign * (hw + t * 0.5), 0.0, 0.0]))
            jaw.paint_uniform_color([0.88, 0.18, 0.12])
            jaw.compute_vertex_normals()
            parts.append(jaw)

        palm_hw = hw + t
        palm = o3d.geometry.TriangleMesh.create_box(
            palm_hw * 2, g.jaw_full_width, g.palm_thickness
        )
        palm.translate((-palm_hw, -g.jaw_half_width, -g.jaw_depth - g.palm_thickness))
        palm = _place(palm, np.zeros(3))
        palm.paint_uniform_color([0.48, 0.07, 0.05])
        palm.compute_vertex_normals()
        parts.append(palm)

        # Approach arrow
        shaft_h = 0.055
        head_h  = 0.012
        palm_root = c - grasp.approach * g.jaw_depth
        shaft = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0015, height=shaft_h, resolution=16)
        shaft.translate([0, 0, shaft_h * 0.5])
        shaft.rotate(R, center=(0, 0, 0))
        shaft.translate(palm_root - grasp.approach * (shaft_h + head_h))
        shaft.paint_uniform_color([0.08, 0.78, 0.22])
        shaft.compute_vertex_normals()
        parts.append(shaft)

        head = o3d.geometry.TriangleMesh.create_cone(radius=0.004, height=head_h, resolution=16)
        head.translate([0, 0, -head_h * 0.5])
        head.rotate(R, center=(0, 0, 0))
        head.translate(palm_root - grasp.approach * head_h)
        head.paint_uniform_color([0.08, 0.78, 0.22])
        head.compute_vertex_normals()
        parts.append(head)

        return parts


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dedup_dirs(
    close_dirs: np.ndarray,
    P2: np.ndarray,
    bin_size: float = 0.10,          # ~5.7° angular bins
) -> tuple[np.ndarray, np.ndarray]:
    """Keep one representative per angular bin in X-Y to reduce redundant evals."""
    seen: set[tuple[int, int]] = set()
    keep_cd, keep_p2 = [], []
    inv = 1.0 / bin_size
    for cd, p2 in zip(close_dirs, P2):
        key = (int(round(cd[0] * inv)), int(round(cd[1] * inv)))
        if key not in seen:
            seen.add(key)
            keep_cd.append(cd)
            keep_p2.append(p2)
    if not keep_cd:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    return np.array(keep_cd, dtype=np.float32), np.array(keep_p2, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Scene helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_table() -> o3d.geometry.TriangleMesh:
    t = o3d.geometry.TriangleMesh.create_box(0.40, 0.40, 0.01)
    t.translate((-0.20, -0.20, -0.01))
    t.paint_uniform_color([0.18, 0.18, 0.18])
    return t


def _sample(mesh: o3d.geometry.TriangleMesh, n: int) -> o3d.geometry.PointCloud:
    mesh.compute_vertex_normals()
    return mesh.sample_points_uniformly(n, use_triangle_normal=True)


def _with_table(obj_pcd: o3d.geometry.PointCloud, table_n: int = 8_000) -> o3d.geometry.PointCloud:
    return obj_pcd + _sample(_make_table(), table_n)


# ─────────────────────────────────────────────────────────────────────────────
# Scenes  (unchanged geometry)
# ─────────────────────────────────────────────────────────────────────────────

def scene_screwdriver() -> o3d.geometry.PointCloud:
    R90y = o3d.geometry.get_rotation_matrix_from_xyz([0, np.pi / 2, 0])
    handle = o3d.geometry.TriangleMesh.create_cylinder(radius=0.015, height=0.080)
    handle.rotate(R90y, center=(0, 0, 0)); handle.translate((0.0, 0.0, 0.015))
    handle.paint_uniform_color([0.85, 0.30, 0.05])
    shaft = o3d.geometry.TriangleMesh.create_cylinder(radius=0.003, height=0.100)
    shaft.rotate(R90y, center=(0, 0, 0)); shaft.translate((0.09, 0.0, 0.015))
    shaft.paint_uniform_color([0.80, 0.80, 0.80])
    return _with_table(_sample(handle + shaft, 22_000))


def scene_throttle_stop() -> o3d.geometry.PointCloud:
    base = o3d.geometry.TriangleMesh.create_box(0.040, 0.030, 0.003)
    base.translate((-0.020, -0.015, 0.0))
    arm = o3d.geometry.TriangleMesh.create_box(0.010, 0.040, 0.002)
    arm.rotate(o3d.geometry.get_rotation_matrix_from_xyz([np.pi / 4, 0, 0]), center=(0, 0, 0))
    arm.translate((0.010, 0.0, 0.003))
    mesh = base + arm; mesh.paint_uniform_color([0.70, 0.70, 0.50])
    return _with_table(_sample(mesh, 22_000))


def scene_hex_bolt() -> o3d.geometry.PointCloud:
    shaft = o3d.geometry.TriangleMesh.create_cylinder(radius=0.008, height=0.040, resolution=32)
    shaft.translate((0, 0, 0.020)); shaft.paint_uniform_color([0.72, 0.72, 0.72])
    head = o3d.geometry.TriangleMesh.create_cylinder(radius=0.014, height=0.013, resolution=6)
    head.translate((0, 0, 0.0465)); head.paint_uniform_color([0.55, 0.55, 0.55])
    return _with_table(_sample(shaft + head, 20_000))


def scene_l_bracket() -> o3d.geometry.PointCloud:
    fa = o3d.geometry.TriangleMesh.create_box(0.060, 0.010, 0.025)
    fa.translate((-0.030, -0.005, 0.0)); fa.paint_uniform_color([0.60, 0.62, 0.65])
    fb = o3d.geometry.TriangleMesh.create_box(0.010, 0.050, 0.025)
    fb.translate((0.020, -0.005, 0.0)); fb.paint_uniform_color([0.60, 0.62, 0.65])
    return _with_table(_sample(fa + fb, 20_000))


def scene_u_channel() -> o3d.geometry.PointCloud:
    wall_h, wall_t, base_w, length = 0.020, 0.003, 0.060, 0.060
    base = o3d.geometry.TriangleMesh.create_box(length, base_w, wall_t)
    base.translate((-length / 2, -base_w / 2, 0.0))
    lw = o3d.geometry.TriangleMesh.create_box(length, wall_t, wall_h)
    lw.translate((-length / 2, -base_w / 2, wall_t))
    rw = o3d.geometry.TriangleMesh.create_box(length, wall_t, wall_h)
    rw.translate((-length / 2, base_w / 2 - wall_t, wall_t))
    mesh = base + lw + rw; mesh.paint_uniform_color([0.78, 0.80, 0.85])
    return _with_table(_sample(mesh, 24_000))


def scene_hollow_tube() -> o3d.geometry.PointCloud:
    R90y = o3d.geometry.get_rotation_matrix_from_xyz([0, np.pi / 2, 0])
    outer_r, inner_r, length = 0.018, 0.013, 0.070
    outer = o3d.geometry.TriangleMesh.create_cylinder(radius=outer_r, height=length, resolution=48)
    outer.rotate(R90y, center=(0, 0, 0)); outer.translate((0, 0, outer_r))
    inner = o3d.geometry.TriangleMesh.create_cylinder(radius=inner_r, height=length + 0.001, resolution=48)
    inner.rotate(R90y, center=(0, 0, 0)); inner.translate((0, 0, outer_r))
    outer_pcd = _sample(outer, 14_000)
    inner_pcd = _sample(inner, 7_000)
    inner_pcd.normals = o3d.utility.Vector3dVector(-np.asarray(inner_pcd.normals))
    obj_pcd = outer_pcd + inner_pcd; obj_pcd.paint_uniform_color([0.52, 0.58, 0.63])
    return _with_table(obj_pcd)


def scene_rect_ring() -> o3d.geometry.PointCloud:
    outer_x, outer_y, height, wall = 0.060, 0.040, 0.012, 0.007
    outer = o3d.geometry.TriangleMesh.create_box(outer_x, outer_y, height)
    outer.translate((-outer_x / 2, -outer_y / 2, 0.0))
    inner_x, inner_y = outer_x - 2 * wall, outer_y - 2 * wall
    inner = o3d.geometry.TriangleMesh.create_box(inner_x, inner_y, height + 0.002)
    inner.translate((-inner_x / 2, -inner_y / 2, -0.001))
    outer_pcd = _sample(outer, 14_000)
    inner_pcd = _sample(inner, 8_000)
    inner_pcd.normals = o3d.utility.Vector3dVector(-np.asarray(inner_pcd.normals))
    obj_pcd = outer_pcd + inner_pcd; obj_pcd.paint_uniform_color([0.68, 0.55, 0.38])
    return _with_table(obj_pcd)


def scene_c_clamp() -> o3d.geometry.PointCloud:
    spine = o3d.geometry.TriangleMesh.create_box(0.010, 0.050, 0.060)
    spine.translate((-0.005, -0.025, 0.0))
    top_jaw = o3d.geometry.TriangleMesh.create_box(0.040, 0.050, 0.008)
    top_jaw.translate((-0.005, -0.025, 0.052))
    bot_jaw = o3d.geometry.TriangleMesh.create_box(0.040, 0.050, 0.008)
    bot_jaw.translate((-0.005, -0.025, 0.0))
    mesh = spine + top_jaw + bot_jaw; mesh.paint_uniform_color([0.30, 0.30, 0.32])
    return _with_table(_sample(mesh, 24_000))


# ─────────────────────────────────────────────────────────────────────────────
# Scene registry
# ─────────────────────────────────────────────────────────────────────────────

SCENES: dict[str, tuple] = {
    "screwdriver":   (scene_screwdriver,   "Cylindrical handle lying on side"),
    "throttle_stop": (scene_throttle_stop, "Flat base + angled arm (sheet metal)"),
    "hex_bolt":      (scene_hex_bolt,      "Upright bolt with hex head"),
    "l_bracket":     (scene_l_bracket,     "Right-angle metal bracket flat"),
    "u_channel":     (scene_u_channel,     "U-shaped aluminium channel, opening up"),
    "hollow_tube":   (scene_hollow_tube,   "Thick-walled hollow tube on side"),
    "rect_ring":     (scene_rect_ring,     "Rectangular ring with square cross-section"),
    "c_clamp":       (scene_c_clamp,       "C-clamp body – tests concave geometry"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SELECTED = "u_channel"   # ← change to test a different scene

    fn, description = SCENES[SELECTED]
    print(f"\n{'═'*62}")
    print(f"  Object : {SELECTED}  –  {description}")
    print(f"{'═'*62}")

    gripper = GripperGeometry(
        max_aperture   = 0.040,
        jaw_depth      = 0.030,
        jaw_thickness  = 0.005,
        palm_thickness = 0.010,
        jaw_half_width = 0.010,
    )
    planner = GraspPlanner(
        gripper        = gripper,
        voxel_size     = 0.003,
        min_z          = 0.005,
        num_seeds      = 500,
        n_approaches   = 6,
        min_contact    = 4,
        min_line       = 0.008,
        squeeze_tol    = 0.0015,
        normal_dot_max = -0.35,
        min_descent    = 0.50,
        score_weights  = {"descent": 2.0, "com": 0.1, "contact": -0.01},
    )

    pcd   = fn()
    grasp = planner.plan(pcd)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    if grasp:
        print(f"\n  Result – {SELECTED}")
        print(f"  Width      : {grasp.width * 100:.2f} cm")
        print(f"  Contact    : {grasp.contact} pts")
        descent_deg = float(np.degrees(np.arcsin(-float(grasp.approach[2]))))
        print(f"  Descent    : {descent_deg:.1f}°  (90° = perfectly vertical, from above)")
        print(f"  Score      : {grasp.score:.4f}  (lower = better)\n")
        geom = planner.visualise(grasp)
        o3d.visualization.draw_geometries(
            [pcd, frame] + geom,
            window_name=f"Grasp – {SELECTED}",
            mesh_show_back_face=True,
        )
    else:
        print(f"\n  No valid grasp found for '{SELECTED}'\n")
        o3d.visualization.draw_geometries(
            [pcd, frame], window_name=f"No grasp – {SELECTED}",
        )