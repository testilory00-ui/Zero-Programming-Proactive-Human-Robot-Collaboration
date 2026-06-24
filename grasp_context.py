"""
grasp_context.py  —  runtime GP data provider for the inference script.

GP types in gp_data:
  "standard"          — regular object slot, YOLO-matched by bbox, green
  "screwdriver"       — dedicated screwdriver slot, yellow, grasp at center
  "screws_container"  — dedicated screws container, orange, grasp at edge
  "general_container" — container routed zone-to-zone; two per zone supported

Screws, screwdriver, and general container are found by type, not by string matching:
  find_screws_container(zone)              → CTR_SCREW_S or CTR_SCREW_R
  find_screwdriver_gp(zone)               → GP_SCREWDRIVER_S (shared) or GP_SCREWDRIVER_R (remote)
  find_general_container(zone, excluded)  → first free CTR_GENERAL_* in zone
  find_paired_general_container(gp_id)   → CTR_GENERAL_R1↔S1, CTR_GENERAL_R2↔S2 (paired routing)
"""

import re
from sheet_and_calibration import run_check, get_gp_data, snap_with_overlay


class GraspContext:
    """
    Single shared object that holds GP data for the entire session.
    Instantiate once in main(), pass to robot_command_worker.
    Thread safety: gp_data is read-only after setup() returns.
    """

    def __init__(self, robot_serial: str | None = None):
        self._gp_data: dict | None = None
        self._robot_serial = robot_serial

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self, force_recalibrate: bool = False) -> None:
        """
        Opens robot camera, shows GP overlay, waits for operator Q to confirm.
        Must be called BEFORE pipeline_robot.start() in main().
        """
        self._gp_data = run_check(
            robot_serial=self._robot_serial,
            force_recalibrate=force_recalibrate
        )

    def setup_silent(self) -> None:
        """Loads saved homography without showing any window."""
        self._gp_data = get_gp_data()

    def _require_setup(self):
        if self._gp_data is None:
            raise RuntimeError(
                "GraspContext.setup() was not called. "
                "Call it in main() before starting threads."
            )

    # ── Basic accessors ───────────────────────────────────────────────────────

    @property
    def gp_data(self) -> dict:
        """Full GP dict keyed by GP id. Each entry has: type, zone, content, px_center, px_grasp, label."""
        self._require_setup()
        return self._gp_data

    def get_gp(self, gp_id: str) -> dict:
        self._require_setup()
        return self._gp_data[gp_id]

    def get_zone(self, zone: str) -> dict:
        """Return all GPs in a zone ('shared' or 'remote')."""
        self._require_setup()
        return {k: v for k, v in self._gp_data.items()
                if v["zone"] == zone}

    # ── Dedicated GP lookups ──────────────────────────────────────────────────

    def find_screws_container(self, zone: str) -> dict | None:
        """Return the screws_container GP in `zone`, or None."""
        self._require_setup()
        for gp_id, info in self.get_zone(zone).items():
            if info["type"] == "screws_container":
                return {"id": gp_id, **info}
        return None

    def find_screwdriver_gp(self, zone: str) -> dict | None:
        """Return the screwdriver GP in `zone`, or None."""
        self._require_setup()
        for gp_id, info in self.get_zone(zone).items():
            if info["type"] == "screwdriver":
                return {"id": gp_id, **info}
        return None

    def find_general_container(self, zone: str, excluded: set | None = None) -> dict | None:
        """Return a free general_container GP in `zone`, or None.
        When multiple general_container GPs exist in the zone, skips any whose id is in `excluded`."""
        self._require_setup()
        for gp_id, info in self.get_zone(zone).items():
            if info["type"] == "general_container":
                if excluded and gp_id in excluded:
                    continue
                return {"id": gp_id, **info}
        return None

    def find_paired_general_container(self, source_gp_id: str) -> dict | None:
        """Return the zone-paired general_container GP for `source_gp_id`, or None.
        Pairing is determined by the trailing zone letter and index:
          CTR_GENERAL_R1 ↔ CTR_GENERAL_S1
          CTR_GENERAL_R2 ↔ CTR_GENERAL_S2
        """
        self._require_setup()
        m = re.search(r'_([RS])(\d+)$', source_gp_id)
        if not m:
            return None
        zone_char, num = m.group(1), m.group(2)
        paired_char = 'S' if zone_char == 'R' else 'R'
        target_id = source_gp_id[:m.start()] + f'_{paired_char}{num}'
        if target_id in self._gp_data:
            return {"id": target_id, **self._gp_data[target_id]}
        return None

    # ── Standard GP helpers ───────────────────────────────────────────────────

    def find_nearest_gp(self,
                        px: tuple,
                        zone: str,
                        tolerance_px: int = 140,
                        type_filter: str | None = None) -> dict | None:
        """
        Match a YOLO bbox center to the nearest GP within tolerance.
        type_filter="standard" also matches general_container by distance.
        Do not use for container/screwdriver routing — use dedicated methods.
        """
        self._require_setup()
        best_id, best_dist = None, float("inf")
        for gp_id, info in self.get_zone(zone).items():
            if type_filter and info["type"] != type_filter:
                # general_container participates in distance-based detection like a standard GP
                if not (type_filter == "standard" and info["type"] == "general_container"):
                    continue
            u, v = info["px_center"]
            dist = ((px[0] - u) ** 2 + (px[1] - v) ** 2) ** 0.5
            if dist < tolerance_px and dist < best_dist:
                best_dist = dist
                best_id   = gp_id
        if best_id is None:
            return None
        return {"id": best_id, **self._gp_data[best_id]}

    def get_free_gps(self,
                     zone: str,
                     occupied: set,
                     type_filter: str | None = None) -> dict:
        """Return unoccupied standard GPs in zone. Containers and screwdriver are always excluded."""
        excluded = {"screws_container", "screwdriver", "general_container"}
        return {
            k: v for k, v in self.get_zone(zone).items()
            if k not in occupied
            and v["type"] not in excluded
            and (type_filter is None or v["type"] == type_filter)
        }

    # ── Camera snapshot ───────────────────────────────────────────────────────

    def snap(self) -> tuple:
        """Grab one frame. Returns (raw_frame, annotated_frame). Do not call while pipeline_robot is running."""
        self._require_setup()
        return snap_with_overlay(self._gp_data,
                                 robot_serial=self._robot_serial)
