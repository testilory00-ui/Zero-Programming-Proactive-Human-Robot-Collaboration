"""
test_robot_socket.py

Standalone test script for RobotSocketClient + pick_and_place RAPID module.
No cameras, no perception, no LLM — pure socket communication.

Targets the RobotStudio simulation (127.0.0.1:5000 by default).
To test against the real robot, edit HOST/PORT below.

Usage:
    1. Start RobotStudio simulation and run the pick_and_place RAPID program.
    2. Wait until the FlexPendant shows "waiting for Python...".
    3. Run:  python test_robot_socket.py

Batch progression (simple → challenging):

  Batch 1 — SIMPLE
    1 command, standard object, no Z offset.
    Validates the basic pick-place cycle and OK handshake.

  Batch 2 — MEDIUM
    3 commands: one tall object (diaphragm, z_offset=20), one dedicated-GP
    move (screws container), one return-to-remote.
    Validates Z-offset handling and dedicated GP resolution.

  Batch 3 — CHALLENGING
    6 commands: two tall objects (float_bowl z=30, diaphragm z=20),
    screws container, screwdriver, and two remove commands.
    The 79-char RAPID string limit forces automatic chunk splitting
    (4 commands in chunk 1, 2 in chunk 2) — validates multi-chunk sequencing.
"""

from robot_socket_client import RobotSocketClient

# HOST = "127.0.0.1"   # RobotStudio simulation — change to "192.168.125.1" for real robot
# PORT = 5000           # simulation port         — change to 1025 for real robot
HOST = "192.168.125.1"
PORT = 1025

# ── Batch definitions — tuples are (pick_gp_id, place_gp_id, z_offset_mm) ───
# GP IDs must match robtarget names declared in pick_and_place.modx.
# z_offset > 0: robot grasps that many mm above the nominal GP Z (taller objects).

# ── Batch 1: SIMPLE ───────────────────────────────────────────────────────────
# One bring command, flat object, no height correction.
BATCH_1: list[tuple[str, str, int]] = [
    ("GP_R2", "GP_S2", 75),   # bring standard object: remote GP_R2 → shared GP_S2
]

# ── Batch 2: MEDIUM ───────────────────────────────────────────────────────────
# Three commands:
#   - diaphragm (tall, z_offset=20 mm) from remote to shared
#   - screws container via dedicated GPs (CTR_SCREW_R → CTR_SCREW_S)
#   - return a no-longer-needed object from shared back to remote
BATCH_2: list[tuple[str, str, int]] = [
    ("CTR_GENERAL_R1",        "CTR_GENERAL_S1",         0),  # diaphragm (tall) → shared
    ("CTR_GENERAL_R2", "CTR_GENERAL_S2",   0),  # screws container → shared zone
    ("GP_SCREWDRIVER_S","GP_SCREWDRIVER_R",0),  # return unneeded object to remote
]

# ── Batch 3: CHALLENGING ──────────────────────────────────────────────────────
# Six commands including both tall-object types and the screwdriver.
# Total wire length exceeds RAPID's 80-char limit → auto-splits into 2 chunks:
#   Chunk 1 (4 cmds): float_bowl, diaphragm, screws, screwdriver
#   Chunk 2 (2 cmds): two return-to-remote commands
BATCH_3: list[tuple[str, str, int]] = [
    ("GP_R1",           "GP_S1",          20),  # float_bowl (tall)   → shared
    ("GP_R2",           "GP_S2",         0),  # diaphragm (tall)    → shared
    ("CTR_SCREW_R",     "CTR_SCREW_S",     0),  # screws container    → shared
    ("GP_SCREWDRIVER_R","GP_SCREWDRIVER_S", 0),  # screwdriver         → shared
    ("CTR_SPRING_R",    "CTR_SPRING_S",    0),  # return cover        → remote  (chunk 2)
    ("GP_S3",           "GP_R4",           0),  # return object       → remote  (chunk 2)
]

# ── Menu ──────────────────────────────────────────────────────────────────────

ACTIONS = {
    "1": ("HOME only",                       "home"),
    "2": ("Batch 1 only  — simple",          "batch1"),
    "3": ("Batch 2 only  — medium",          "batch2"),
    "4": ("Batch 3 only  — challenging",     "batch3"),
    "5": ("HOME + Batch 1",                  "home+batch1"),
    "6": ("HOME + Batch 2",                  "home+batch2"),
    "7": ("HOME + Batch 3",                  "home+batch3"),
    "8": ("Full sequence (HOME + all three)", "full"),
}


def prompt_action() -> str:
    print("\nSelect action:")
    for key, (label, _) in ACTIONS.items():
        print(f"  {key}) {label}")
    while True:
        choice = input("Enter number: ").strip()
        if choice in ACTIONS:
            return ACTIONS[choice][1]
        print("  Invalid choice, try again.")


def run_home(robot: RobotSocketClient) -> bool:
    print("\n--- HOME ---")
    ok = robot.return_home()
    print("[Test] Robot is at HomePose." if ok else "[Test] HOME failed.")
    return ok


def run_batch1(robot: RobotSocketClient) -> bool:
    print("\n--- Batch 1: SIMPLE (1 command, no Z offset) ---")
    ok = robot.execute_batch(BATCH_1)
    print("[Test] Batch 1 complete." if ok else "[Test] Batch 1 failed.")
    return ok


def run_batch2(robot: RobotSocketClient) -> bool:
    print("\n--- Batch 2: MEDIUM (3 commands, diaphragm z=20, screws container, remove) ---")
    ok = robot.execute_batch(BATCH_2)
    print("[Test] Batch 2 complete." if ok else "[Test] Batch 2 failed.")
    return ok


def run_batch3(robot: RobotSocketClient) -> bool:
    print("\n--- Batch 3: CHALLENGING (6 commands, two Z offsets, auto-splits into 2 chunks) ---")
    ok = robot.execute_batch(BATCH_3)
    print("[Test] Batch 3 complete." if ok else "[Test] Batch 3 failed.")
    return ok


def main() -> None:
    action = prompt_action()

    steps = {
        "home":        [run_home],
        "batch1":      [run_batch1],
        "batch2":      [run_batch2],
        "batch3":      [run_batch3],
        "home+batch1": [run_home, run_batch1],
        "home+batch2": [run_home, run_batch2],
        "home+batch3": [run_home, run_batch3],
        "full":        [run_home, run_batch1, run_batch2, run_batch3],
    }[action]

    with RobotSocketClient(host=HOST, port=PORT) as robot:
        for step in steps:
            if not step(robot):
                print("[Test] Aborting.")
                return
        print("\n[Test] All steps passed. Disconnecting...")
    # __exit__ sends DONE and closes the socket


if __name__ == "__main__":
    main()
