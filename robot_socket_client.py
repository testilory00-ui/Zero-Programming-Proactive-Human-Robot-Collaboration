"""
robot_socket_client.py

TCP socket client for the ABB GoFa pick-and-place RAPID server.
Run AFTER the RAPID module has started ("waiting for Python..." in FlexPendant).

Simulation (RobotStudio virtual controller, same PC):
    HOST = "127.0.0.1", PORT = 5000
Real robot (OmniCore controller):
    HOST = "192.168.125.1", PORT = 1025
"""

import socket


# HOST         = "127.0.0.1"  
# PORT         = 5000           
HOST         = "192.168.125.1"   
PORT         = 1025
RECV_TIMEOUT = 130            # slightly above RAPID's SOCKET_TIMEOUT (120s)
                              # so Python times out AFTER RAPID, not before


class RobotSocketClient:
    """
    TCP client for the RAPID pick-and-place batch server.

    Wire format:
        Python → RAPID : "GP_R1,GP_S1,z_offset|GP_R2,GP_S2,z_offset\n"
        RAPID  → Python: "OK\n"   (after all motions + HomePose)
        Python → RAPID : "HOME\n"
        RAPID  → Python: "OK\n"
        Python → RAPID : "DONE\n"
        RAPID  → Python: "BYE\n"
    """

    def __init__(self, host: str = HOST, port: int = PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(RECV_TIMEOUT)
        self._host = host
        self._port = port
        self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        print(f"[Robot] Connecting to {self._host}:{self._port} ...")
        self._sock.connect((self._host, self._port))
        self._connected = True
        greeting = self._recv_line()
        print(f"[Robot] RAPID: {greeting.strip()}")

    def close(self) -> None:
        if not self._connected:
            return
        self._connected = False
        try:
            self._sock.settimeout(3)     # short deadline — don't wait if robot is down
            self._sock.sendall(b"DONE\n")
            bye = self._recv_line()
            print(f"[Robot] RAPID: {bye.strip()}")
        except Exception:
            pass
        finally:
            # shutdown(SHUT_RDWR) unblocks any thread currently blocked in recv()
            # on this socket (robot_command_worker waiting for "OK").
            # Without this, the worker thread keeps the process alive on Ctrl+C / ESC.
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()
        print("[Robot] Connection closed.")

    # ── Commands ──────────────────────────────────────────────────────────────

    def execute_batch(self, commands: list[tuple[str, str, int]]) -> bool:
        """
        Send commands to RAPID, splitting into ≤79-char chunks (RAPID string cap is 80).
        Each chunk is acknowledged with "OK" before the next is sent.
        Wire format: "GP_R1,GP_S1,0|GP_R2,GP_S2,30\n" — no trailing pipe (would create spurious RAPID token).
        """
        if not commands:
            return self.return_home()

        MAX_LEN = 79  # RAPID string cap (80) minus 1 for the \n terminator
        chunks: list[list[str]] = []
        current: list[str] = []
        current_len = 0
        for pick, place, z_offset in commands:
            token = f"{pick},{place},{z_offset}"
            separator = 1 if current else 0   # "|" between tokens
            if current and current_len + separator + len(token) > MAX_LEN:
                chunks.append(current)
                current = [token]
                current_len = len(token)
            else:
                current.append(token)
                current_len += separator + len(token)
        if current:
            chunks.append(current)

        for chunk in chunks:
            batch = "|".join(chunk)   # no trailing pipe
            msg   = batch + "\n"
            print(f"[Robot] Sending {len(chunk)} command(s): {batch}")
            self._sock.sendall(msg.encode("utf-8"))
            reply = self._recv_line()
            print(f"[Robot] RAPID: {reply.strip()}")
            if reply.strip() != "OK":
                return False
        return True

    def return_home(self) -> bool:
        """Move robot to HomePose without any pick-and-place."""
        print("[Robot] Sending HOME...")
        self._sock.sendall(b"HOME\n")
        reply = self._recv_line()
        print(f"[Robot] RAPID: {reply.strip()}")
        return reply.strip() == "OK"

    def execute_batch_from_commands(self,
                                    bring_commands: list[dict],
                                    remove_commands: list[dict]) -> bool:
        """Flatten bring/remove command dicts to (pick_gp, place_gp, z_offset) triples and send."""
        all_cmds = bring_commands + remove_commands
        if not all_cmds:
            return self.return_home()

        triples = [
            (cmd["pick_gp_id"], cmd["place_gp_id"], cmd.get("z_offset", 0))
            for cmd in all_cmds
        ]
        return self.execute_batch(triples)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _recv_line(self) -> str:
        """Read one newline-terminated message from the TCP stream."""
        data = b""
        while True:
            chunk = self._sock.recv(1)
            if not chunk:
                raise ConnectionError("RAPID closed the connection.")
            data += chunk
            if data.endswith(b"\n"):
                break
        return data.decode("utf-8")