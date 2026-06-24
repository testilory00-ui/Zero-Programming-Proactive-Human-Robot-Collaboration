"""
Annotation tool for post-experiment labelling of prediction JSONL logs.

For each call record it shows:
  - The captured frame (VLM calls) with zoom/pan, or a summary panel (LLM calls)
  - The Bayesian step-probability prior fed into the model (input.step_probabilities)
  - The model's output: stage_of_assembly + next_operation
  - Controls to set ground_truth_step (1–5 / none) and prediction_correct (True/False/unclear)

NOTE on step_probabilities
  These are the Bayesian PRIOR at the moment of the call — they encode what the
  system believed was the current assembly step BEFORE running inference.
  They are NOT confidence scores from the model output.

Keyboard shortcuts
  1–5          set ground_truth_step to that step number
  0            set ground_truth_step to "none" (between / outside assembly)
  y            mark prediction_correct = True
  n            mark prediction_correct = False
  u            mark prediction_correct = "unclear"
  Left/Right   navigate without saving
  r            reset zoom/pan on the image
  s            force-save

Image controls (VLM calls)
  Scroll wheel   zoom in / out centred on cursor
  Click + drag   pan when zoomed in
"""

import json
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

LOGS_DIR    = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "learned_memory.json")

# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_memory() -> dict[int, str]:
    try:
        with open(MEMORY_PATH, encoding="utf-8") as f:
            steps = json.load(f)
        return {s["step number"]: s["step description"] for s in steps}
    except Exception:
        return {}


def load_summary(predictions_path: str) -> dict:
    """Load the _summary.json that corresponds to a _predictions.jsonl file."""
    summary_path = predictions_path.replace("_predictions.jsonl", "_summary.json")
    try:
        with open(summary_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def find_trials() -> list[str]:
    trials = []
    if not os.path.isdir(LOGS_DIR):
        return trials
    for name in sorted(os.listdir(LOGS_DIR)):
        if name.endswith("_predictions.jsonl"):
            trials.append(os.path.join(LOGS_DIR, name))
    return trials


def annotation_stats(records: list[dict]) -> tuple[int, int]:
    preds = [r for r in records if "event_type" not in r]
    annotated = sum(
        1 for r in preds
        if r.get("ground_truth_step") is not None
        and r.get("prediction_correct") is not None
    )
    return annotated, len(preds)


# ── main app ──────────────────────────────────────────────────────────────────

class AnnotatorApp(tk.Tk):
    BAR_H  = 120   # height of the probability canvas (px)

    def __init__(self):
        super().__init__()
        self.title("Prediction Annotator")
        self.resizable(True, True)

        self.steps = load_memory()
        self.trials = find_trials()
        self.records: list[dict] = []       # ALL records (incl. events) — written back on save
        self.pred_records: list[dict] = []  # prediction records only (no event_type)
        self.dispatched_ids: set = set()    # call_ids that were dispatched to robot
        self.current_path: str = ""
        self.idx: int = 0                   # index into pred_records

        # image / zoom state
        self._current_pil: Image.Image | None = None
        self._photo = None          # keep ImageTk reference alive
        self._zoom   = 1.0
        self._pan_x  = 0
        self._pan_y  = 0
        self._drag_start: tuple | None = None
        self._drag_pan_origin: tuple   = (0, 0)

        self._last_probs: dict = {}
        self._dirty = False

        self._build_ui()
        self._bind_keys()

        if self.trials:
            self._load_trial(self.trials[0])
        else:
            self._open_file()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── toolbar ───────────────────────────────────────────────────────────
        tb = tk.Frame(self, bd=1, relief=tk.SUNKEN)
        tb.pack(side=tk.TOP, fill=tk.X)

        tk.Button(tb, text="Open file…", command=self._open_file).pack(side=tk.LEFT, padx=4, pady=2)
        tk.Label(tb, text="Trial:").pack(side=tk.LEFT, padx=(8, 2))

        self.trial_var = tk.StringVar()
        self.trial_cb  = ttk.Combobox(tb, textvariable=self.trial_var, width=52, state="readonly")
        self.trial_cb["values"] = [os.path.basename(p) for p in self.trials]
        self.trial_cb.pack(side=tk.LEFT, padx=2)
        self.trial_cb.bind("<<ComboboxSelected>>", self._on_trial_selected)

        self.score_lbl = tk.Label(tb, text="Execution score: —",
                                   font=("Helvetica", 9, "bold"), fg="#1a5276")
        self.score_lbl.pack(side=tk.RIGHT, padx=12)
        self.progress_lbl = tk.Label(tb, text="0 / 0 annotated", font=("Helvetica", 9))
        self.progress_lbl.pack(side=tk.RIGHT, padx=8)
        self.progress_bar = ttk.Progressbar(tb, length=180, mode="determinate")
        self.progress_bar.pack(side=tk.RIGHT, padx=4)

        # ── PanedWindow: left (image) | right (controls) ──────────────────────
        paned = tk.PanedWindow(self, orient=tk.HORIZONTAL,
                               sashwidth=6, sashrelief=tk.RAISED,
                               bg="#cccccc")
        paned.pack(fill=tk.BOTH, expand=True)

        # ── LEFT pane ─────────────────────────────────────────────────────────
        left_frame = tk.Frame(paned, bg="#1a1a1a")
        paned.add(left_frame, stretch="always")

        # zoom label
        zoom_bar = tk.Frame(left_frame, bg="#111111")
        zoom_bar.pack(side=tk.TOP, fill=tk.X)
        self.zoom_lbl = tk.Label(zoom_bar, text="zoom: 100%  |  scroll to zoom  |  drag to pan  |  r = reset",
                                 bg="#111111", fg="#aaaaaa", font=("Helvetica", 8))
        self.zoom_lbl.pack(side=tk.LEFT, padx=6, pady=1)

        # image canvas
        self.img_canvas = tk.Canvas(left_frame, bg="#1a1a1a",
                                    cursor="crosshair", highlightthickness=0)
        self.img_canvas.pack(fill=tk.BOTH, expand=True)
        self.img_canvas.bind("<Configure>",        self._on_canvas_resize)
        self.img_canvas.bind("<MouseWheel>",        self._on_zoom)         # Windows / macOS
        self.img_canvas.bind("<Button-4>",          self._on_zoom)         # Linux scroll up
        self.img_canvas.bind("<Button-5>",          self._on_zoom)         # Linux scroll down
        self.img_canvas.bind("<ButtonPress-1>",     self._on_pan_start)
        self.img_canvas.bind("<B1-Motion>",         self._on_pan_move)
        self.img_canvas.bind("<ButtonRelease-1>",   lambda _: setattr(self, "_drag_start", None))

        # LLM text widget (shown instead of canvas for LLM calls)
        self.llm_info = tk.Text(left_frame, wrap=tk.WORD, state=tk.DISABLED,
                                bg="#1a1a1a", fg="#e0e0e0",
                                font=("Courier", 11), height=12)

        # ── RIGHT pane ────────────────────────────────────────────────────────
        right_outer = tk.Frame(paned)
        paned.add(right_outer, stretch="always")

        # scrollable inner frame
        vscroll = ttk.Scrollbar(right_outer, orient=tk.VERTICAL)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        sc = tk.Canvas(right_outer, yscrollcommand=vscroll.set,
                       borderwidth=0, highlightthickness=0)
        sc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vscroll.config(command=sc.yview)

        right = tk.Frame(sc)
        win   = sc.create_window((0, 0), window=right, anchor=tk.NW)

        def _update_scroll(event=None):
            sc.configure(scrollregion=sc.bbox("all"))
            sc.itemconfig(win, width=sc.winfo_width())
        right.bind("<Configure>", _update_scroll)
        sc.bind("<Configure>",   _update_scroll)

        # mouse-wheel scrolling on the right pane
        def _scroll_right(event):
            sc.yview_scroll(int(-1 * (event.delta / 120)), "units")
        right.bind_all("<Control-MouseWheel>", _scroll_right)

        # ── call info ──────────────────────────────────────────────────────────
        hdr = tk.LabelFrame(right, text="Call info", padx=6, pady=4)
        hdr.pack(fill=tk.X, padx=6, pady=(6, 2))

        self.call_lbl = tk.Label(hdr, text="", anchor=tk.W,
                                 font=("Helvetica", 10, "bold"), wraplength=460)
        self.call_lbl.pack(fill=tk.X)
        self.action_lbl = tk.Label(hdr, text="", anchor=tk.W,
                                   font=("Helvetica", 9), fg="#333", wraplength=460)
        self.action_lbl.pack(fill=tk.X)

        # ── probability bars ───────────────────────────────────────────────────
        prob_frame = tk.LabelFrame(
            right,
            text="step_probabilities  [Bayesian prior BEFORE this call — not model confidence]",
            padx=6, pady=4,
        )
        prob_frame.pack(fill=tk.X, padx=6, pady=2)
        self.prob_canvas = tk.Canvas(prob_frame, height=self.BAR_H, bg="white",
                                     highlightthickness=0)
        self.prob_canvas.pack(fill=tk.X)
        self.prob_canvas.bind("<Configure>", lambda _: self._redraw_probs())

        # ── model output ───────────────────────────────────────────────────────
        out_frame = tk.LabelFrame(right, text="Model output", padx=6, pady=4)
        out_frame.pack(fill=tk.X, padx=6, pady=2)

        self.stage_lbl = tk.Label(out_frame, text="", anchor=tk.W,
                                  font=("Helvetica", 9), fg="#1a5276", wraplength=460)
        self.stage_lbl.pack(fill=tk.X)
        self.next_lbl  = tk.Label(out_frame, text="", anchor=tk.W,
                                  font=("Helvetica", 9, "bold"), fg="#1a5276", wraplength=460)
        self.next_lbl.pack(fill=tk.X)

        # ── ground truth step ──────────────────────────────────────────────────
        gt_frame = tk.LabelFrame(right, text="Ground truth step  [keys 1–5, 0=none]",
                                 padx=6, pady=4)
        gt_frame.pack(fill=tk.X, padx=6, pady=2)

        self.gt_var = tk.StringVar(value="")
        gt_grid = tk.Frame(gt_frame)
        gt_grid.pack(fill=tk.X)
        gt_grid.columnconfigure(0, weight=1)
        gt_grid.columnconfigure(1, weight=1)
        gt_grid.columnconfigure(2, weight=1)

        step_colors = ["#d4efdf", "#d6eaf8", "#fdebd0", "#e8daef", "#f0f0f0"]
        positions   = [(1, 0, 0), (2, 0, 1), (3, 0, 2), (4, 1, 0), (5, 1, 1)]
        for step, row, col in positions:
            desc = self.steps.get(step, f"Step {step}")
            b = tk.Radiobutton(
                gt_grid, text=f"[{step}]  {desc}",
                variable=self.gt_var, value=str(step),
                indicatoron=False, bg=step_colors[step - 1],
                activebackground=step_colors[step - 1],
                relief=tk.RAISED, padx=6, pady=5,
                font=("Helvetica", 9), justify=tk.LEFT, anchor=tk.W,
                wraplength=130,
                command=self._on_annotation_change,
            )
            b.grid(row=row, column=col, sticky=tk.NSEW, padx=2, pady=2)

        b_none = tk.Radiobutton(
            gt_grid, text="[0]  None / between",
            variable=self.gt_var, value="none",
            indicatoron=False, bg="#e8e8e8", activebackground="#d5d5d5",
            relief=tk.RAISED, padx=6, pady=5,
            font=("Helvetica", 9), anchor=tk.W,
            command=self._on_annotation_change,
        )
        b_none.grid(row=1, column=2, sticky=tk.NSEW, padx=2, pady=2)

        self.gt_current_lbl = tk.Label(gt_frame, text="Current: (not set)",
                                       fg="#c0392b", font=("Helvetica", 9, "italic"))
        self.gt_current_lbl.pack(anchor=tk.W, pady=(2, 0))

        # ── prediction correct ─────────────────────────────────────────────────
        pc_frame = tk.LabelFrame(right, text="Prediction correct?  [y=yes  n=no  u=unclear]",
                                 padx=6, pady=4)
        pc_frame.pack(fill=tk.X, padx=6, pady=2)

        self.pc_var = tk.StringVar(value="")
        pc_row = tk.Frame(pc_frame)
        pc_row.pack(fill=tk.X)
        pc_row.columnconfigure(0, weight=1)
        pc_row.columnconfigure(1, weight=1)
        pc_row.columnconfigure(2, weight=1)

        for col, (val, label, bg) in enumerate([
            ("True",    "✓  Correct  [y]", "#a9dfbf"),
            ("False",   "✗  Wrong  [n]",   "#f1948a"),
            ("unclear", "?  Unclear  [u]", "#f9e79f"),
        ]):
            b = tk.Radiobutton(
                pc_row, text=label,
                variable=self.pc_var, value=val,
                indicatoron=False, bg=bg, activebackground=bg,
                relief=tk.RAISED, padx=8, pady=6,
                font=("Helvetica", 10),
                command=self._on_annotation_change,
            )
            b.grid(row=0, column=col, sticky=tk.NSEW, padx=3, pady=2)

        self.pc_current_lbl = tk.Label(pc_frame, text="Current: (not set)",
                                       fg="#c0392b", font=("Helvetica", 9, "italic"))
        self.pc_current_lbl.pack(anchor=tk.W, pady=(2, 0))

        # ── navigation ─────────────────────────────────────────────────────────
        nav = tk.Frame(right)
        nav.pack(fill=tk.X, padx=6, pady=(10, 4))

        tk.Button(nav, text="◀ Prev  [←]", command=self._prev, width=14).pack(side=tk.LEFT, padx=4)
        tk.Button(nav, text="Next  [→] ▶", command=self._next, width=14).pack(side=tk.LEFT, padx=4)
        self.nav_lbl = tk.Label(nav, text="0 / 0", font=("Helvetica", 9))
        self.nav_lbl.pack(side=tk.LEFT, padx=8)
        tk.Button(nav, text="Save  [s]", command=self._save, bg="#a9cce3").pack(side=tk.RIGHT, padx=4)

        self.status_lbl = tk.Label(right, text="", fg="#555", font=("Helvetica", 9, "italic"))
        self.status_lbl.pack(anchor=tk.W, padx=6, pady=(0, 6))

        # set initial 50/50 split after window is drawn
        self.after(100, lambda: paned.sash_place(0, self.winfo_width() // 2, 0))

    def _bind_keys(self):
        self.bind("<Left>",  lambda _: self._prev())
        self.bind("<Right>", lambda _: self._next())
        self.bind("<s>",     lambda _: self._save())
        self.bind("<r>",     lambda _: self._reset_zoom())
        for i in range(1, 6):
            self.bind(str(i), lambda _, v=str(i): self._set_gt(v))
        self.bind("0", lambda _: self._set_gt("none"))
        self.bind("y", lambda _: self._set_pc("True"))
        self.bind("n", lambda _: self._set_pc("False"))
        self.bind("u", lambda _: self._set_pc("unclear"))

    # ── loading ───────────────────────────────────────────────────────────────

    def _open_file(self):
        path = filedialog.askopenfilename(
            initialdir=LOGS_DIR,
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")],
        )
        if path:
            self._load_trial(path)

    def _on_trial_selected(self, _=None):
        i = self.trial_cb.current()
        if 0 <= i < len(self.trials):
            self._load_trial(self.trials[i])

    def _load_trial(self, path: str):
        if self._dirty:
            if messagebox.askyesno("Unsaved changes", "Save before switching?"):
                self._save()
        self.current_path = path
        self.records      = load_jsonl(path)
        self._dirty       = False

        # Separate prediction records from event records (dispatch, validation_failure, …)
        self.pred_records  = [r for r in self.records if "event_type" not in r]
        self.dispatched_ids = {
            r.get("dispatched_from_call_id")
            for r in self.records
            if r.get("event_type") == "dispatch"
        }

        summary = load_summary(path)
        score   = summary.get("execution_success_score")
        if score is None:
            self.score_lbl.config(text="Execution score: —", fg="#888888")
        else:
            self.score_lbl.config(text=f"Execution score: {score:.2f}", fg="#1a5276")

        base  = os.path.basename(path)
        names = list(self.trial_cb["values"])
        if base in names:
            self.trial_cb.current(names.index(base))

        self.idx = 0
        for i, r in enumerate(self.pred_records):
            if r.get("ground_truth_step") is None or r.get("prediction_correct") is None:
                self.idx = i
                break

        self._refresh()

    # ── display ───────────────────────────────────────────────────────────────

    def _refresh(self):
        if not self.pred_records:
            return
        r = self.pred_records[self.idx]

        call_type = r.get("type", "?").upper()
        cid = r.get("call_id", "?")
        ts  = r.get("timestamp", "")[:19]
        lat = r.get("inference_time_s", 0)

        # Build badges for dispatch and hint-injected calls
        badges = []
        if cid in self.dispatched_ids:
            badges.append("[DISPATCHED]")
        if r.get("was_hint_injection"):
            badges.append("[HINT]")
        badge_str = "  " + "  ".join(badges) if badges else ""

        self.call_lbl.config(
            text=f"Call #{cid}  |  {call_type}  |  {ts}  |  {lat:.3f}s{badge_str}"
        )

        inp    = r.get("input", {})
        action = inp.get("semantic_action", "—")
        ctx    = "✓ context" if inp.get("context_available") else "✗ no context"
        self.action_lbl.config(text=f"semantic_action: {action}   ({ctx})")

        self._last_probs = inp.get("step_probabilities", {})
        self._redraw_probs()

        out   = r.get("output", {})
        self.stage_lbl.config(text=f"Stage detected:  {out.get('stage_of_assembly', '—')}")
        self.next_lbl.config( text=f"Next operation:  {out.get('next_operation',    '—')}")

        frame_path = self._resolve_path(inp.get("frame_saved", ""))
        if frame_path:
            self._load_image(frame_path)
        else:
            self._current_pil = None
            self._show_llm_text(r)

        gt = r.get("ground_truth_step")
        pc = r.get("prediction_correct")
        self.gt_var.set("" if gt is None else str(gt))
        self.pc_var.set("" if pc is None else str(pc))
        self._update_annotation_labels()

        ann, total = annotation_stats(self.records)
        self.nav_lbl.config(text=f"{self.idx + 1} / {len(self.pred_records)}")
        self.progress_lbl.config(text=f"{ann} / {total} annotated")
        self.progress_bar["value"] = (ann / total * 100) if total else 0
        self.title(f"Annotator — {os.path.basename(self.current_path)}")

    def _resolve_path(self, raw: str) -> str:
        if not raw:
            return ""
        if os.path.isfile(raw):
            return raw
        # Try relative to the logs/ directory (same dir as this script)
        joined = os.path.join(os.path.dirname(__file__), raw)
        if os.path.isfile(joined):
            return joined
        # Try relative to the project root (parent of logs/).
        # Needed when frame_saved is "logs\xxx_frames\call_001_vlm.jpg"
        # and the script lives inside the logs/ directory.
        project_root = os.path.dirname(LOGS_DIR)
        joined2 = os.path.join(project_root, raw)
        return joined2 if os.path.isfile(joined2) else ""

    # ── probability bars ──────────────────────────────────────────────────────

    def _redraw_probs(self):
        probs = self._last_probs
        c = self.prob_canvas
        c.delete("all")
        c.update_idletasks()
        W = c.winfo_width()
        H = self.BAR_H

        if W < 10:
            self.after(60, self._redraw_probs)
            return

        if not probs:
            c.create_text(W // 2, H // 2 - 8,
                          text="(no probabilities recorded)",
                          fill="#aaa", font=("Helvetica", 10))
            c.create_text(W // 2, H // 2 + 10,
                          text="A/no_context trials have no step tracker — this is expected",
                          fill="#bbb", font=("Helvetica", 8))
            return

        n        = len(probs)
        margin   = 12
        label_h  = 16
        val_h    = 16
        bar_zone = H - label_h - val_h
        slot_w   = (W - 2 * margin) / n

        step_colors = ["#82e0aa", "#7fb3d3", "#f0b27a", "#c39bd3", "#85c1e9"]

        # Scale bars against 1.0 so uniform priors (all ~0.2) don't misleadingly
        # fill the canvas — the absolute height now reflects actual confidence.
        for i, (key, val) in enumerate(sorted(probs.items())):
            cx  = margin + i * slot_w + slot_w / 2
            bw  = slot_w * 0.65
            x0, x1 = cx - bw / 2, cx + bw / 2
            bar_h = int(bar_zone * val)          # absolute scale: 1.0 = full height
            y_bot = H - label_h
            y_top = y_bot - bar_h

            c.create_rectangle(x0, y_top, x1, y_bot,
                               fill=step_colors[i % len(step_colors)], outline="#aaa")
            # value label — clamped so it never leaves the canvas
            c.create_text(cx, max(y_top - 8, val_h - 2),
                          text=f"{val:.2f}", font=("Helvetica", 8), fill="#333")
            c.create_text(cx, H - label_h // 2,
                          text=f"S{key.replace('step_', '')}",
                          font=("Helvetica", 9, "bold"), fill="#444")

    # ── image zoom/pan ────────────────────────────────────────────────────────

    def _load_image(self, path: str):
        self.llm_info.pack_forget()
        self.img_canvas.pack(fill=tk.BOTH, expand=True)
        try:
            self._current_pil = Image.open(path)
        except Exception as e:
            self._current_pil = None
            self.img_canvas.delete("all")
            self.img_canvas.create_text(20, 20, anchor=tk.NW,
                                        text=f"Cannot load:\n{path}\n{e}",
                                        fill="red", font=("Helvetica", 10))
            return
        self._reset_zoom()

    def _reset_zoom(self):
        self._zoom  = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._draw_image()

    def _draw_image(self):
        if self._current_pil is None:
            return
        c = self.img_canvas
        c.update_idletasks()
        W = c.winfo_width()
        H = c.winfo_height()
        if W < 2 or H < 2:
            self.after(60, self._draw_image)
            return

        iw, ih = self._current_pil.size
        # fit-to-canvas base scale, then apply zoom
        base  = min(W / iw, H / ih)
        scale = base * self._zoom
        dw    = max(1, int(iw * scale))
        dh    = max(1, int(ih * scale))

        img   = self._current_pil.resize((dw, dh), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)

        c.delete("all")
        cx = W // 2 + self._pan_x
        cy = H // 2 + self._pan_y
        c.create_image(cx, cy, anchor=tk.CENTER, image=photo)
        self._photo = photo

        self.zoom_lbl.config(text=(
            f"zoom: {int(self._zoom * 100)}%  |  "
            f"scroll to zoom  |  drag to pan  |  r = reset"
        ))

    def _on_canvas_resize(self, _event):
        self._draw_image()

    def _on_zoom(self, event):
        # Windows/macOS: event.delta; Linux Button-4/5
        if event.num == 4:
            delta = 120
        elif event.num == 5:
            delta = -120
        else:
            delta = event.delta

        factor = 1.1 if delta > 0 else (1 / 1.1)
        new_zoom = max(0.2, min(self._zoom * factor, 10.0))

        # zoom centred on cursor: adjust pan so the point under the cursor stays fixed
        c  = self.img_canvas
        W, H = c.winfo_width(), c.winfo_height()
        # cursor position relative to image centre
        mx = event.x - (W // 2 + self._pan_x)
        my = event.y - (H // 2 + self._pan_y)
        scale_ratio = new_zoom / self._zoom
        self._pan_x += int(mx - mx * scale_ratio)
        self._pan_y += int(my - my * scale_ratio)

        self._zoom = new_zoom
        self._draw_image()

    def _on_pan_start(self, event):
        self._drag_start      = (event.x, event.y)
        self._drag_pan_origin = (self._pan_x, self._pan_y)

    def _on_pan_move(self, event):
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._pan_x = self._drag_pan_origin[0] + dx
        self._pan_y = self._drag_pan_origin[1] + dy
        self._draw_image()

    def _show_llm_text(self, r: dict):
        self.img_canvas.pack_forget()
        self.img_canvas.delete("all")
        self.llm_info.pack(fill=tk.BOTH, expand=True)
        inp = r.get("input", {})
        lines = [
            f"  TYPE : {r.get('type','?').upper()}  (LLM call — no frame captured)",
            "",
            f"  semantic_action : {inp.get('semantic_action', '—')}",
            "",
            "  step_probabilities (Bayesian prior — see bar chart →):",
        ]
        for k, v in sorted((inp.get("step_probabilities") or {}).items()):
            bar = "█" * int(v * 40)
            lines.append(f"    {k}: {v:.3f}  {bar}")
        lines += ["", "  [Use keys 1-5, 0, y/n/u to annotate]"]
        self.llm_info.config(state=tk.NORMAL)
        self.llm_info.delete("1.0", tk.END)
        self.llm_info.insert(tk.END, "\n".join(lines))
        self.llm_info.config(state=tk.DISABLED)

    # ── annotation labels ─────────────────────────────────────────────────────

    def _update_annotation_labels(self):
        gt = self.gt_var.get()
        if gt and gt.isdigit():
            desc = self.steps.get(int(gt), "")
            self.gt_current_lbl.config(text=f"Current: step {gt}  —  {desc}", fg="#1a5276")
        elif gt == "none":
            self.gt_current_lbl.config(text="Current: none / between steps", fg="#1a5276")
        else:
            self.gt_current_lbl.config(text="Current: (not set)", fg="#c0392b")

        pc = self.pc_var.get()
        self.pc_current_lbl.config(
            text=f"Current: {pc}" if pc else "Current: (not set)",
            fg="#1a5276" if pc else "#c0392b",
        )

    # ── annotation actions ────────────────────────────────────────────────────

    def _set_gt(self, val: str):
        self.gt_var.set(val)
        self._on_annotation_change()

    def _set_pc(self, val: str):
        self.pc_var.set(val)
        self._on_annotation_change()

    def _on_annotation_change(self):
        if not self.pred_records:
            return
        r      = self.pred_records[self.idx]
        gt_raw = self.gt_var.get()
        pc_raw = self.pc_var.get()

        if gt_raw == "":
            r["ground_truth_step"] = None
        elif gt_raw == "none":
            r["ground_truth_step"] = "none"
        else:
            try:
                r["ground_truth_step"] = int(gt_raw)
            except ValueError:
                r["ground_truth_step"] = gt_raw

        if pc_raw == "":
            r["prediction_correct"] = None
        elif pc_raw == "True":
            r["prediction_correct"] = True
        elif pc_raw == "False":
            r["prediction_correct"] = False
        else:
            r["prediction_correct"] = "unclear"

        self._dirty = True
        self._update_annotation_labels()
        self._update_status()

        if r["ground_truth_step"] is not None and r["prediction_correct"] is not None:
            self._save(silent=True)

    def _update_status(self):
        ann, total = annotation_stats(self.records)
        pct = ann / total * 100 if total else 0
        self.progress_lbl.config(text=f"{ann} / {total} annotated")
        self.progress_bar["value"] = pct
        saved = "" if self._dirty else "  ✓ saved"
        self.status_lbl.config(text=f"{ann}/{total} annotated ({pct:.0f}%){saved}")

    # ── navigation ────────────────────────────────────────────────────────────

    def _prev(self):
        if self.pred_records and self.idx > 0:
            self.idx -= 1
            self._refresh()

    def _next(self):
        if self.pred_records and self.idx < len(self.pred_records) - 1:
            self.idx += 1
            self._refresh()

    def _save(self, silent: bool = False):
        if not self.current_path or not self.records:
            return
        try:
            save_jsonl(self.current_path, self.records)
            self._dirty = False
            self._update_status()
            if not silent:
                self.status_lbl.config(
                    text=f"Saved → {os.path.basename(self.current_path)}"
                )
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def _on_close(self):
        if self._dirty:
            if messagebox.askyesno("Unsaved changes", "Save before quitting?"):
                self._save()
        self.destroy()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = AnnotatorApp()
    app.protocol("WM_DELETE_WINDOW", app._on_close)
    app.geometry("1400x820")
    app.mainloop()
