import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

from PIL import Image, ImageTk

from gamConsole.cvgames_launcher import (
    cvgames_binary,
    is_webcam_games_installed,
    open_repo_in_browser,
)
from gamConsole.game_launcher import launch_game_entry
from gamConsole.games_registry import GameEntry, games_for_menu
from gamConsole.webcam_preview import WebcamPreview

PREVIEW_W = 320
PREVIEW_H = 240
TICK_MS = 33

BG = "#d4eeff"
BG_CARD = "#eaf6ff"
FG = "#1e466e"
FG_MUTED = "#3d6f99"
PREVIEW_BG = "#c5e4fb"


class GamConsoleMenu:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("gamConsole")
        self.root.minsize(720, 520)
        self.root.configure(bg=BG)

        self.camera_var = tk.IntVar(value=0)
        self.preview = WebcamPreview(0)
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._preview_job: Optional[str] = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _frame(self, parent: tk.Misc, **kwargs) -> tk.Frame:
        return tk.Frame(parent, bg=BG, **kwargs)

    def _label(self, parent: tk.Misc, text: str, *, muted: bool = False, font=None) -> tk.Label:
        return tk.Label(
            parent,
            text=text,
            bg=BG,
            fg=FG_MUTED if muted else FG,
            font=font or ("Helvetica", 11),
            anchor=tk.W,
            justify=tk.LEFT,
        )

    def _build_ui(self) -> None:
        outer = self._frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        self._label(outer, "gamConsole", font=("Helvetica", 22, "bold")).pack(anchor=tk.W)
        self._label(
            outer,
            "Vælg et webcam-spil. Bubble / 3D kræver CvGames fra webcam-games.",
            muted=True,
        ).pack(anchor=tk.W, pady=(0, 12))

        body = self._frame(outer)
        body.pack(fill=tk.BOTH, expand=True)

        left = self._frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        self._label(left, "Spil", muted=True).pack(anchor=tk.W, pady=(0, 6))
        list_frame = tk.Frame(left, bg=BG_CARD)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0, ipadx=12, ipady=12)

        for game in games_for_menu():
            self._add_game_row(list_frame, game)

        tools = self._frame(left)
        tools.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(tools, text="Åbn webcam-games på GitHub", command=open_repo_in_browser).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        status = "CvGames: klar" if cvgames_binary() else (
            "CvGames: ikke bygget" if is_webcam_games_installed() else "CvGames: ikke installeret"
        )
        self._label(tools, status, muted=True).pack(side=tk.LEFT)

        right = self._frame(body, width=PREVIEW_W + 32)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        self._label(right, "Webcam", muted=True).pack(anchor=tk.W)
        preview_box = tk.Frame(right, bg=PREVIEW_BG, width=PREVIEW_W, height=PREVIEW_H)
        preview_box.pack(pady=8)
        preview_box.pack_propagate(False)

        self.preview_label = tk.Label(
            preview_box,
            bg=PREVIEW_BG,
            fg=FG_MUTED,
            text="Forhåndsvisning",
            width=1,
            height=1,
        )
        self.preview_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        cam_row = self._frame(right)
        cam_row.pack(fill=tk.X)
        self._label(cam_row, "Kamera-index:", muted=True).pack(side=tk.LEFT)
        spin = ttk.Spinbox(
            cam_row,
            from_=0,
            to=4,
            width=4,
            textvariable=self.camera_var,
            command=self._on_camera_change,
        )
        spin.pack(side=tk.LEFT, padx=8)
        ttk.Button(cam_row, text="Genstart preview", command=self._restart_preview).pack(side=tk.LEFT)

        self.root.update_idletasks()
        self.root.lift()
        self.root.after(250, self._restart_preview)

    def _add_game_row(self, parent: tk.Frame, game: GameEntry) -> None:
        row = tk.Frame(parent, bg=BG_CARD)
        row.pack(fill=tk.X, pady=6)
        tk.Label(
            row,
            text=game.title,
            bg=BG_CARD,
            fg=FG,
            font=("Helvetica", 13, "bold"),
            anchor=tk.W,
        ).pack(anchor=tk.W)
        tk.Label(
            row,
            text=game.description,
            bg=BG_CARD,
            fg=FG_MUTED,
            font=("Helvetica", 11),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=380,
        ).pack(anchor=tk.W)
        ttk.Button(row, text="Spil", command=lambda g=game: self._play(g)).pack(anchor=tk.W, pady=(4, 0))

    def _play(self, game: GameEntry) -> None:
        if game.mode_arg is None:
            messagebox.showinfo("gamConsole", "Dette spil er ikke tilkoblet endnu.")
            return
        cam = int(self.camera_var.get())
        ok, msg = launch_game_entry(game, camera=cam)
        if ok:
            messagebox.showinfo("gamConsole", msg)
        else:
            messagebox.showwarning("CvGames", msg)

    def _on_camera_change(self) -> None:
        self.preview.camera_index = int(self.camera_var.get())

    def _restart_preview(self) -> None:
        self._stop_preview_loop()
        self.preview.camera_index = int(self.camera_var.get())
        ok, err = self.preview.start()
        if not ok:
            self._photo = None
            self.preview_label.configure(image="", text=err)
            return
        self.preview_label.configure(text="")
        self._preview_tick()

    def _preview_tick(self) -> None:
        rgb = self.preview.read_rgb_array(PREVIEW_W, PREVIEW_H)
        if rgb is not None:
            image = Image.fromarray(rgb)
            self._photo = ImageTk.PhotoImage(image=image)
            self.preview_label.configure(image=self._photo, text="")
        self._preview_job = self.root.after(TICK_MS, self._preview_tick)

    def _stop_preview_loop(self) -> None:
        if self._preview_job is not None:
            self.root.after_cancel(self._preview_job)
            self._preview_job = None

    def _on_close(self) -> None:
        self._stop_preview_loop()
        self.preview.stop()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def run_menu() -> None:
    GamConsoleMenu().run()
