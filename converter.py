"""
MP4 to MP3 + Transcript GUI
Drag & Drop application with transcription and optional speaker diarization.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError as exc:
    raise SystemExit(
        "tkinterdnd2 がインストールされていません。"
        " `pip install -r requirements.txt` を再実行してください。"
    ) from exc

from transcription_pipeline import (
    DependencyError,
    ExternalToolError,
    MediaTranscriptionProcessor,
    ProcessingCancelled,
    ProcessingError,
    ProcessingOptions,
    is_faster_whisper_available,
    is_ffmpeg_available,
)


STAGE_LABELS = {
    "extracting": "MP3 抽出中…",
    "transcribing": "文字起こし中…",
    "diarizing": "話者分離中…",
    "saving": "保存中…",
}


class MP4toTextConverter(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()

        self.title("MP4 → MP3 → テキスト化")
        self.geometry("820x780")
        self.resizable(True, True)
        self.configure(bg="#1e1e2e")

        self._conversion_queue = queue.Queue()
        self._cancel_flag = False
        self._is_converting = False
        self._file_items: dict[str, str] = {}
        self._shown_messages: set[str] = set()
        self._processor = MediaTranscriptionProcessor()
        self._current_options: ProcessingOptions | None = None
        self._output_setting = "入力ファイルと同じフォルダ"

        self._build_ui()
        self._setup_dnd()

    def _build_ui(self):
        pad = 12

        drop_frame = tk.Frame(
            self,
            bg="#313244",
            bd=0,
            highlightthickness=2,
            highlightbackground="#7c7f9c",
            relief="flat",
        )
        drop_frame.pack(fill="x", padx=pad, pady=(pad, 6))

        self._drop_label = tk.Label(
            drop_frame,
            text="ここに MP4 ファイルをドラッグ＆ドロップ\n（複数ファイル対応）",
            font=("Meiryo", 13),
            fg="#cdd6f4",
            bg="#313244",
            pady=22,
        )
        self._drop_label.pack(fill="both")

        row = tk.Frame(self, bg="#1e1e2e")
        row.pack(fill="x", padx=pad, pady=(0, 6))

        tk.Label(
            row,
            text="出力先:",
            font=("Meiryo", 10),
            fg="#a6adc8",
            bg="#1e1e2e",
        ).pack(side="left")

        self._out_var = tk.StringVar(value="入力ファイルと同じフォルダ")
        self._out_entry = tk.Entry(
            row,
            textvariable=self._out_var,
            font=("Consolas", 9),
            fg="#cdd6f4",
            bg="#313244",
            insertbackground="white",
            relief="flat",
            bd=4,
        )
        self._out_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))

        tk.Button(
            row,
            text="選択…",
            command=self._choose_output,
            font=("Meiryo", 9),
            bg="#45475a",
            fg="#cdd6f4",
            activebackground="#585b70",
            relief="flat",
            cursor="hand2",
            padx=6,
        ).pack(side="left")

        tk.Button(
            row,
            text="クリア",
            command=self._clear_output,
            font=("Meiryo", 9),
            bg="#45475a",
            fg="#cdd6f4",
            activebackground="#585b70",
            relief="flat",
            cursor="hand2",
            padx=6,
        ).pack(side="left", padx=(4, 0))

        bitrate_row = tk.Frame(self, bg="#1e1e2e")
        bitrate_row.pack(fill="x", padx=pad, pady=(0, 6))

        tk.Label(
            bitrate_row,
            text="品質 (kbps):",
            font=("Meiryo", 10),
            fg="#a6adc8",
            bg="#1e1e2e",
        ).pack(side="left")

        self._bitrate_var = tk.StringVar(value="192")
        for bitrate in ("128", "192", "256", "320"):
            tk.Radiobutton(
                bitrate_row,
                text=bitrate,
                variable=self._bitrate_var,
                value=bitrate,
                font=("Meiryo", 10),
                fg="#cdd6f4",
                bg="#1e1e2e",
                activebackground="#1e1e2e",
                activeforeground="#89b4fa",
                selectcolor="#313244",
                relief="flat",
            ).pack(side="left", padx=6)

        model_row = tk.Frame(self, bg="#1e1e2e")
        model_row.pack(fill="x", padx=pad, pady=(0, 6))

        tk.Label(
            model_row,
            text="Whisper モデル:",
            font=("Meiryo", 10),
            fg="#a6adc8",
            bg="#1e1e2e",
        ).pack(side="left")

        self._model_var = tk.StringVar(value="small")
        for model_size in ("small", "medium"):
            tk.Radiobutton(
                model_row,
                text=model_size,
                variable=self._model_var,
                value=model_size,
                font=("Meiryo", 10),
                fg="#cdd6f4",
                bg="#1e1e2e",
                activebackground="#1e1e2e",
                activeforeground="#89b4fa",
                selectcolor="#313244",
                relief="flat",
            ).pack(side="left", padx=6)

        tk.Label(
            model_row,
            text="言語:",
            font=("Meiryo", 10),
            fg="#a6adc8",
            bg="#1e1e2e",
        ).pack(side="left", padx=(18, 0))

        self._language_var = tk.StringVar(value="auto")
        for language in ("auto", "ja"):
            tk.Radiobutton(
                model_row,
                text=language,
                variable=self._language_var,
                value=language,
                font=("Meiryo", 10),
                fg="#cdd6f4",
                bg="#1e1e2e",
                activebackground="#1e1e2e",
                activeforeground="#89b4fa",
                selectcolor="#313244",
                relief="flat",
            ).pack(side="left", padx=6)

        output_row = tk.Frame(self, bg="#1e1e2e")
        output_row.pack(fill="x", padx=pad, pady=(0, 6))

        tk.Label(
            output_row,
            text="出力形式:",
            font=("Meiryo", 10),
            fg="#a6adc8",
            bg="#1e1e2e",
        ).pack(side="left")

        self._format_vars = {
            "txt": tk.BooleanVar(value=True),
            "srt": tk.BooleanVar(value=False),
            "json": tk.BooleanVar(value=False),
        }
        for output_format in ("txt", "srt", "json"):
            tk.Checkbutton(
                output_row,
                text=output_format.upper(),
                variable=self._format_vars[output_format],
                font=("Meiryo", 10),
                fg="#cdd6f4",
                bg="#1e1e2e",
                activebackground="#1e1e2e",
                activeforeground="#89b4fa",
                selectcolor="#313244",
                relief="flat",
            ).pack(side="left", padx=6)

        self._diarization_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            output_row,
            text="話者分離を有効にする",
            variable=self._diarization_var,
            font=("Meiryo", 10),
            fg="#cdd6f4",
            bg="#1e1e2e",
            activebackground="#1e1e2e",
            activeforeground="#89b4fa",
            selectcolor="#313244",
            relief="flat",
        ).pack(side="left", padx=(18, 0))

        token_row = tk.Frame(self, bg="#1e1e2e")
        token_row.pack(fill="x", padx=pad, pady=(0, 6))

        tk.Label(
            token_row,
            text="HF Token:",
            font=("Meiryo", 10),
            fg="#a6adc8",
            bg="#1e1e2e",
        ).pack(side="left")

        self._hf_token_var = tk.StringVar()
        self._hf_token_entry = tk.Entry(
            token_row,
            textvariable=self._hf_token_var,
            font=("Consolas", 9),
            fg="#cdd6f4",
            bg="#313244",
            insertbackground="white",
            relief="flat",
            bd=4,
            show="*",
        )
        self._hf_token_entry.pack(side="left", fill="x", expand=True, padx=(6, 8))

        tk.Label(
            token_row,
            text="環境変数 HF_TOKEN / HUGGINGFACE_TOKEN を優先",
            font=("Meiryo", 9),
            fg="#7f849c",
            bg="#1e1e2e",
        ).pack(side="left")

        list_frame = tk.Frame(self, bg="#1e1e2e")
        list_frame.pack(fill="both", expand=True, padx=pad, pady=(0, 6))

        columns = ("file", "status")
        self._tree = ttk.Treeview(
            list_frame,
            columns=columns,
            show="headings",
            selectmode="extended",
            height=10,
        )
        self._tree.heading("file", text="ファイル名")
        self._tree.heading("status", text="状態")
        self._tree.column("file", width=500, stretch=True)
        self._tree.column("status", width=140, anchor="center", stretch=False)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background="#313244",
            foreground="#cdd6f4",
            fieldbackground="#313244",
            rowheight=26,
            font=("Meiryo", 10),
        )
        style.configure(
            "Treeview.Heading",
            background="#45475a",
            foreground="#cdd6f4",
            font=("Meiryo", 10, "bold"),
        )
        style.map("Treeview", background=[("selected", "#585b70")])
        style.configure(
            "TProgressbar",
            troughcolor="#313244",
            background="#89b4fa",
            thickness=14,
        )

        self._tree.tag_configure("pending", foreground="#cdd6f4")
        self._tree.tag_configure("done", foreground="#a6e3a1")
        self._tree.tag_configure("error", foreground="#f38ba8")
        self._tree.tag_configure("running", foreground="#89b4fa")

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=scrollbar.set)
        self._tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        tk.Button(
            self,
            text="選択ファイルを削除",
            command=self._remove_selected,
            font=("Meiryo", 9),
            bg="#45475a",
            fg="#cdd6f4",
            activebackground="#585b70",
            relief="flat",
            cursor="hand2",
            padx=8,
        ).pack(anchor="e", padx=pad)

        self._progress_var = tk.DoubleVar(value=0)
        self._progress = ttk.Progressbar(
            self,
            variable=self._progress_var,
            maximum=100,
            length=200,
        )
        self._progress.pack(fill="x", padx=pad, pady=(4, 2))

        self._status_var = tk.StringVar(value="ファイルをドロップしてください")
        tk.Label(
            self,
            textvariable=self._status_var,
            font=("Meiryo", 9),
            fg="#a6adc8",
            bg="#1e1e2e",
            anchor="w",
        ).pack(fill="x", padx=pad)

        button_row = tk.Frame(self, bg="#1e1e2e")
        button_row.pack(pady=(6, pad))

        self._convert_btn = tk.Button(
            button_row,
            text="  処理開始  ",
            command=self._start_conversion,
            font=("Meiryo", 12, "bold"),
            bg="#89b4fa",
            fg="#1e1e2e",
            activebackground="#74c7ec",
            relief="flat",
            cursor="hand2",
            padx=16,
            pady=6,
        )
        self._convert_btn.pack(side="left", padx=8)

        self._cancel_btn = tk.Button(
            button_row,
            text="  キャンセル  ",
            command=self._cancel_conversion,
            font=("Meiryo", 12),
            bg="#f38ba8",
            fg="#1e1e2e",
            activebackground="#eba0ac",
            relief="flat",
            cursor="hand2",
            padx=16,
            pady=6,
            state="disabled",
        )
        self._cancel_btn.pack(side="left", padx=8)

    def _setup_dnd(self):
        for widget in (self, self._drop_label, self._drop_label.master):
            widget.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
            widget.dnd_bind("<<Drop>>", self._on_drop)  # type: ignore[attr-defined]

    def _on_drop(self, event):
        paths = self._parse_dnd_paths(event.data)
        added = 0
        for path in paths:
            if path.lower().endswith(".mp4") and path not in self._file_items:
                item_id = self._tree.insert(
                    "",
                    "end",
                    values=(Path(path).name, "待機中"),
                    tags=("pending",),
                )
                self._file_items[path] = item_id
                self._conversion_queue.put(path)
                added += 1

        if added:
            self._status_var.set(f"{added} ファイルを追加しました（合計 {len(self._file_items)} 件）")
        else:
            self._status_var.set("MP4 ファイルのみ対応しています")

    @staticmethod
    def _parse_dnd_paths(raw: str) -> list[str]:
        paths: list[str] = []
        raw = raw.strip()
        index = 0
        while index < len(raw):
            if raw[index] == "{":
                end = raw.index("}", index)
                paths.append(raw[index + 1:end])
                index = end + 2
                continue
            space = raw.find(" ", index)
            if space == -1:
                paths.append(raw[index:])
                break
            paths.append(raw[index:space])
            index = space + 1
        return paths

    def _choose_output(self):
        directory = filedialog.askdirectory(title="出力フォルダを選択")
        if directory:
            self._out_var.set(directory)

    def _clear_output(self):
        self._out_var.set("入力ファイルと同じフォルダ")

    def _remove_selected(self):
        if self._is_converting:
            return
        for item_id in self._tree.selection():
            path = next((src for src, iid in self._file_items.items() if iid == item_id), None)
            if path:
                del self._file_items[path]
            self._tree.delete(item_id)

    def _selected_output_formats(self) -> tuple[str, ...]:
        return tuple(
            output_format
            for output_format, var in self._format_vars.items()
            if var.get()
        )

    @staticmethod
    def _resolve_output_dir(input_path: str, output_setting: str) -> str:
        if output_setting == "入力ファイルと同じフォルダ" or not output_setting:
            return str(Path(input_path).parent)
        return output_setting

    def _build_processing_options(self) -> ProcessingOptions:
        return ProcessingOptions(
            bitrate=self._bitrate_var.get(),
            model_size=self._model_var.get(),
            language=self._language_var.get(),
            diarization_enabled=bool(self._diarization_var.get()),
            hf_token=self._hf_token_var.get(),
            output_formats=self._selected_output_formats(),
        )

    def _validate_before_start(self) -> bool:
        if not self._tree.get_children():
            messagebox.showinfo(
                "情報",
                "処理するファイルがありません。\nMP4 ファイルをドロップしてください。",
            )
            return False

        if not self._selected_output_formats():
            messagebox.showerror(
                "エラー",
                "少なくとも 1 つの出力形式を選択してください。",
            )
            return False

        if not is_ffmpeg_available():
            messagebox.showerror(
                "エラー",
                "ffmpeg が見つかりません。\nPATH に ffmpeg を追加してください。",
            )
            return False

        if not is_faster_whisper_available():
            messagebox.showerror(
                "エラー",
                "faster-whisper が見つかりません。\n`pip install -r requirements.txt` を再実行してください。",
            )
            return False

        return True

    def _start_conversion(self):
        if self._is_converting:
            return
        if not self._validate_before_start():
            return

        self._shown_messages.clear()
        self._is_converting = True
        self._cancel_flag = False
        self._processor.reset_cancel()
        self._current_options = self._build_processing_options()
        self._output_setting = self._out_var.get().strip()
        self._convert_btn.configure(state="disabled")
        self._cancel_btn.configure(state="normal")
        self._progress_var.set(0)

        while not self._conversion_queue.empty():
            try:
                self._conversion_queue.get_nowait()
            except queue.Empty:
                break

        for path, item_id in self._file_items.items():
            self._tree.item(item_id, values=(Path(path).name, "待機中"), tags=("pending",))
            self._conversion_queue.put(path)

        threading.Thread(target=self._worker, daemon=True).start()

    def _cancel_conversion(self):
        if not self._is_converting:
            return
        self._cancel_flag = True
        self._processor.cancel()
        self._status_var.set("キャンセル要求中です…")

    def _worker(self):
        total = self._conversion_queue.qsize()
        done = 0
        options = self._current_options
        output_setting = self._output_setting
        if options is None:
            self.after(0, self._on_worker_done, done, total)
            return

        while not self._conversion_queue.empty() and not self._cancel_flag:
            try:
                path = self._conversion_queue.get_nowait()
            except queue.Empty:
                break

            item_id = self._file_items.get(path)
            if not item_id:
                continue

            name = Path(path).name
            out_dir = self._resolve_output_dir(path, output_setting)
            progress_callback = self._make_progress_callback(item_id, name)
            warning_callback = self._make_warning_callback(name)

            self.after(0, self._progress_var.set, 0)
            self.after(
                0,
                lambda iid=item_id, filename=name: self._tree.item(
                    iid,
                    values=(filename, "準備中…"),
                    tags=("running",),
                ),
            )

            try:
                self._processor.process_file(
                    input_path=path,
                    output_dir=out_dir,
                    options=options,
                    progress_callback=progress_callback,
                    warning_callback=warning_callback,
                )
            except ProcessingCancelled:
                self.after(
                    0,
                    lambda iid=item_id, filename=name: self._tree.item(
                        iid,
                        values=(filename, "キャンセル"),
                        tags=("error",),
                    ),
                )
                break
            except (DependencyError, ExternalToolError, ProcessingError) as exc:
                self.after(
                    0,
                    lambda iid=item_id, filename=name: self._tree.item(
                        iid,
                        values=(filename, "エラー"),
                        tags=("error",),
                    ),
                )
                self._show_message_once(
                    method="showerror",
                    title="エラー",
                    message=f"{name}\n{exc}",
                )
            except Exception as exc:
                self.after(
                    0,
                    lambda iid=item_id, filename=name: self._tree.item(
                        iid,
                        values=(filename, "エラー"),
                        tags=("error",),
                    ),
                )
                self._show_message_once(
                    method="showerror",
                    title="エラー",
                    message=f"{name}\n予期しないエラーが発生しました: {exc}",
                )
            else:
                done += 1
                self.after(
                    0,
                    lambda iid=item_id, filename=name: self._tree.item(
                        iid,
                        values=(filename, "完了 ✓"),
                        tags=("done",),
                    ),
                )

        self.after(0, self._on_worker_done, done, total)

    def _make_progress_callback(self, item_id: str, file_name: str):
        def callback(stage: str, percent: float, detail: str):
            self.after(
                0,
                self._update_progress_ui,
                item_id,
                file_name,
                stage,
                percent,
                detail,
            )

        return callback

    def _make_warning_callback(self, file_name: str):
        def callback(message: str):
            self._show_message_once(
                method="showwarning",
                title="警告",
                message=f"{file_name}\n{message}",
                key=message,
            )
            self.after(
                0,
                self._status_var.set,
                f"{file_name} | 警告 | {message}",
            )

        return callback

    def _update_progress_ui(
        self,
        item_id: str,
        file_name: str,
        stage: str,
        percent: float,
        detail: str,
    ):
        stage_label = STAGE_LABELS.get(stage, "処理中…")
        self._tree.item(item_id, values=(file_name, stage_label), tags=("running",))
        self._progress_var.set(percent)

        if stage == "transcribing" and "/" in detail:
            status = f"{file_name} | {stage_label} | {detail}"
        elif stage == "extracting" and "経過" not in detail and "/" in detail:
            status = f"{file_name} | {stage_label} | {detail}"
        else:
            status = f"{file_name} | {stage_label}"
            if detail:
                status += f" | {detail}"
        self._status_var.set(status)

    def _show_message_once(
        self,
        method: str,
        title: str,
        message: str,
        key: str | None = None,
    ):
        unique_key = key or f"{method}:{title}:{message}"
        if unique_key in self._shown_messages:
            return
        self._shown_messages.add(unique_key)
        self.after(0, lambda: getattr(messagebox, method)(title, message))

    def _on_worker_done(self, done: int, total: int):
        self._is_converting = False
        self._convert_btn.configure(state="normal")
        self._cancel_btn.configure(state="disabled")

        if self._cancel_flag:
            self._progress_var.set(0)
            self._status_var.set("キャンセルしました")
            return

        self._progress_var.set(100 if total else 0)
        self._status_var.set(f"完了: {done}/{total} ファイルを処理しました")


def main():
    app = MP4toTextConverter()
    app.mainloop()


if __name__ == "__main__":
    main()
