from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .calibration import build_roi_profile, roi_rect_from_profile
from .image_io import read_image
from .models import CropRecord, ImageSample, ProcessingResult, ProjectState, Rect
from .parser import scan_images
from .processor import (
    build_project_state,
    clip_rect,
    detect_crops_for_samples,
    export_results,
    process_rgb_results,
    refresh_single_crop,
)
from .project_store import load_project_state, save_project_state


def format_triplet(values: tuple[float | None, float | None, float | None], labels: tuple[str, str, str]) -> str:
    if any(value is None for value in values):
        return ""
    return " ".join(f"{label}{value:.1f}" for label, value in zip(labels, values))


class ImageRectCanvas(ttk.Frame):
    def __init__(self, master, editable: bool = False, on_rect_changed=None) -> None:
        super().__init__(master)
        self.on_rect_changed = on_rect_changed
        self.editable = editable
        self.canvas = tk.Canvas(self, bg="#171717", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.redraw())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        self.image_bgr = None
        self.photo = None
        self.scale = 1.0
        self.origin = (0, 0)
        self.display_size = (0, 0)
        self.drag_start = None
        self.editable_rect: Rect | None = None
        self.overlays: list[tuple[Rect, str, int]] = []

    def set_editable(self, editable: bool) -> None:
        self.editable = editable

    def set_image(self, image_bgr) -> None:
        self.image_bgr = image_bgr
        self.redraw()

    def set_rect(self, rect: Rect | None) -> None:
        self.editable_rect = rect
        self.redraw()

    def get_rect(self) -> Rect | None:
        return self.editable_rect

    def set_overlays(self, overlays: list[tuple[Rect, str, int]]) -> None:
        self.overlays = overlays
        self.redraw()

    def clear(self) -> None:
        self.image_bgr = None
        self.photo = None
        self.editable_rect = None
        self.overlays = []
        self.canvas.delete("all")

    def redraw(self) -> None:
        if self.image_bgr is None:
            self.canvas.delete("all")
            return

        canvas_w = max(self.canvas.winfo_width(), 100)
        canvas_h = max(self.canvas.winfo_height(), 100)
        image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        width, height = image.size
        scale = min(canvas_w / width, canvas_h / height, 1.0)
        display_w = max(1, int(width * scale))
        display_h = max(1, int(height * scale))
        resized = image.resize((display_w, display_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        origin_x = (canvas_w - display_w) // 2
        origin_y = (canvas_h - display_h) // 2
        self.scale = scale
        self.origin = (origin_x, origin_y)
        self.display_size = (display_w, display_h)

        self.canvas.delete("all")
        self.canvas.create_image(origin_x, origin_y, image=self.photo, anchor="nw")

        for rect, color, width_px in self.overlays:
            self._draw_rect(rect, color, width_px)
        if self.editable_rect is not None:
            self._draw_rect(self.editable_rect, "#f39c12", 3)

    def _draw_rect(self, rect: Rect, color: str, width_px: int) -> None:
        origin_x, origin_y = self.origin
        scale = self.scale
        self.canvas.create_rectangle(
            origin_x + rect.x * scale,
            origin_y + rect.y * scale,
            origin_x + rect.right * scale,
            origin_y + rect.bottom * scale,
            outline=color,
            width=width_px,
        )

    def _on_press(self, event) -> None:
        if not self.editable or self.image_bgr is None:
            return
        self.drag_start = (event.x, event.y)

    def _on_drag(self, event) -> None:
        if not self.editable or self.drag_start is None:
            return
        rect = self._canvas_to_rect(self.drag_start[0], self.drag_start[1], event.x, event.y)
        if rect is not None:
            self.editable_rect = rect
            self.redraw()

    def _on_release(self, event) -> None:
        if not self.editable or self.drag_start is None:
            return
        rect = self._canvas_to_rect(self.drag_start[0], self.drag_start[1], event.x, event.y)
        self.drag_start = None
        if rect is None:
            return
        self.editable_rect = rect
        self.redraw()
        if self.on_rect_changed is not None:
            self.on_rect_changed(rect)

    def _canvas_to_rect(self, x1: int, y1: int, x2: int, y2: int) -> Rect | None:
        origin_x, origin_y = self.origin
        display_w, display_h = self.display_size
        if self.scale <= 0:
            return None
        left = max(origin_x, min(x1, x2))
        top = max(origin_y, min(y1, y2))
        right = min(origin_x + display_w, max(x1, x2))
        bottom = min(origin_y + display_h, max(y1, y2))
        if right - left < 8 or bottom - top < 8:
            return None
        image_left = int((left - origin_x) / self.scale)
        image_top = int((top - origin_y) / self.scale)
        image_right = int((right - origin_x) / self.scale)
        image_bottom = int((bottom - origin_y) / self.scale)
        return Rect(image_left, image_top, max(1, image_right - image_left), max(1, image_bottom - image_top))


class DistinguishTCApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Distinguish_TC 两阶段裁剪与取色工具")
        self.root.geometry("1780x1020")

        self.state: ProjectState | None = None
        self.results: list[ProcessingResult] = []
        self.crop_tab_samples: list[ImageSample] = []
        self.review_tab_records: list[CropRecord] = []
        self.roi_tab_records: list[CropRecord] = []
        self.current_review_record: CropRecord | None = None
        self.current_roi_record: CropRecord | None = None

        self.source_dir_var = tk.StringVar(value=str(Path.cwd() / "Photo"))
        self.project_dir_var = tk.StringVar(value=str(Path.cwd() / "project_data"))
        self.roi_profile_name_var = tk.StringVar(value="default_roi")
        self.trim_percent_var = tk.DoubleVar(value=0.05)
        self.summary_var = tk.StringVar(value="尚未创建项目")
        self.status_var = tk.StringVar(value="请选择原图目录并创建项目")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.review_reason_var = tk.StringVar(value="")

        self._build_ui()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        project_bar = ttk.Frame(self.root, padding=10)
        project_bar.grid(row=0, column=0, sticky="ew")
        project_bar.columnconfigure(1, weight=1)
        project_bar.columnconfigure(4, weight=1)

        ttk.Label(project_bar, text="原图目录").grid(row=0, column=0, sticky="w")
        ttk.Entry(project_bar, textvariable=self.source_dir_var).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(project_bar, text="选择", command=self.choose_source_dir).grid(row=0, column=2, padx=4)
        ttk.Button(project_bar, text="创建项目", command=self.create_project).grid(row=0, column=3, padx=4)

        ttk.Label(project_bar, text="项目目录").grid(row=0, column=4, sticky="w", padx=(16, 0))
        ttk.Entry(project_bar, textvariable=self.project_dir_var).grid(row=0, column=5, sticky="ew", padx=4)
        ttk.Button(project_bar, text="选择", command=self.choose_project_dir).grid(row=0, column=6, padx=4)
        ttk.Button(project_bar, text="打开项目", command=self.open_project).grid(row=0, column=7, padx=4)
        ttk.Button(project_bar, text="保存项目", command=self.save_project).grid(row=0, column=8, padx=4)

        ttk.Label(project_bar, textvariable=self.summary_var, wraplength=720).grid(row=1, column=0, columnspan=9, sticky="w", pady=(8, 0))
        ttk.Progressbar(project_bar, variable=self.progress_var, maximum=100).grid(row=2, column=0, columnspan=9, sticky="ew", pady=(8, 0))
        ttk.Label(project_bar, textvariable=self.status_var, wraplength=900).grid(row=3, column=0, columnspan=9, sticky="w")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, sticky="nsew")

        self.crop_tab = ttk.Frame(self.notebook, padding=10)
        self.review_tab = ttk.Frame(self.notebook, padding=10)
        self.roi_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.crop_tab, text="阶段1 原图裁剪")
        self.notebook.add(self.review_tab, text="阶段2 异常修正")
        self.notebook.add(self.roi_tab, text="阶段3 ROI取色")

        self._build_crop_tab()
        self._build_review_tab()
        self._build_roi_tab()

    def _build_crop_tab(self) -> None:
        self.crop_tab.columnconfigure(1, weight=1)
        self.crop_tab.columnconfigure(2, weight=1)
        self.crop_tab.rowconfigure(0, weight=1)

        left = ttk.Frame(self.crop_tab)
        left.grid(row=0, column=0, sticky="nsw")
        left.rowconfigure(2, weight=1)

        ttk.Label(left, text="阶段1：先批量自动裁出全部比色皿，再只修异常图").grid(row=0, column=0, sticky="w")
        ttk.Button(left, text="开始自动裁剪", command=self.start_auto_crop).grid(row=1, column=0, sticky="ew", pady=(8, 8))
        self.crop_listbox = tk.Listbox(left, width=42, height=36)
        self.crop_listbox.grid(row=2, column=0, sticky="nsw")
        self.crop_listbox.bind("<<ListboxSelect>>", lambda _event: self.show_crop_tab_selection())
        ttk.Button(left, text="切换当前图手动异常标记", command=self.toggle_manual_flag_from_crop_tab).grid(row=3, column=0, sticky="ew", pady=(8, 0))

        self.crop_original_canvas = ImageRectCanvas(self.crop_tab, editable=False)
        self.crop_original_canvas.grid(row=0, column=1, sticky="nsew", padx=(10, 10))

        self.crop_preview_canvas = ImageRectCanvas(self.crop_tab, editable=False)
        self.crop_preview_canvas.grid(row=0, column=2, sticky="nsew")

    def _build_review_tab(self) -> None:
        self.review_tab.columnconfigure(1, weight=1)
        self.review_tab.columnconfigure(2, weight=1)
        self.review_tab.rowconfigure(0, weight=1)

        left = ttk.Frame(self.review_tab)
        left.grid(row=0, column=0, sticky="nsw")
        left.rowconfigure(1, weight=1)

        ttk.Label(left, text="异常待修正列表").grid(row=0, column=0, sticky="w")
        self.review_listbox = tk.Listbox(left, width=42, height=30)
        self.review_listbox.grid(row=1, column=0, sticky="nsw", pady=(8, 0))
        self.review_listbox.bind("<<ListboxSelect>>", lambda _event: self.show_review_selection())

        ttk.Label(left, textvariable=self.review_reason_var, wraplength=320).grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Button(left, text="保存当前裁剪框", command=self.save_review_rect).grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(left, text="保存并标记已修正", command=self.mark_review_fixed).grid(row=4, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(left, text="保持自动结果并忽略", command=self.mark_review_ignored).grid(row=5, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(left, text="切换手动异常标记", command=self.toggle_manual_flag_from_review_tab).grid(row=6, column=0, sticky="ew", pady=(6, 0))

        self.review_original_canvas = ImageRectCanvas(self.review_tab, editable=True, on_rect_changed=self.on_review_rect_changed)
        self.review_original_canvas.grid(row=0, column=1, sticky="nsew", padx=(10, 10))

        self.review_crop_preview_canvas = ImageRectCanvas(self.review_tab, editable=False)
        self.review_crop_preview_canvas.grid(row=0, column=2, sticky="nsew")

    def _build_roi_tab(self) -> None:
        self.roi_tab.columnconfigure(1, weight=1)
        self.roi_tab.columnconfigure(2, weight=1)
        self.roi_tab.rowconfigure(0, weight=1)
        self.roi_tab.rowconfigure(1, weight=1)

        left = ttk.Frame(self.roi_tab)
        left.grid(row=0, column=0, rowspan=2, sticky="nsw")
        left.rowconfigure(4, weight=1)

        ttk.Label(left, text="ROI 配置名").grid(row=0, column=0, sticky="w")
        ttk.Entry(left, textvariable=self.roi_profile_name_var, width=30).grid(row=1, column=0, sticky="ew", pady=(4, 8))
        ttk.Label(left, text="裁剪后再标 ROI，后续全部颜色特征都基于裁剪图计算").grid(row=2, column=0, sticky="w")
        ttk.Label(left, text="亮度裁剪比例").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(left, variable=self.trim_percent_var, from_=0.0, to=0.2, orient="horizontal").grid(row=4, column=0, sticky="ew")
        ttk.Label(left, textvariable=self.trim_percent_var).grid(row=5, column=0, sticky="w")

        ttk.Label(left, text="裁剪图列表").grid(row=6, column=0, sticky="w", pady=(10, 0))
        self.roi_listbox = tk.Listbox(left, width=42, height=18)
        self.roi_listbox.grid(row=7, column=0, sticky="nsw", pady=(8, 0))
        self.roi_listbox.bind("<<ListboxSelect>>", lambda _event: self.show_roi_selection())
        ttk.Button(left, text="保存当前 ROI 配置", command=self.save_roi_profile).grid(row=8, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(left, text="批量提取颜色特征", command=self.start_rgb_processing).grid(row=9, column=0, sticky="ew", pady=(6, 0))

        self.roi_canvas = ImageRectCanvas(self.roi_tab, editable=True, on_rect_changed=self.on_roi_rect_changed)
        self.roi_canvas.grid(row=0, column=1, sticky="nsew", padx=(10, 10))

        self.result_preview_canvas = ImageRectCanvas(self.roi_tab, editable=False)
        self.result_preview_canvas.grid(row=0, column=2, sticky="nsew")

        result_frame = ttk.Frame(self.roi_tab)
        result_frame.grid(row=1, column=1, columnspan=2, sticky="nsew", pady=(10, 0))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

        columns = ("sample_name", "replicate_id", "status", "rgb_mean", "hsv_mean", "warning")
        self.result_tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=14)
        for column in columns:
            self.result_tree.heading(column, text=column)
            self.result_tree.column(column, width=130, anchor="center")
        self.result_tree.column("sample_name", width=230, anchor="w")
        self.result_tree.column("warning", width=260, anchor="w")
        self.result_tree.grid(row=0, column=0, sticky="nsew")
        self.result_tree.bind("<<TreeviewSelect>>", lambda _event: self.show_result_preview())

        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.result_tree.configure(yscrollcommand=scrollbar.set)

    def choose_source_dir(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.source_dir_var.get() or str(Path.cwd()))
        if selected:
            self.source_dir_var.set(selected)

    def choose_project_dir(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.project_dir_var.get() or str(Path.cwd()))
        if selected:
            self.project_dir_var.set(selected)

    def project_file_path(self) -> Path:
        return Path(self.project_dir_var.get()) / "project.json"

    def create_project(self) -> None:
        try:
            source_dir = Path(self.source_dir_var.get())
            project_dir = Path(self.project_dir_var.get())
            samples = scan_images(source_dir)
            state = build_project_state(source_dir, project_dir, samples)
            self.state = state
            save_project_state(state, self.project_file_path())
            self.results = []
            self.status_var.set("项目已创建，请进入阶段1进行自动裁剪")
            self.refresh_all_views()
        except Exception as exc:
            messagebox.showerror("创建项目失败", str(exc))

    def open_project(self) -> None:
        selected = filedialog.askopenfilename(
            initialdir=self.project_dir_var.get() or str(Path.cwd()),
            filetypes=[("项目文件", "*.json")],
        )
        if not selected:
            return
        try:
            state = load_project_state(Path(selected))
            self.state = state
            self.source_dir_var.set(state.source_dir)
            self.project_dir_var.set(state.project_dir)
            if state.roi_profile is not None:
                self.roi_profile_name_var.set(state.roi_profile.profile_name)
                self.trim_percent_var.set(state.roi_profile.trim_percent)
            self.results = []
            self.status_var.set("项目已打开")
            self.refresh_all_views()
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc))

    def save_project(self) -> None:
        if self.state is None:
            messagebox.showwarning("无法保存", "当前还没有可保存的项目，请先创建或打开一个项目。")
            return
        try:
            self.state.source_dir = self.source_dir_var.get()
            self.state.project_dir = self.project_dir_var.get()
            project_file = self.project_file_path()
            save_project_state(self.state, project_file)
            self.status_var.set(f"项目已保存到 {project_file}")
            messagebox.showinfo("保存成功", f"项目已保存成功。\n\n保存路径：\n{project_file}")
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))

    def refresh_all_views(self) -> None:
        self.refresh_summary()
        self.refresh_crop_list()
        self.refresh_review_list()
        self.refresh_roi_list()
        self.refresh_result_table()

    def refresh_summary(self) -> None:
        if self.state is None:
            self.summary_var.set("尚未创建项目")
            return
        total = len(self.state.samples)
        crop_count = len([record for record in self.state.crops if record.final_rect is not None])
        flagged = len(self.flagged_records())
        roi_name = self.state.roi_profile.profile_name if self.state.roi_profile else "未设置"
        self.summary_var.set(
            f"样本 {total} 张；已保存裁剪图 {crop_count} 张；待检查异常 {flagged} 张；当前 ROI 配置: {roi_name}"
        )

    def sample_map(self) -> dict[str, ImageSample]:
        if self.state is None:
            return {}
        return {sample.relative_path.as_posix(): sample for sample in self.state.samples}

    def crop_record_map(self) -> dict[str, CropRecord]:
        if self.state is None:
            return {}
        return {record.image_relative_path: record for record in self.state.crops}

    def flagged_records(self) -> list[CropRecord]:
        if self.state is None:
            return []
        return [
            record
            for record in self.state.crops
            if record.review_status == "pending" or record.auto_flagged or record.manual_flagged
        ]

    def refresh_crop_list(self) -> None:
        self.crop_listbox.delete(0, tk.END)
        self.crop_tab_samples = []
        if self.state is None:
            return
        crop_map = self.crop_record_map()
        for sample in self.state.samples:
            record = crop_map.get(sample.relative_path.as_posix())
            prefix = "未裁剪"
            if record is not None:
                prefix = "异常" if (record.auto_flagged or record.manual_flagged or record.review_status == "pending") else "正常"
            self.crop_tab_samples.append(sample)
            self.crop_listbox.insert(tk.END, f"[{prefix}] {sample.relative_path.as_posix()}")
        if self.crop_tab_samples:
            self.crop_listbox.selection_set(0)
            self.show_crop_tab_selection()

    def refresh_review_list(self) -> None:
        self.review_listbox.delete(0, tk.END)
        self.review_tab_records = self.flagged_records()
        for record in self.review_tab_records:
            self.review_listbox.insert(tk.END, record.image_relative_path)
        if self.review_tab_records:
            self.review_listbox.selection_set(0)
            self.show_review_selection()
        else:
            self.review_reason_var.set("当前没有待修正异常图")
            self.review_original_canvas.clear()
            self.review_crop_preview_canvas.clear()

    def refresh_roi_list(self) -> None:
        self.roi_listbox.delete(0, tk.END)
        self.roi_tab_records = []
        if self.state is None:
            return
        for record in self.state.crops:
            if record.final_rect is None:
                continue
            crop_path = Path(self.state.project_dir) / record.crop_relative_path
            if crop_path.exists():
                self.roi_tab_records.append(record)
                self.roi_listbox.insert(tk.END, record.image_relative_path)
        if self.roi_tab_records:
            self.roi_listbox.selection_set(0)
            self.show_roi_selection()

    def show_crop_tab_selection(self) -> None:
        if self.state is None or not self.crop_tab_samples:
            return
        selection = self.crop_listbox.curselection()
        if not selection:
            return
        sample = self.crop_tab_samples[selection[0]]
        image_bgr = read_image(sample.image_path)
        self.crop_original_canvas.set_image(image_bgr)
        record = self.crop_record_map().get(sample.relative_path.as_posix())
        overlays = []
        if record is not None and record.auto_rect is not None:
            overlays.append((record.auto_rect, "#2ecc71", 2))
        if record is not None and record.final_rect is not None:
            overlays.append((record.final_rect, "#f39c12", 3))
        self.crop_original_canvas.set_overlays(overlays)

        if record is not None:
            crop_path = Path(self.state.project_dir) / record.crop_relative_path
            if crop_path.exists():
                self.crop_preview_canvas.set_image(read_image(crop_path))
                self.crop_preview_canvas.set_overlays([])
                self.crop_preview_canvas.set_rect(None)
                return
        self.crop_preview_canvas.clear()

    def toggle_manual_flag_from_crop_tab(self) -> None:
        if self.state is None or not self.crop_tab_samples:
            return
        selection = self.crop_listbox.curselection()
        if not selection:
            return
        sample = self.crop_tab_samples[selection[0]]
        record = self.crop_record_map().get(sample.relative_path.as_posix())
        if record is None:
            messagebox.showwarning("尚未裁剪", "请先完成阶段1自动裁剪")
            return
        self.toggle_manual_flag(record)

    def start_auto_crop(self) -> None:
        if self.state is None:
            messagebox.showerror("无法开始", "请先创建或打开一个项目")
            return

        self.progress_var.set(0)
        self.status_var.set("正在批量自动裁剪比色皿...")

        def worker() -> None:
            def progress(index: int, total: int, sample: ImageSample, record: CropRecord) -> None:
                percent = index * 100 / total
                self.root.after(
                    0,
                    lambda: (
                        self.progress_var.set(percent),
                        self.status_var.set(f"自动裁剪 {index}/{total}: {sample.relative_path.as_posix()}"),
                    ),
                )

            crops = detect_crops_for_samples(self.state.samples, Path(self.state.project_dir), progress_callback=progress)
            self.state.crops = crops
            save_project_state(self.state, self.project_file_path())
            self.root.after(0, self._finish_auto_crop)

        threading.Thread(target=worker, daemon=True).start()

    def _finish_auto_crop(self) -> None:
        self.progress_var.set(100)
        self.status_var.set("阶段1完成，请进入阶段2检查异常裁剪，再到阶段3标注 ROI")
        self.refresh_all_views()

    def show_review_selection(self) -> None:
        if self.state is None or not self.review_tab_records:
            return
        selection = self.review_listbox.curselection()
        if not selection:
            return
        record = self.review_tab_records[selection[0]]
        sample = self.sample_map()[record.image_relative_path]
        self.current_review_record = record
        image_bgr = read_image(sample.image_path)
        self.review_original_canvas.set_image(image_bgr)
        self.review_original_canvas.set_overlays(
            [(record.auto_rect, "#2ecc71", 2)] if record.auto_rect is not None else []
        )
        self.review_original_canvas.set_rect(record.final_rect or record.auto_rect)
        self.review_reason_var.set(f"异常原因: {record.flag_reason or '手动加入异常列表'}")

        crop_path = Path(self.state.project_dir) / record.crop_relative_path
        if crop_path.exists():
            self.review_crop_preview_canvas.set_image(read_image(crop_path))
            self.review_crop_preview_canvas.set_rect(None)
            self.review_crop_preview_canvas.set_overlays([])
        else:
            self.review_crop_preview_canvas.clear()

    def on_review_rect_changed(self, _rect: Rect) -> None:
        pass

    def save_review_rect(self) -> None:
        if self.state is None or self.current_review_record is None:
            return
        rect = self.review_original_canvas.get_rect()
        if rect is None:
            messagebox.showerror("保存失败", "请先在原图上框定一个有效比色皿区域")
            return
        sample = self.sample_map()[self.current_review_record.image_relative_path]
        self.current_review_record.final_rect = clip_rect(rect, read_image(sample.image_path).shape[1], read_image(sample.image_path).shape[0])
        refresh_single_crop(sample, self.current_review_record, Path(self.state.project_dir))
        save_project_state(self.state, self.project_file_path())
        self.status_var.set(f"已保存裁剪框: {sample.relative_path.as_posix()}")
        self.show_review_selection()
        self.refresh_crop_list()
        self.refresh_roi_list()

    def mark_review_fixed(self) -> None:
        if self.current_review_record is None:
            return
        self.save_review_rect()
        self.current_review_record.auto_flagged = False
        self.current_review_record.manual_flagged = False
        self.current_review_record.review_status = "fixed"
        self.current_review_record.flag_reason = "已人工修正"
        save_project_state(self.state, self.project_file_path())
        self.refresh_all_views()

    def mark_review_ignored(self) -> None:
        if self.current_review_record is None:
            return
        if self.current_review_record.auto_rect is not None and self.current_review_record.final_rect is None:
            self.current_review_record.final_rect = self.current_review_record.auto_rect
        self.current_review_record.auto_flagged = False
        self.current_review_record.manual_flagged = False
        self.current_review_record.review_status = "ignored"
        self.current_review_record.flag_reason = "已人工忽略"
        save_project_state(self.state, self.project_file_path())
        self.refresh_all_views()

    def toggle_manual_flag_from_review_tab(self) -> None:
        if self.current_review_record is None:
            return
        self.toggle_manual_flag(self.current_review_record)

    def toggle_manual_flag(self, record: CropRecord) -> None:
        record.manual_flagged = not record.manual_flagged
        if record.manual_flagged:
            record.review_status = "pending"
            if "手动加入异常列表" not in record.flag_reason:
                record.flag_reason = f"{record.flag_reason}；手动加入异常列表".strip("；")
        elif not record.auto_flagged:
            record.review_status = "auto_ok"
            if record.flag_reason == "手动加入异常列表":
                record.flag_reason = ""
        save_project_state(self.state, self.project_file_path())
        self.refresh_all_views()

    def show_roi_selection(self) -> None:
        if self.state is None or not self.roi_tab_records:
            return
        selection = self.roi_listbox.curselection()
        if not selection:
            return
        record = self.roi_tab_records[selection[0]]
        crop_path = Path(self.state.project_dir) / record.crop_relative_path
        crop_bgr = read_image(crop_path)
        self.current_roi_record = record
        self.roi_canvas.set_image(crop_bgr)
        self.roi_canvas.set_overlays([])

        if self.state.roi_profile is not None:
            roi_rect = roi_rect_from_profile(self.state.roi_profile, crop_bgr.shape[1], crop_bgr.shape[0])
        else:
            roi_rect = Rect(
                int(crop_bgr.shape[1] * 0.20),
                int(crop_bgr.shape[0] * 0.45),
                int(crop_bgr.shape[1] * 0.60),
                int(crop_bgr.shape[0] * 0.33),
            )
        self.roi_canvas.set_rect(clip_rect(roi_rect, crop_bgr.shape[1], crop_bgr.shape[0]))

    def on_roi_rect_changed(self, _rect: Rect) -> None:
        pass

    def save_roi_profile(self) -> None:
        if self.state is None or self.current_roi_record is None:
            messagebox.showerror("无法保存", "请先在裁剪图列表中选一张图并框定 ROI")
            return
        rect = self.roi_canvas.get_rect()
        if rect is None:
            messagebox.showerror("无法保存", "请先在裁剪图上框定 ROI")
            return
        crop_path = Path(self.state.project_dir) / self.current_roi_record.crop_relative_path
        crop_bgr = read_image(crop_path)
        profile = build_roi_profile(
            profile_name=self.roi_profile_name_var.get().strip() or "default_roi",
            source_crop_path=self.current_roi_record.crop_relative_path,
            roi_rect=rect,
            crop_width=crop_bgr.shape[1],
            crop_height=crop_bgr.shape[0],
            trim_percent=self.trim_percent_var.get(),
        )
        self.state.roi_profile = profile
        save_project_state(self.state, self.project_file_path())
        self.status_var.set("ROI 配置已保存，后续颜色特征提取将基于裁剪图使用该配置")
        self.refresh_summary()

    def start_rgb_processing(self) -> None:
        if self.state is None:
            messagebox.showerror("无法开始", "请先创建或打开一个项目")
            return
        if self.state.roi_profile is None:
            messagebox.showerror("无法开始", "请先保存一个 ROI 配置")
            return

        self.progress_var.set(0)
        self.status_var.set("正在基于裁剪图批量提取颜色特征...")

        def worker() -> None:
            def progress(index: int, total: int, sample: ImageSample, result: ProcessingResult) -> None:
                percent = index * 100 / total
                self.root.after(
                    0,
                    lambda: (
                        self.progress_var.set(percent),
                        self.status_var.set(f"提取特征 {index}/{total}: {sample.relative_path.as_posix()} | {result.status}"),
                    ),
                )

            results = process_rgb_results(self.state, progress_callback=progress)
            csv_path = export_results(results, Path(self.state.project_dir))

            def finish() -> None:
                self.results = results
                self.progress_var.set(100)
                self.status_var.set(f"颜色特征提取完成，结果已保存到 {csv_path}")
                self.refresh_result_table()
                self.show_roi_selection()

            self.root.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def refresh_result_table(self) -> None:
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        for index, result in enumerate(self.results):
            values = (
                result.sample_name,
                result.replicate_id,
                result.status,
                format_triplet((result.roi_mean_r, result.roi_mean_g, result.roi_mean_b), ("R", "G", "B")),
                format_triplet((result.hsv_mean_h, result.hsv_mean_s, result.hsv_mean_v), ("H", "S", "V")),
                result.warning or result.error_message,
            )
            self.result_tree.insert("", "end", iid=str(index), values=values)

    def show_result_preview(self) -> None:
        if self.state is None:
            return
        selection = self.result_tree.selection()
        if not selection:
            return
        result = self.results[int(selection[0])]
        if result.preview_path:
            preview_path = Path(self.state.project_dir) / result.preview_path
            if preview_path.exists():
                self.result_preview_canvas.set_image(read_image(preview_path))
                self.result_preview_canvas.set_rect(None)
                self.result_preview_canvas.set_overlays([])
                return
        crop_path = Path(self.state.project_dir) / result.crop_path
        if crop_path.exists():
            self.result_preview_canvas.set_image(read_image(crop_path))
            self.result_preview_canvas.set_rect(None)
            self.result_preview_canvas.set_overlays([])


def run() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    DistinguishTCApp(root)
    root.mainloop()
