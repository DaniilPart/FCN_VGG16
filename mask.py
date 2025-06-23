#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
interactive_overlay.py
Позволяет:
 • выбрать папки с фонами, масками и готовыми сегментациями;
 • щёлчком мыши разместить маску на фоне;
 • колёсиком мыши масштабировать маску;
 • при сохранении закрашивать область под маской красным на соответствующей
   сегментации и сохранять оба файла.
 • Корректно находит сегментацию по шаблону 'prefix_lane_suffix.png'.

Зависимости: Pillow (pip install pillow)
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from typing import List, Optional

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


class OverlayApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Интерактивное наложение масок и редактирование сегментации")
        self.geometry("1200x800")

        # --- Пути к данным ---
        self.bg_files: List[str] = []
        self.ov_files: List[str] = []
        self.seg_dir: Optional[str] = None
        self.current_seg_path: Optional[str] = None
        self.bg_idx = self.ov_idx = 0

        # --- Параметры текущего оверлея ---
        self.scale = 1.0
        self.ov_pos = (0, 0)
        self.bg_img: Optional[Image.Image] = None
        self.ov_img: Optional[Image.Image] = None
        self.ov_img_orig: Optional[Image.Image] = None

        # --- UI (Пользовательский интерфейс) ---
        bar = tk.Frame(self);
        bar.pack(fill="x", padx=4, pady=4)

        tk.Button(bar, text="Папка с фонами…", command=self.choose_bg_dir).pack(side="left")
        tk.Button(bar, text="Папка с масками…", command=self.choose_ov_dir).pack(side="left")
        tk.Button(bar, text="Папка с сегментациями…", command=self.choose_seg_dir).pack(side="left")

        tk.Button(bar, text="← фон", command=lambda: self.shift_bg(-1)).pack(side="left", padx=(10, 0))
        tk.Button(bar, text="фон →", command=lambda: self.shift_bg(1)).pack(side="left")
        tk.Button(bar, text="← маска", command=lambda: self.shift_ov(-1)).pack(side="left", padx=(10, 0))
        tk.Button(bar, text="маска →", command=lambda: self.shift_ov(1)).pack(side="left")

        self.save_button = tk.Button(bar, text="Сохранить результат и сегментацию", command=self.save_all)
        self.save_button.pack(side="right", padx=10)

        self.canvas = tk.Canvas(self, bg="grey")
        self.canvas.pack(fill="both", expand=True)

        # --- Привязка событий мыши ---
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", self.on_wheel)
        self.canvas.bind("<Button-5>", self.on_wheel)

    # ------------- Выбор директорий ----------------------------------------
    def choose_bg_dir(self):
        path = filedialog.askdirectory(title="Выберите папку с фонами")
        if path:
            self.bg_files = self.scan_dir(path)
            self.bg_idx = 0
            self.update_view()

    def choose_ov_dir(self):
        path = filedialog.askdirectory(title="Выберите папку с масками (прозрачный фон)")
        if path:
            self.ov_files = self.scan_dir(path)
            self.ov_idx = 0
            self.update_view()

    def choose_seg_dir(self):
        path = filedialog.askdirectory(title="Выберите папку с файлами сегментации")
        if path:
            self.seg_dir = path
            self.update_view()

    @staticmethod
    def scan_dir(folder: str) -> List[str]:
        return sorted([os.path.join(folder, f) for f in os.listdir(folder)
                       if f.lower().endswith(IMG_EXT)])

    # ------------- Логика поиска сегментации -------------------------------
    def find_segmentation_for_current_background(self) -> Optional[str]:
        """
        Ищет файл сегментации, преобразуя имя фона по шаблону.
        Пример: 'um_000000.png' -> 'um_lane_000000.png'
        """
        if not self.seg_dir or not self.bg_files:
            return None

        bg_path = self.bg_files[self.bg_idx]
        base_name = os.path.basename(bg_path)  # e.g., "um_000000.png"

        try:
            # Разделяем имя файла по первому подчеркиванию
            prefix, suffix = base_name.split('_', 1)  # -> ('um', '000000.png')

            # Собираем новое имя файла сегментации
            seg_name = f"{prefix}_lane_{suffix}"  # -> "um_lane_000000.png"

            # Формируем полный путь и проверяем его существование
            seg_path = os.path.join(self.seg_dir, seg_name)

            if os.path.exists(seg_path):
                return seg_path
            else:
                return None
        except ValueError:
            # Сработает, если в имени файла нет символа '_'
            # Выводим в консоль, чтобы не мешать пользователю всплывающими окнами
            print(f"Предупреждение: Имя файла фона '{base_name}' не соответствует шаблону 'prefix_suffix.png'")
            return None

    # ------------- Переключение изображений --------------------------------
    def shift_bg(self, step: int):
        if self.bg_files:
            self.bg_idx = (self.bg_idx + step) % len(self.bg_files)
            self.update_view()

    def shift_ov(self, step: int):
        if self.ov_files:
            self.ov_idx = (self.ov_idx + step) % len(self.ov_files)
            self.update_view()

    # ------------- Обработчики мыши ----------------------------------------
    def on_click(self, ev):
        """ЛКМ — задать позицию центра маски."""
        self.ov_pos = (ev.x, ev.y)
        self.redraw_overlay()

    def on_wheel(self, ev):
        """Колёсико мыши — масштабирование."""
        direction = +1 if (ev.delta > 0 or ev.num == 4) else -1
        self.scale *= 1.1 if direction > 0 else 0.9
        self.scale = max(0.1, min(self.scale, 5.0))
        self.redraw_overlay()

    # ------------- Отрисовка на холсте -------------------------------------
    def update_view(self):
        """Полностью обновляет холст при смене фона, маски или папки."""
        if not (self.bg_files and self.ov_files):
            return

        # Загрузка фона
        self.bg_img = Image.open(self.bg_files[self.bg_idx]).convert("RGBA")
        self.tk_bg = ImageTk.PhotoImage(self.bg_img)

        # Загрузка маски
        self.ov_img_orig = Image.open(self.ov_files[self.ov_idx]).convert("RGBA")
        self.scale = 1.0
        self.ov_pos = (self.bg_img.width // 2, self.bg_img.height // 2)

        # Поиск соответствующей сегментации
        self.current_seg_path = self.find_segmentation_for_current_background()
        if self.current_seg_path:
            self.title(f"Редактор - Сегментация найдена: {os.path.basename(self.current_seg_path)}")
        else:
            self.title(
                f"Редактор - ВНИМАНИЕ: Сегментация для этого фона ({os.path.basename(self.bg_files[self.bg_idx])}) НЕ НАЙДЕНА")

        # Очистка и настройка холста
        self.canvas.delete("all")
        self.canvas.config(width=self.tk_bg.width(), height=self.tk_bg.height())
        self.canvas.create_image(0, 0, image=self.tk_bg, anchor="nw")
        self.redraw_overlay()

    def redraw_overlay(self):
        """Перерисовывает только маску при её перемещении или масштабировании."""
        if not self.ov_img_orig:
            return

        w, h = self.ov_img_orig.size
        new_size = (int(w * self.scale), int(h * self.scale))
        self.ov_img = self.ov_img_orig.resize(new_size, Image.LANCZOS)
        self.tk_ov = ImageTk.PhotoImage(self.ov_img)

        # Перемещаем или создаем объект маски на холсте
        if self.canvas.find_withtag("ov"):
            self.canvas.itemconfigure("ov", image=self.tk_ov)
            self.canvas.coords("ov", self.ov_pos)
        else:
            self.canvas.create_image(*self.ov_pos, image=self.tk_ov, anchor="center", tags="ov")

    # ------------- Сохранение результатов -----------------------------------
    def save_all(self):
        """Сохраняет итоговое изображение и изменяет файл сегментации."""
        if not (self.bg_img and self.ov_img):
            messagebox.showwarning("Нет данных", "Не выбран фон или маска.")
            return

        # --- 1. Сохранение визуального результата (фон + маска) ---
        visual_result = self.bg_img.copy()
        top_left_x = self.ov_pos[0] - self.ov_img.width // 2
        top_left_y = self.ov_pos[1] - self.ov_img.height // 2
        visual_result.paste(self.ov_img, (top_left_x, top_left_y), self.ov_img)

        save_path_visual = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            title="Сохранить визуальный результат как...")

        if not save_path_visual:
            messagebox.showinfo("Отмена", "Сохранение отменено.")
            return

        visual_result.save(save_path_visual)

        # --- 2. Модификация и сохранение файла сегментации ---
        if not self.current_seg_path:
            messagebox.showerror("Ошибка", "Не удалось найти файл сегментации для этого фона. Сегментация не изменена.")
            return

        try:
            # Загружаем оригинальную сегментацию
            seg_img = Image.open(self.current_seg_path)
            # Если сегментация не в режиме RGB (например, палитра 'P'), конвертируем
            if seg_img.mode != 'RGB':
                seg_img = seg_img.convert('RGB')

            # Создаем красную "заливку" размером с нашу маску
            red_fill = Image.new("RGB", self.ov_img.size, (255, 0, 0))

            # Накладываем красную заливку на сегментацию, используя альфа-канал
            # нашей маски как трафарет. Закрасятся только непрозрачные пиксели.
            seg_img.paste(red_fill, (top_left_x, top_left_y), mask=self.ov_img)

            # Сохраняем измененную сегментацию поверх старого файла
            seg_img.save(self.current_seg_path)

            messagebox.showinfo("Успех!", f"Визуальный результат сохранен в:\n{save_path_visual}\n\n"
                                          f"Сегментация обновлена:\n{self.current_seg_path}")

        except Exception as e:
            messagebox.showerror("Ошибка сохранения сегментации", f"Произошла ошибка:\n{e}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = OverlayApp()
    app.mainloop()
