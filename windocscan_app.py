
import os
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog, messagebox, Canvas

APP_TITLE = "WINDOCSCAN"
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

COLORS = {
    "bg": "#EEF3F8",
    "card": "#FFFFFF",
    "border": "#D6DEE8",
    "text": "#1F2937",
    "muted": "#5B6472",
    "primary": "#2563EB",
    "primary_hover": "#1D4ED8",
    "secondary": "#E8EEF7",
    "secondary_hover": "#DCE6F3",
    "canvas_bg": "#F2F5F9",
    "canvas_border": "#CBD5E1",
    "status_bg": "#EAF1F8",
    "danger": "#DC2626",
    "warn_bg": "#FFF7ED",
    "warn_border": "#FED7AA",
    "warn_text": "#9A3412",
}

HANDLE_RADIUS = 8
CLICK_RADIUS = 22
WIA_FORMAT_BMP = "{B96B3CAB-0728-11D3-9D7B-0000F81EF32E}"


def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), relative_path)


def now_date_string():
    return datetime.now().strftime('%Y-%m-%d')


def unique_output_path(base_dir: Path, stem_prefix="documento_scan", ext=".pdf"):
    date_str = now_date_string()
    p = base_dir / f"{stem_prefix}_{date_str}{ext}"
    if not p.exists():
        return p
    i = 2
    while True:
        p = base_dir / f"{stem_prefix}_{date_str}_{i}{ext}"
        if not p.exists():
            return p
        i += 1


def unique_stem(base_dir: Path, stem_prefix="scanner_scan"):
    date_str = now_date_string()
    p = f"{stem_prefix}_{date_str}"
    if not (base_dir / f"{p}.pdf").exists():
        return p
    i = 2
    while True:
        p = f"{stem_prefix}_{date_str}_{i}"
        if not (base_dir / f"{p}.pdf").exists():
            return p
        i += 1


def list_images_in_folder(folder):
    return [str(p) for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in SUPPORTED_EXTS and p.is_file()]


def resize_to_fit(image_bgr, max_w, max_h):
    h, w = image_bgr.shape[:2]
    if w <= 0 or h <= 0:
        return image_bgr.copy(), 1.0
    scale = min(max_w / w, max_h / h, 1.0)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b), 1)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b), 1)
    dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (max_width, max_height))



def detect_document_corners(image_bgr):
    img = image_bgr.copy()
    h, w = img.shape[:2]
    img_area = h * w
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.dilate(edged, kernel, iterations=1)
    edged = cv2.erode(edged, kernel, iterations=1)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
    thresh = 255 - thresh
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.bitwise_or(edged, thresh)
    contours, _ = cv2.findContours(combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.12:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return order_points(approx.reshape(4, 2))

    if contours:
        rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        box = cv2.boxPoints(rect)
        return order_points(box)

    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def detect_document_corners_scanner_safe(image_bgr):
    """
    Conservative detection for scanner input:
    accept only large, plausible document contours, otherwise keep full image.
    """
    img = image_bgr.copy()
    h, w = img.shape[:2]
    img_area = h * w
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.dilate(edged, kernel, iterations=1)
    edged = cv2.erode(edged, kernel, iterations=1)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.45:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        rect = order_points(approx.reshape(4, 2))
        ww = np.linalg.norm(rect[1] - rect[0])
        hh = np.linalg.norm(rect[3] - rect[0])
        if ww < w * 0.35 or hh < h * 0.35:
            continue
        return rect

    # Safe fallback: full image
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def apply_lens_correction(image_bgr, strength=0.0):
    if abs(strength) < 0.01:
        return image_bgr
    h, w = image_bgr.shape[:2]
    fx, fy, cx, cy = w, h, w / 2.0, h / 2.0
    cam = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    k1 = -float(strength) / 100.0
    dist = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    map1, map2 = cv2.initUndistortRectifyMap(cam, dist, None, cam, (w, h), cv2.CV_32FC1)
    return cv2.remap(image_bgr, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _smooth_fill(arr, default_val):
    arr = arr.astype(np.float32)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.full_like(arr, default_val, dtype=np.float32)
    x = np.arange(len(arr))
    arr[~valid] = np.interp(x[~valid], x[valid], arr[valid])
    return arr


def estimate_page_borders(gray):
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    grad = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    top_edges = np.full(w, np.nan, dtype=np.float32)
    bottom_edges = np.full(w, np.nan, dtype=np.float32)
    margin = max(12, int(w * 0.03))
    band_h = max(20, h // 3)
    for x in range(margin, w - margin):
        col = grad[:, x]
        abs_col = np.abs(col)
        t_idx = int(np.argmax(abs_col[:band_h]))
        b_idx = int(np.argmax(abs_col[h - band_h:])) + h - band_h
        if abs_col[t_idx] > 8:
            top_edges[x] = t_idx
        if abs_col[b_idx] > 8:
            bottom_edges[x] = b_idx
    top_edges = _smooth_fill(top_edges, 0)
    bottom_edges = _smooth_fill(bottom_edges, h - 1)
    k = max(31, (w // 20) | 1)
    top_edges = cv2.GaussianBlur(top_edges.reshape(1, -1), (k, 1), 0).reshape(-1)
    bottom_edges = cv2.GaussianBlur(bottom_edges.reshape(1, -1), (k, 1), 0).reshape(-1)
    return top_edges, bottom_edges


def apply_page_flatten(image_bgr, strength=0.0):
    if strength <= 0.01:
        return image_bgr
    img = image_bgr.copy()
    h, w = img.shape[:2]
    if min(h, w) < 80:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    top, bottom = estimate_page_borders(gray)
    top_ref = float(np.median(top))
    bottom_ref = float(np.median(bottom))
    eff = np.clip(float(strength) / 100.0, 0.0, 1.0)
    src_y = np.arange(h, dtype=np.float32)
    result = np.zeros_like(img)
    for x in range(w):
        t = top[x]
        b = bottom[x]
        if b - t < h * 0.35:
            result[:, x] = img[:, x]
            continue
        target_t = (1.0 - eff) * t + eff * top_ref
        target_b = (1.0 - eff) * b + eff * bottom_ref
        if target_b - target_t < 10:
            result[:, x] = img[:, x]
            continue
        mapped = (src_y - target_t) * ((b - t) / max(1.0, (target_b - target_t))) + t
        mapped = np.clip(mapped, 0, h - 1)
        for c in range(3):
            result[:, x, c] = np.interp(src_y, mapped, img[:, x, c])
    return result


def apply_enhancement(image_bgr, mode, brightness=0, contrast=1.0, white_thresh=210):
    img = image_bgr.copy()
    img = cv2.convertScaleAbs(img, alpha=float(contrast), beta=float(brightness))
    if mode == 'Documento B/N pulito':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bg = cv2.medianBlur(gray, 21)
        norm = cv2.divide(gray, bg, scale=255)
        bw = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11)
        bw[norm > white_thresh] = 255
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    if mode == 'Documento scala di grigi':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=15)
        gray = cv2.addWeighted(gray, 1.4, bg, -0.4, 0)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray[gray > white_thresh] = 255
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    mask = gray > white_thresh
    out[mask] = [255, 255, 255]
    return out


def cv_to_tk(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


def save_images_as_pdf(images_bgr, pdf_path):
    pil_images = []
    for img in images_bgr:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(rgb).convert('RGB'))
    if not pil_images:
        raise ValueError("Nessuna immagine da salvare")
    pil_images[0].save(pdf_path, save_all=True, append_images=pil_images[1:], resolution=200.0)


def save_images_as_jpg(images_bgr, output_dir: Path, base_stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for idx, img in enumerate(images_bgr, start=1):
        path = output_dir / f"{base_stem}_{idx:03d}.jpg"
        cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        saved.append(path)
    return saved


def import_wia_modules():
    try:
        import pythoncom  # noqa
        import win32com.client  # noqa
        return True, None
    except Exception as e:
        return False, e


def wia_list_devices():
    ok, err = import_wia_modules()
    if not ok:
        return []
    import win32com.client
    mgr = win32com.client.Dispatch("WIA.DeviceManager")
    devices = []
    for info in mgr.DeviceInfos:
        try:
            name = str(info.Properties("Name").Value)
        except Exception:
            name = f"Scanner {len(devices)+1}"
        try:
            dev_id = str(info.DeviceID)
        except Exception:
            dev_id = name
        devices.append({"name": name, "id": dev_id})
    return devices


def set_wia_property(item_or_props, prop_id, value):
    try:
        props = item_or_props.Properties if hasattr(item_or_props, "Properties") else item_or_props
        for p in props:
            if getattr(p, "PropertyID", None) == prop_id:
                p.Value = value
                return True
    except Exception:
        pass
    try:
        props(prop_id).Value = value
        return True
    except Exception:
        return False


def connect_wia_device(device_id):
    import win32com.client
    mgr = win32com.client.Dispatch("WIA.DeviceManager")
    return mgr.DeviceInfos(device_id).Connect()


def scan_page_wia(device_id, dpi=300, color_mode="Color", use_feeder=False):
    ok, err = import_wia_modules()
    if not ok:
        raise RuntimeError(f"Modulo WIA non disponibile: {err}")
    device = connect_wia_device(device_id)
    item = device.Items[1]
    color_map = {"Color": 1, "Grayscale": 2, "BlackWhite": 4}
    set_wia_property(item, 6146, color_map.get(color_mode, 1))
    set_wia_property(item, 6147, int(dpi))
    set_wia_property(item, 6148, int(dpi))
    set_wia_property(item, 6149, 0)
    set_wia_property(item, 6150, 0)
    set_wia_property(item, 6151, int(8.27 * dpi))
    set_wia_property(item, 6152, int(11.69 * dpi))
    if use_feeder:
        set_wia_property(device.Properties, 3088, 2)
    else:
        set_wia_property(device.Properties, 3088, 1)
    img = item.Transfer(WIA_FORMAT_BMP)
    temp_path = Path(tempfile.gettempdir()) / f"docscanner_wia_{time.time_ns()}.bmp"
    if temp_path.exists():
        temp_path.unlink()
    img.SaveFile(str(temp_path))
    cv_img = cv2.imread(str(temp_path))
    try:
        temp_path.unlink()
    except Exception:
        pass
    if cv_img is None:
        raise RuntimeError("Scansione completata ma immagine non leggibile.")
    return cv_img


def scan_via_wia_common_dialog():
    ok, err = import_wia_modules()
    if not ok:
        raise RuntimeError(f"Modulo WIA non disponibile: {err}")
    import win32com.client
    dlg = win32com.client.Dispatch("WIA.CommonDialog")
    # DeviceType ScannerDeviceType=1, Intent/ImageBias MaximizeQuality=0, Format BMP
    image = dlg.ShowAcquireImage(1, 0, 0, WIA_FORMAT_BMP, True, False, False)
    if image is None:
        raise RuntimeError("Acquisizione annullata o nessuna immagine ricevuta.")
    temp_path = Path(tempfile.gettempdir()) / f"docscanner_wia_dialog_{time.time_ns()}.bmp"
    if temp_path.exists():
        temp_path.unlink()
    image.SaveFile(str(temp_path))
    cv_img = cv2.imread(str(temp_path))
    try:
        temp_path.unlink()
    except Exception:
        pass
    if cv_img is None:
        raise RuntimeError("Immagine acquisita ma non leggibile.")
    return cv_img


class ScrollableFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        fg = self._apply_appearance_mode(self.cget("fg_color"))
        if fg in ("transparent", None, ""):
            fg = "#FFFFFF"
        self.canvas = Canvas(self, bg=fg, highlightthickness=0, bd=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar = ctk.CTkScrollbar(self, orientation="vertical", command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.inner = ctk.CTkFrame(self.canvas, fg_color="transparent")
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_inner_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)


class ScannerDialog(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Scansione da scanner")
        self.geometry("820x860")
        self.minsize(760, 760)
        self.configure(fg_color=COLORS["bg"])
        self.transient(parent)
        self.grab_set()
        self.result = None
        self.devices = []

        self.method_var = ctk.StringVar(value="Automatica integrata")
        self.device_var = ctk.StringVar(value="")
        self.scan_mode_var = ctk.StringVar(value="Pagina singola")
        self.output_var = ctk.StringVar(value="PDF unico")
        self.color_var = ctk.StringVar(value="Color")
        self.dpi_var = ctk.StringVar(value="300")
        self.source_var = ctk.StringVar(value="Vetro scanner")
        self.process_var = ctk.BooleanVar(value=True)
        self.raw_var = ctk.BooleanVar(value=False)
        self.color_policy_var = ctk.StringVar(value="Mantieni il colore scelto nello scanner")
        self.allow_delete_var = ctk.BooleanVar(value=True)
        self.allow_reorder_var = ctk.BooleanVar(value=True)

        self.build_ui()
        self.refresh_devices()

    def make_option(self, parent, title, variable, values, row, col, colspan=1):
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.grid(row=row, column=col, columnspan=colspan, sticky="ew", padx=(0 if col == 0 else 10), pady=8)
        ctk.CTkLabel(wrap, text=title, font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w")
        opt = ctk.CTkOptionMenu(
            wrap, values=values, variable=variable, height=38, corner_radius=12,
            fg_color=COLORS["secondary"], button_color=COLORS["primary"], button_hover_color=COLORS["primary_hover"],
            dropdown_fg_color="white", dropdown_text_color=COLORS["text"], text_color=COLORS["text"],
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold")
        )
        opt.pack(fill="x", pady=(6, 0))
        if title == "Scanner":
            self.device_option = opt

    def build_ui(self):
        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.pack(fill="both", expand=True, padx=18, pady=18)

        title = ctk.CTkFrame(outer, corner_radius=20, fg_color=COLORS["card"], border_width=1, border_color=COLORS["border"])
        title.pack(fill="x", pady=(0, 14))
        ctk.CTkLabel(title, text="Scansione da scanner", font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=18, pady=(16, 2))
        ctk.CTkLabel(title, text="Acquisizione diretta da scanner Windows (WIA)",
                     font=ctk.CTkFont(family="Segoe UI", size=12), text_color=COLORS["muted"]).pack(anchor="w", padx=18, pady=(0, 16))

        scroll_card = ScrollableFrame(outer, corner_radius=20, fg_color=COLORS["card"], border_width=1, border_color=COLORS["border"])
        scroll_card.pack(fill="both", expand=True)
        grid = scroll_card.inner
        for i in range(2):
            grid.grid_columnconfigure(i, weight=1)

        self.make_option(grid, "Scanner", self.device_var, ["Rilevamento..."], 0, 0)
        ctk.CTkButton(grid, text="Aggiorna elenco", width=150, height=38, corner_radius=12,
                      fg_color=COLORS["secondary"], hover_color=COLORS["secondary_hover"],
                      text_color=COLORS["text"], command=self.refresh_devices).grid(row=0, column=1, sticky="e", pady=8, padx=(10,0))

        self.make_option(grid, "Metodo di acquisizione", self.method_var,
                         ["Automatica integrata", "Avanzata con interfaccia scanner"], 1, 0, colspan=2)

        self.make_option(grid, "Modalità scansione", self.scan_mode_var,
                         ["Pagina singola", "Più pagine (manuale)", "Più pagine (caricatore automatico)"], 2, 0, colspan=2)
        self.make_option(grid, "Output finale", self.output_var,
                         ["PDF unico", "JPG separati"], 3, 0, colspan=2)
        self.make_option(grid, "Origine foglio", self.source_var,
                         ["Vetro scanner", "Caricatore fogli automatico"], 4, 0)
        self.make_option(grid, "Colore", self.color_var,
                         ["Color", "Grayscale", "BlackWhite"], 4, 1)
        self.make_option(grid, "Risoluzione DPI", self.dpi_var,
                         ["150", "200", "300", "600"], 5, 0)
        self.make_option(grid, "Colore finale scansione", self.color_policy_var,
                         ["Mantieni il colore scelto nello scanner", "Segui modalità principale dell'app"], 6, 0, colspan=2)

        tips = ctk.CTkFrame(grid, corner_radius=16, fg_color=COLORS["secondary"])
        tips.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(16, 8))
        ctk.CTkLabel(tips, text="Opzioni operative", font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=14, pady=(12, 6))
        for text, var in [
            ("Applica miglioramento automatico a ogni pagina", self.process_var),
            ("Salva scansioni grezze senza post-processing", self.raw_var),
            ("Permetti eliminazione pagine prima del salvataggio", self.allow_delete_var),
            ("Permetti riordino pagine prima del salvataggio", self.allow_reorder_var),
        ]:
            ctk.CTkCheckBox(tips, text=text, variable=var,
                            text_color=COLORS["text"], font=ctk.CTkFont(family="Segoe UI", size=12)).pack(anchor="w", padx=14, pady=4)
        ctk.CTkLabel(tips, text="", height=4).pack()

        explain = ctk.CTkFrame(grid, corner_radius=16, fg_color=COLORS["warn_bg"], border_width=1, border_color=COLORS["warn_border"])
        explain.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(8, 12))
        ctk.CTkLabel(explain, text="Spiegazione rapida", font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
                     text_color=COLORS["warn_text"]).pack(anchor="w", padx=14, pady=(12, 4))
        ctk.CTkLabel(explain,
                     text="Vetro scanner = appoggi il foglio sul piano.\nCaricatore fogli automatico = inserisci un fascio di fogli nell'alimentatore.",
                     wraplength=680, justify="left", font=ctk.CTkFont(family="Segoe UI", size=12),
                     text_color=COLORS["warn_text"]).pack(anchor="w", padx=14, pady=(0, 10))

        warn = ctk.CTkFrame(grid, corner_radius=16, fg_color=COLORS["warn_bg"], border_width=1, border_color=COLORS["warn_border"])
        warn.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ctk.CTkLabel(warn,
                     text="Nota: la compatibilità WIA dipende dai driver dello scanner. Pagina singola e multipagina manuale sono le modalità più affidabili. Il caricatore automatico dipende dal modello.",
                     wraplength=680, justify="left", font=ctk.CTkFont(family="Segoe UI", size=12),
                     text_color=COLORS["warn_text"]).pack(anchor="w", padx=14, pady=12)

        actions = ctk.CTkFrame(outer, fg_color="transparent")
        actions.pack(fill="x", pady=(12, 0))
        ctk.CTkButton(actions, text="Annulla", width=120, height=40, corner_radius=12,
                      fg_color=COLORS["secondary"], hover_color=COLORS["secondary_hover"],
                      text_color=COLORS["text"], command=self.cancel).pack(side="right", padx=(8, 0))
        ctk.CTkButton(actions, text="Avvia scansione", width=160, height=40, corner_radius=12,
                      fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"],
                      text_color="white", command=self.confirm).pack(side="right")

    def refresh_devices(self):
        try:
            devices = wia_list_devices()
        except Exception as e:
            devices = []
            messagebox.showwarning("Scanner", f"Impossibile leggere gli scanner WIA.\n\n{e}")
        self.devices = devices
        vals = [d["name"] for d in devices] if devices else ["Nessuno scanner WIA trovato"]
        self.device_option.configure(values=vals)
        self.device_var.set(vals[0])

    def selected_device(self):
        for d in self.devices:
            if d["name"] == self.device_var.get():
                return d
        return None

    def confirm(self):
        dev = self.selected_device()
        if dev is None:
            messagebox.showwarning("Scanner", "Seleziona uno scanner valido.")
            return
        self.result = {
            "method": self.method_var.get(),
            "device": dev,
            "scan_mode": self.scan_mode_var.get(),
            "output": self.output_var.get(),
            "source": self.source_var.get(),
            "color": self.color_var.get(),
            "dpi": int(self.dpi_var.get()),
            "color_policy": self.color_policy_var.get(),
            "process": bool(self.process_var.get() and not self.raw_var.get()),
            "save_raw": bool(self.raw_var.get()),
            "allow_delete": bool(self.allow_delete_var.get()),
            "allow_reorder": bool(self.allow_reorder_var.get()),
        }
        self.destroy()

    def cancel(self):
        self.result = None
        self.destroy()


class PageReviewDialog(ctk.CTkToplevel):
    def __init__(self, parent, images_bgr):
        super().__init__(parent)
        self.title("Revisione pagine")
        self.geometry("1180x760")
        self.minsize(1040, 700)
        self.configure(fg_color=COLORS["bg"])
        self.images = list(images_bgr)
        self.index = 0
        self.tk_img = None
        self.thumb_imgs = []
        self.result = None
        self.transient(parent)
        self.grab_set()
        self.build_ui()
        self.refresh_thumbnails()
        self.refresh_view()

    def build_ui(self):
        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.pack(fill="both", expand=True, padx=18, pady=18)

        top = ctk.CTkFrame(outer, corner_radius=20, fg_color=COLORS["card"], border_width=1, border_color=COLORS["border"])
        top.pack(fill="x")
        ctk.CTkLabel(top, text="Revisione pagine", font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=16, pady=(14, 4))
        self.info_label = ctk.CTkLabel(top, text="", font=ctk.CTkFont(family="Segoe UI", size=12), text_color=COLORS["muted"])
        self.info_label.pack(anchor="w", padx=16, pady=(0, 14))

        body = ctk.CTkFrame(outer, fg_color="transparent")
        body.pack(fill="both", expand=True, pady=14)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        thumbs_card = ctk.CTkFrame(body, width=240, corner_radius=20, fg_color=COLORS["card"], border_width=1, border_color=COLORS["border"])
        thumbs_card.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        thumbs_card.grid_propagate(False)
        ctk.CTkLabel(thumbs_card, text="Miniature pagine", font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=14, pady=(14, 8))
        self.thumb_scroll = ScrollableFrame(thumbs_card, fg_color="transparent")
        self.thumb_scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        preview_card = ctk.CTkFrame(body, corner_radius=20, fg_color=COLORS["card"], border_width=1, border_color=COLORS["border"])
        preview_card.grid(row=0, column=1, sticky="nsew")
        self.canvas = Canvas(preview_card, bg=COLORS["canvas_bg"], highlightthickness=1, highlightbackground=COLORS["canvas_border"], bd=0)
        self.canvas.pack(fill="both", expand=True, padx=14, pady=14)
        self.canvas.bind("<Configure>", lambda e: self.refresh_view())

        bar = ctk.CTkFrame(outer, fg_color="transparent")
        bar.pack(fill="x")
        def btn(text, cmd, fg, hover, textc, width):
            return ctk.CTkButton(bar, text=text, command=cmd, width=width, height=38, corner_radius=12,
                                 fg_color=fg, hover_color=hover, text_color=textc,
                                 font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"))
        btn("← Precedente", self.prev_page, COLORS["secondary"], COLORS["secondary_hover"], COLORS["text"], 130).pack(side="left")
        btn("Successiva →", self.next_page, COLORS["secondary"], COLORS["secondary_hover"], COLORS["text"], 130).pack(side="left", padx=8)
        btn("Sposta su", self.move_up, COLORS["secondary"], COLORS["secondary_hover"], COLORS["text"], 100).pack(side="left", padx=(16,8))
        btn("Sposta giù", self.move_down, COLORS["secondary"], COLORS["secondary_hover"], COLORS["text"], 100).pack(side="left", padx=8)
        btn("Elimina pagina", self.delete_page, "#FEE2E2", "#FECACA", COLORS["danger"], 130).pack(side="left", padx=(16,0))
        btn("Annulla", self.cancel, COLORS["secondary"], COLORS["secondary_hover"], COLORS["text"], 110).pack(side="right")
        btn("Conferma", self.confirm, COLORS["primary"], COLORS["primary_hover"], "white", 130).pack(side="right", padx=(0,8))

    def refresh_thumbnails(self):
        for w in self.thumb_scroll.inner.winfo_children():
            w.destroy()
        self.thumb_imgs = []
        for idx, img in enumerate(self.images):
            selected = idx == self.index
            card = ctk.CTkFrame(
                self.thumb_scroll.inner, corner_radius=14,
                fg_color=("#DBEAFE" if selected else COLORS["secondary"]),
                border_width=(2 if selected else 1),
                border_color=(COLORS["primary"] if selected else COLORS["border"])
            )
            card.pack(fill="x", padx=8, pady=6)
            thumb, _ = resize_to_fit(img, 180, 120)
            tk_img = cv_to_tk(thumb)
            self.thumb_imgs.append(tk_img)
            cv = Canvas(card, width=180, height=120, bg=COLORS["canvas_bg"], highlightthickness=0, bd=0)
            cv.pack(padx=10, pady=(10, 6))
            h, w = thumb.shape[:2]
            cv.create_image((180 - w) // 2, (120 - h) // 2, image=tk_img, anchor="nw")
            lbl = ctk.CTkLabel(card, text=f"Pagina {idx+1}", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                               text_color=COLORS["text"])
            lbl.pack(pady=(0, 10))
            card.bind("<Button-1>", lambda e, i=idx: self.select_page(i))
            cv.bind("<Button-1>", lambda e, i=idx: self.select_page(i))
            lbl.bind("<Button-1>", lambda e, i=idx: self.select_page(i))

    def select_page(self, idx):
        self.index = idx
        self.refresh_thumbnails()
        self.refresh_view()

    def refresh_view(self):
        self.canvas.delete("all")
        if not self.images:
            self.info_label.configure(text="Nessuna pagina disponibile.")
            return
        self.index = max(0, min(self.index, len(self.images)-1))
        self.info_label.configure(text=f"Pagina {self.index + 1} di {len(self.images)}")
        img = self.images[self.index]
        cw = max(200, self.canvas.winfo_width())
        ch = max(200, self.canvas.winfo_height())
        disp, _ = resize_to_fit(img, cw - 20, ch - 20)
        h, w = disp.shape[:2]
        off_x = (cw - w) // 2
        off_y = (ch - h) // 2
        self.tk_img = cv_to_tk(disp)
        self.canvas.create_image(off_x, off_y, image=self.tk_img, anchor="nw")

    def prev_page(self):
        if self.images:
            self.index = (self.index - 1) % len(self.images)
            self.refresh_thumbnails()
            self.refresh_view()

    def next_page(self):
        if self.images:
            self.index = (self.index + 1) % len(self.images)
            self.refresh_thumbnails()
            self.refresh_view()

    def move_up(self):
        if self.index > 0:
            self.images[self.index - 1], self.images[self.index] = self.images[self.index], self.images[self.index - 1]
            self.index -= 1
            self.refresh_thumbnails()
            self.refresh_view()

    def move_down(self):
        if self.index < len(self.images) - 1:
            self.images[self.index + 1], self.images[self.index] = self.images[self.index], self.images[self.index + 1]
            self.index += 1
            self.refresh_thumbnails()
            self.refresh_view()

    def delete_page(self):
        if not self.images:
            return
        del self.images[self.index]
        if self.index >= len(self.images):
            self.index = max(0, len(self.images)-1)
        self.refresh_thumbnails()
        self.refresh_view()

    def confirm(self):
        self.result = self.images
        self.destroy()

    def cancel(self):
        self.result = None
        self.destroy()


class SplashScreen:
    def __init__(self):
        self.win = ctk.CTk()
        self.win.overrideredirect(True)
        self.win.configure(fg_color="#EAF2FF")
        width, height = 520, 270
        x = (self.win.winfo_screenwidth() // 2) - (width // 2)
        y = (self.win.winfo_screenheight() // 2) - (height // 2)
        self.win.geometry(f"{width}x{height}+{x}+{y}")
        try:
            ico = resource_path("windocscan.ico")
            if os.path.exists(ico):
                self.win.iconbitmap(ico)
        except Exception:
            pass
        card = ctk.CTkFrame(self.win, corner_radius=24, fg_color="white", border_width=1, border_color="#D8E2EF")
        card.pack(fill="both", expand=True, padx=18, pady=18)
        ctk.CTkLabel(card, text="WINDOCSCAN", font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold"),
                     text_color=COLORS["text"]).pack(pady=(34, 6))
        ctk.CTkLabel(card, text="Preparazione applicazione…", font=ctk.CTkFont(family="Segoe UI", size=13),
                     text_color=COLORS["muted"]).pack()
        self.progress = ctk.CTkProgressBar(card, width=360, height=18, corner_radius=12,
                                           progress_color=COLORS["primary"], fg_color="#DCE7F5")
        self.progress.pack(pady=(28, 8))
        self.progress.set(0)
        self.label = ctk.CTkLabel(card, text="Caricamento moduli", font=ctk.CTkFont(family="Segoe UI", size=12),
                                  text_color=COLORS["muted"])
        self.label.pack()
        self.win.update_idletasks()

    def update(self, value, text):
        self.progress.set(value / 100.0)
        self.label.configure(text=text)
        self.win.update_idletasks()

    def close(self):
        self.win.destroy()


class DocumentScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1500x920")
        self.root.minsize(1280, 800)
        self.root.configure(fg_color=COLORS["bg"])
        try:
            ico = resource_path("windocscan.ico")
            if os.path.exists(ico):
                self.root.iconbitmap(ico)
        except Exception:
            pass

        self.current_file = None
        self.current_image = None
        self.current_corners = None
        self.processed_image = None
        self.batch_files = []
        self.before_tk = None
        self.after_tk = None
        self.before_display_scale = 1.0
        self.before_offset = (0, 0)
        self.drag_idx = None

        self.mode_var = ctk.StringVar(value='Documento B/N pulito')
        self.brightness_var = ctk.DoubleVar(value=8)
        self.contrast_var = ctk.DoubleVar(value=1.20)
        self.white_var = ctk.DoubleVar(value=215)
        self.lens_var = ctk.DoubleVar(value=0)
        self.flatten_var = ctk.DoubleVar(value=0)
        self.auto_lens_var = ctk.BooleanVar(value=False)
        self.auto_flatten_var = ctk.BooleanVar(value=False)
        self.status_var = ctk.StringVar(value='Pronto')

        self.build_ui()
        self.refresh_controls()

    def build_ui(self):
        outer = ctk.CTkFrame(self.root, fg_color="transparent")
        outer.pack(fill="both", expand=True, padx=18, pady=16)

        header = ctk.CTkFrame(outer, corner_radius=20, fg_color=COLORS["card"], border_width=1, border_color=COLORS["border"])
        header.pack(fill="x", pady=(0, 14))

        title_wrap = ctk.CTkFrame(header, fg_color="transparent")
        title_wrap.pack(side="left", padx=18, pady=16)
        ctk.CTkLabel(title_wrap, text="WINDOCSCAN", font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w")
        ctk.CTkLabel(title_wrap, text="WINDOCSCAN V7.6 • fix colore scanner",
                     font=ctk.CTkFont(family="Segoe UI", size=12), text_color=COLORS["muted"]).pack(anchor="w", pady=(2, 0))

        actions = ctk.CTkFrame(header, fg_color="transparent")
        actions.pack(side="left", padx=14, pady=12)

        btn_primary = dict(height=38, corner_radius=12, font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
                           fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"], text_color="white")
        btn_secondary = dict(height=38, corner_radius=12, font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
                             fg_color=COLORS["secondary"], hover_color=COLORS["secondary_hover"], text_color=COLORS["text"])

        ctk.CTkButton(actions, text="Apri foto e converti", command=self.open_file, width=150, **btn_primary).grid(row=0, column=0, padx=6, pady=4)
        ctk.CTkButton(actions, text="Cartella intera → PDF", command=self.open_folder, width=155, **btn_secondary).grid(row=0, column=1, padx=6, pady=4)
        ctk.CTkButton(actions, text="Scansiona da scanner", command=self.scan_from_scanner, width=170, **btn_primary).grid(row=0, column=2, padx=6, pady=4)
        ctk.CTkButton(actions, text="Auto rileva angoli", command=self.auto_detect_corners, width=145, **btn_secondary).grid(row=0, column=3, padx=6, pady=4)
        ctk.CTkButton(actions, text="Salva PDF", command=self.save_current_pdf, width=110, **btn_primary).grid(row=0, column=4, padx=6, pady=4)
        ctk.CTkButton(actions, text="Salva JPG pulito", command=self.save_current_jpg, width=130, **btn_secondary).grid(row=0, column=5, padx=6, pady=4)

        right_mode = ctk.CTkFrame(header, fg_color="transparent")
        right_mode.pack(side="right", padx=18, pady=16)
        ctk.CTkLabel(right_mode, text="Modalità", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                     text_color=COLORS["muted"]).pack(anchor="e")
        self.mode_menu = ctk.CTkOptionMenu(
            right_mode, values=['Documento B/N pulito', 'Documento a colori migliorato', 'Documento scala di grigi'],
            variable=self.mode_var, width=250, height=38, corner_radius=12,
            fg_color=COLORS["secondary"], button_color=COLORS["primary"], button_hover_color=COLORS["primary_hover"],
            dropdown_fg_color="white", dropdown_text_color=COLORS["text"], text_color=COLORS["text"],
            command=lambda _: self.reprocess_preview(), font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold")
        )
        self.mode_menu.pack(anchor="e", pady=(4, 0))

        main = ctk.CTkFrame(outer, fg_color="transparent")
        main.pack(fill="both", expand=True)

        left_card = ctk.CTkFrame(main, width=340, corner_radius=22, fg_color=COLORS["card"], border_width=1, border_color=COLORS["border"])
        left_card.pack(side="left", fill="y", padx=(0, 14))
        left_card.pack_propagate(False)
        ctk.CTkLabel(left_card, text="Controlli immagine", font=ctk.CTkFont(family="Segoe UI", size=19, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=18, pady=(18, 4))
        ctk.CTkLabel(left_card, text="Regola l'elaborazione e la geometria del documento",
                     font=ctk.CTkFont(family="Segoe UI", size=12), text_color=COLORS["muted"]).pack(anchor="w", padx=18, pady=(0, 12))

        self.s_brightness = self.make_slider(left_card, "Luminosità", self.brightness_var, -80, 80, 1)
        self.s_contrast = self.make_slider(left_card, "Contrasto", self.contrast_var, 0.5, 2.5, 0.05)
        self.s_white = self.make_slider(left_card, "Soglia bianco", self.white_var, 150, 250, 1)

        ctk.CTkFrame(left_card, height=2, fg_color="#E9EEF5").pack(fill="x", padx=18, pady=12)

        self.s_lens = self.make_slider(left_card, "Correzione lente / barilotto", self.lens_var, -40, 40, 1)
        self.auto_lens_chk = ctk.CTkCheckBox(left_card, text="Auto lieve", variable=self.auto_lens_var,
                                             command=self.reprocess_preview, text_color=COLORS["text"],
                                             font=ctk.CTkFont(family="Segoe UI", size=12))
        self.auto_lens_chk.pack(anchor="w", padx=18, pady=(0, 10))
        self.s_flatten = self.make_slider(left_card, "Appiattisci pagina", self.flatten_var, 0, 100, 1)
        self.auto_flatten_chk = ctk.CTkCheckBox(left_card, text="Auto lieve", variable=self.auto_flatten_var,
                                                command=self.reprocess_preview, text_color=COLORS["text"],
                                                font=ctk.CTkFont(family="Segoe UI", size=12))
        self.auto_flatten_chk.pack(anchor="w", padx=18, pady=(0, 10))

        hint = ctk.CTkFrame(left_card, corner_radius=16, fg_color=COLORS["secondary"])
        hint.pack(fill="x", padx=16, pady=16)
        ctk.CTkLabel(hint, text="Uso foto", font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=14, pady=(12, 6))
        ctk.CTkLabel(hint,
                     text="• trascina i 4 punti rossi nella finestra Prima\n• l'anteprima Dopo si aggiorna in tempo reale\n• doppio clic: torna all'auto-rilevamento",
                     justify="left", font=ctk.CTkFont(family="Segoe UI", size=12), text_color=COLORS["muted"], wraplength=280).pack(anchor="w", padx=14, pady=(0, 10))
        ctk.CTkLabel(hint, text="Uso scanner", font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=14, pady=(4, 6))
        ctk.CTkLabel(hint,
                     text="• scansione pagina singola o multipagina\n• PDF unico oppure JPG separati\n• revisione con miniature laterali prima del salvataggio",
                     justify="left", font=ctk.CTkFont(family="Segoe UI", size=12), text_color=COLORS["muted"], wraplength=280).pack(anchor="w", padx=14, pady=(0, 12))

        right = ctk.CTkFrame(main, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True)
        row = ctk.CTkFrame(right, fg_color="transparent")
        row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(row, text="Anteprime", font=ctk.CTkFont(family="Segoe UI", size=19, weight="bold"),
                     text_color=COLORS["text"]).pack(side="left")
        ctk.CTkLabel(row, text="prima e dopo in tempo reale", font=ctk.CTkFont(family="Segoe UI", size=12),
                     text_color=COLORS["muted"]).pack(side="left", padx=(10,0), pady=(4,0))

        previews = ctk.CTkFrame(right, fg_color="transparent")
        previews.pack(fill="both", expand=True)
        previews.grid_rowconfigure(0, weight=1)
        previews.grid_columnconfigure(0, weight=1, uniform="p")
        previews.grid_columnconfigure(1, weight=1, uniform="p")

        self.before_card = self.make_preview_card(previews, "Prima", "angoli modificabili")
        self.before_card.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        self.after_card = self.make_preview_card(previews, "Dopo", "anteprima live")
        self.after_card.grid(row=0, column=1, sticky="nsew", padx=(8,0))

        self.before_canvas = Canvas(self.before_canvas_host, bg=COLORS["canvas_bg"], highlightthickness=1,
                                    highlightbackground=COLORS["canvas_border"], bd=0)
        self.before_canvas.pack(fill="both", expand=True, padx=12, pady=(0,12))
        self.after_canvas = Canvas(self.after_canvas_host, bg=COLORS["canvas_bg"], highlightthickness=1,
                                   highlightbackground=COLORS["canvas_border"], bd=0)
        self.after_canvas.pack(fill="both", expand=True, padx=12, pady=(0,12))

        self.before_canvas.bind("<Button-1>", self.on_before_click)
        self.before_canvas.bind("<B1-Motion>", self.on_before_drag)
        self.before_canvas.bind("<ButtonRelease-1>", self.on_before_release)
        self.before_canvas.bind("<Double-Button-1>", self.on_before_double_click)
        self.before_canvas.bind("<Configure>", self.on_canvas_resize)
        self.after_canvas.bind("<Configure>", self.on_canvas_resize)

        self.status_bar = ctk.CTkFrame(outer, corner_radius=16, fg_color=COLORS["status_bg"], border_width=1, border_color=COLORS["border"])
        self.status_bar.pack(fill="x", pady=(14, 0))
        ctk.CTkLabel(self.status_bar, text="Stato", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                     text_color=COLORS["muted"]).pack(side="left", padx=(16, 8), pady=10)
        self.status_label = ctk.CTkLabel(self.status_bar, textvariable=self.status_var, font=ctk.CTkFont(family="Segoe UI", size=12),
                     text_color=COLORS["text"])
        self.status_label.pack(side="left", padx=2, pady=10)

        self.scan_progress = ctk.CTkProgressBar(self.status_bar, width=260, height=14, corner_radius=10,
                                                progress_color=COLORS["primary"], fg_color="#DCE7F5")
        self.scan_progress.set(0)

    def make_slider(self, parent, title, variable, from_, to, step):
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.pack(fill="x", padx=18, pady=(8, 0))
        row = ctk.CTkFrame(wrap, fg_color="transparent")
        row.pack(fill="x")
        ctk.CTkLabel(row, text=title, font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
                     text_color=COLORS["text"]).pack(side="left")
        label = ctk.CTkLabel(row, text=self.format_val(variable.get(), step), font=ctk.CTkFont(family="Segoe UI", size=12),
                             text_color=COLORS["muted"])
        label.pack(side="right")
        slider = ctk.CTkSlider(wrap, from_=from_, to=to, number_of_steps=max(1, int((to-from_)/step)),
                               variable=variable, progress_color=COLORS["primary"], button_color=COLORS["primary"],
                               button_hover_color=COLORS["primary_hover"],
                               command=lambda v, lbl=label, st=step: self.on_slider(v, lbl, st))
        slider.pack(fill="x", pady=(6, 4))
        return slider

    def format_val(self, v, step):
        return f"{float(v):.2f}" if step < 1 else f"{int(round(float(v)))}"

    def on_slider(self, value, label, step):
        label.configure(text=self.format_val(value, step))
        self.reprocess_preview()

    def make_preview_card(self, parent, title, subtitle):
        card = ctk.CTkFrame(parent, corner_radius=22, fg_color=COLORS["card"], border_width=1, border_color=COLORS["border"])
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.pack(fill="x", padx=14, pady=(12, 6))
        ctk.CTkLabel(head, text=title, font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
                     text_color=COLORS["text"]).pack(side="left")
        ctk.CTkLabel(head, text=subtitle, font=ctk.CTkFont(family="Segoe UI", size=12),
                     text_color=COLORS["muted"]).pack(side="left", padx=(8,0), pady=(4,0))
        host = ctk.CTkFrame(card, fg_color="transparent")
        host.pack(fill="both", expand=True)
        if title == "Prima":
            self.before_canvas_host = host
        else:
            self.after_canvas_host = host
        return card

    def refresh_controls(self):
        state = "normal" if self.current_image is not None else "disabled"
        for w in [self.s_brightness, self.s_contrast, self.s_white, self.s_lens, self.s_flatten, self.auto_lens_chk, self.auto_flatten_chk]:
            w.configure(state=state)

    def set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def show_scan_progress(self, text="Scansione in corso..."):
        self.set_status(text)
        try:
            self.scan_progress.pack(side="right", padx=(8, 16), pady=10)
            self.scan_progress.configure(mode="indeterminate")
            self.scan_progress.start()
        except Exception:
            pass
        self.root.update_idletasks()

    def set_scan_progress_value(self, value, text=None):
        try:
            self.scan_progress.stop()
            self.scan_progress.configure(mode="determinate")
            self.scan_progress.set(max(0.0, min(1.0, float(value))))
        except Exception:
            pass
        if text is not None:
            self.set_status(text)
        self.root.update_idletasks()

    def hide_scan_progress(self, text="Pronto"):
        try:
            self.scan_progress.stop()
            self.scan_progress.pack_forget()
            self.scan_progress.set(0)
        except Exception:
            pass
        self.set_status(text)
        self.root.update_idletasks()

    def open_file(self):
        path = filedialog.askopenfilename(title='Seleziona una foto documento',
                                          filetypes=[('Immagini', '*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff')])
        if path:
            self.load_single_image(path)

    def open_folder(self):
        folder = filedialog.askdirectory(title='Seleziona cartella immagini')
        if not folder:
            return
        files = list_images_in_folder(folder)
        if not files:
            messagebox.showwarning('Nessuna immagine', 'Nella cartella non ci sono immagini supportate.')
            return
        self.batch_files = files
        self.load_single_image(files[0])
        self.save_batch_folder_pdf()

    def load_single_image(self, path):
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror('Errore', 'Impossibile aprire il file immagine.')
            return
        self.current_file = path
        self.current_image = img
        self.batch_files = [path]
        self.current_corners = detect_document_corners(img)
        self.refresh_controls()
        self.set_status(f'Caricato: {os.path.basename(path)}')
        self.redraw_before_canvas()
        self.reprocess_preview()

    def auto_detect_corners(self):
        if self.current_image is None:
            messagebox.showinfo('Info', 'Apri prima una foto.')
            return
        self.current_corners = detect_document_corners(self.current_image)
        self.redraw_before_canvas()
        self.reprocess_preview()
        self.set_status('Auto-rilevamento angoli eseguito.')

    def on_before_double_click(self, event):
        self.auto_detect_corners()

    def get_geometry_settings(self):
        lens = float(self.lens_var.get())
        flat = float(self.flatten_var.get())
        if self.auto_lens_var.get() and abs(lens) < 0.01:
            lens = 10.0
        if self.auto_flatten_var.get() and flat < 0.01:
            flat = 35.0
        return lens, flat

    def process_image_object(self, img, corners=None):
        corners = detect_document_corners(img) if corners is None else corners
        warped = four_point_transform(img, corners)
        lens, flat = self.get_geometry_settings()
        corrected = apply_lens_correction(warped, lens)
        flattened = apply_page_flatten(corrected, flat)
        return apply_enhancement(flattened, self.mode_var.get(), self.brightness_var.get(), self.contrast_var.get(), int(self.white_var.get()))


    def maybe_process_scanned(self, img, do_process=True, color_policy="Segui modalità principale dell'app", scanner_color="Color"):
        if not do_process:
            return img

        corners = detect_document_corners_scanner_safe(img)

        if color_policy == "Segui modalità principale dell'app":
            return self.process_image_object(img, corners)

        warped = four_point_transform(img, corners)
        lens, flat = self.get_geometry_settings()
        corrected = apply_lens_correction(warped, lens)
        flattened = apply_page_flatten(corrected, flat)

        if scanner_color == "BlackWhite":
            gray = cv2.cvtColor(flattened, cv2.COLOR_BGR2GRAY)
            bw = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
            )
            return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

        if scanner_color == "Grayscale":
            gray = cv2.cvtColor(flattened, cv2.COLOR_BGR2GRAY)
            gray = cv2.convertScaleAbs(
                gray, alpha=float(self.contrast_var.get()), beta=float(self.brightness_var.get())
            )
            gray[gray > int(self.white_var.get())] = 255
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        color = cv2.convertScaleAbs(
            flattened, alpha=float(self.contrast_var.get()), beta=float(self.brightness_var.get())
        )
        lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        l = clahe.apply(l)
        merged = cv2.merge((l, a, b))
        out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return out

    def _get_canvas_size(self, canvas):
        self.root.update_idletasks()
        return max(200, canvas.winfo_width()), max(200, canvas.winfo_height())

    def redraw_before_canvas(self):
        c = self.before_canvas
        c.delete("all")
        if self.current_image is None:
            return
        cw, ch = self._get_canvas_size(c)
        disp, scale = resize_to_fit(self.current_image, cw - 20, ch - 20)
        h, w = disp.shape[:2]
        off_x, off_y = (cw - w) // 2, (ch - h) // 2
        self.before_display_scale = scale
        self.before_offset = (off_x, off_y)
        self.before_tk = cv_to_tk(disp)
        c.create_image(off_x, off_y, image=self.before_tk, anchor='nw')
        if self.current_corners is not None:
            pts = (self.current_corners * scale).astype(np.float32)
            pts[:, 0] += off_x
            pts[:, 1] += off_y
            pts_i = pts.astype(int)
            for a, b in [(0,1),(1,2),(2,3),(3,0)]:
                c.create_line(*pts_i[a], *pts_i[b], fill='#22C55E', width=3)
            for i, (x, y) in enumerate(pts_i):
                r = HANDLE_RADIUS
                c.create_oval(x-r, y-r, x+r, y+r, fill='#F97316', outline='white', width=2)
                c.create_text(x+12, y-12, text=str(i+1), fill='#F59E0B', font=('Segoe UI', 11, 'bold'))

    def redraw_after_canvas(self):
        c = self.after_canvas
        c.delete("all")
        if self.processed_image is None:
            return
        cw, ch = self._get_canvas_size(c)
        disp, _ = resize_to_fit(self.processed_image, cw - 20, ch - 20)
        h, w = disp.shape[:2]
        off_x, off_y = (cw - w) // 2, (ch - h) // 2
        self.after_tk = cv_to_tk(disp)
        c.create_image(off_x, off_y, image=self.after_tk, anchor='nw')

    def reprocess_preview(self):
        if self.current_image is None or self.current_corners is None:
            self.redraw_before_canvas()
            self.redraw_after_canvas()
            return
        try:
            self.processed_image = self.process_image_object(self.current_image, self.current_corners)
            self.redraw_before_canvas()
            self.redraw_after_canvas()
            lens, flat = self.get_geometry_settings()
            self.set_status(f'Anteprima aggiornata | lente: {lens:.0f} | appiattimento: {flat:.0f}')
        except Exception as e:
            self.set_status(f'Errore elaborazione: {e}')

    def _canvas_to_image_point(self, x, y):
        if self.current_image is None:
            return None
        off_x, off_y = self.before_offset
        scale = self.before_display_scale
        if scale <= 0:
            return None
        img_x = float(np.clip((x - off_x) / scale, 0, self.current_image.shape[1] - 1))
        img_y = float(np.clip((y - off_y) / scale, 0, self.current_image.shape[0] - 1))
        return np.array([img_x, img_y], dtype=np.float32)

    def _nearest_corner_index(self, x, y):
        if self.current_corners is None:
            return None
        off_x, off_y = self.before_offset
        scale = self.before_display_scale
        pts = (self.current_corners * scale).astype(np.float32)
        pts[:, 0] += off_x
        pts[:, 1] += off_y
        d = np.sqrt((pts[:,0]-x)**2 + (pts[:,1]-y)**2)
        idx = int(np.argmin(d))
        return idx if d[idx] <= CLICK_RADIUS else None

    def on_before_click(self, event):
        if self.current_image is not None and self.current_corners is not None:
            self.drag_idx = self._nearest_corner_index(event.x, event.y)

    def on_before_drag(self, event):
        if self.current_image is None or self.current_corners is None or self.drag_idx is None:
            return
        pt = self._canvas_to_image_point(event.x, event.y)
        if pt is not None:
            self.current_corners[self.drag_idx] = pt
            self.reprocess_preview()

    def on_before_release(self, event):
        self.drag_idx = None

    def on_canvas_resize(self, event):
        if self.current_image is not None:
            self.redraw_before_canvas()
        if self.processed_image is not None:
            self.redraw_after_canvas()

    def save_current_pdf(self):
        if self.processed_image is None or self.current_file is None:
            messagebox.showinfo('Info', 'Apri prima una foto.')
            return
        base_dir = Path(self.current_file).parent
        suggested = unique_output_path(base_dir)
        out = filedialog.asksaveasfilename(title='Salva PDF', defaultextension='.pdf', initialfile=suggested.name,
                                           initialdir=str(base_dir), filetypes=[('PDF', '*.pdf')])
        if out:
            save_images_as_pdf([self.processed_image], out)
            self.set_status(f'PDF salvato: {out}')
            messagebox.showinfo('OK', f'PDF salvato:\n{out}')

    def save_current_jpg(self):
        if self.processed_image is None or self.current_file is None:
            messagebox.showinfo('Info', 'Apri prima una foto.')
            return
        base_dir = Path(self.current_file).parent
        stem = Path(self.current_file).stem + "_pulito.jpg"
        out = filedialog.asksaveasfilename(title='Salva JPG pulito', defaultextension='.jpg', initialfile=stem,
                                           initialdir=str(base_dir), filetypes=[('JPEG', '*.jpg')])
        if out:
            cv2.imwrite(out, self.processed_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            self.set_status(f'JPG salvato: {out}')
            messagebox.showinfo('OK', f'JPG salvato:\n{out}')

    def save_batch_folder_pdf(self):
        if not self.batch_files:
            return
        images = []
        total = len(self.batch_files)
        for idx, path in enumerate(self.batch_files, start=1):
            img = cv2.imread(path)
            if img is None:
                continue
            images.append(self.process_image_object(img, detect_document_corners(img)))
            self.set_status(f'Elaboro cartella: {idx}/{total} - {os.path.basename(path)}')
        if not images:
            messagebox.showwarning('Errore', 'Nessuna immagine valida nella cartella.')
            return
        folder = Path(self.batch_files[0]).parent
        out = unique_output_path(folder)
        save_images_as_pdf(images, str(out))
        self.set_status(f'PDF multipagina salvato: {out}')
        messagebox.showinfo('OK', f'PDF multipagina salvato:\n{out}')

    def ask_output_destination(self, output_mode, stem_prefix="scanner_scan"):
        folder = filedialog.askdirectory(title="Scegli la cartella di salvataggio")
        if not folder:
            return None, None
        folder_path = Path(folder)
        stem = unique_stem(folder_path, stem_prefix)
        return (folder_path / f"{stem}.pdf", stem) if output_mode == "PDF unico" else (folder_path, stem)

    

    def scan_from_scanner(self):
        ok, err = import_wia_modules()
        if not ok:
            messagebox.showwarning("Scanner", "Per usare la scansione diretta serve Windows con WIA disponibile.\n\n"
                                   "Installa pywin32 e usa uno scanner con driver WIA.\n\n"
                                   f"Dettaglio: {err}")
            return
        dlg = ScannerDialog(self.root)
        self.root.wait_window(dlg)
        cfg = dlg.result
        if not cfg:
            return
        try:
            dest, stem = self.ask_output_destination(cfg["output"], "scanner_scan")
            if dest is None:
                return

            raw_pages, final_pages = [], []
            use_feeder = cfg["source"] == "Caricatore fogli automatico" or cfg["scan_mode"] == "Più pagine (caricatore automatico)"
            advanced = cfg.get("method") == "Avanzata con interfaccia scanner"

            self.show_scan_progress("Scansione in corso...")

            if advanced:
                self.show_scan_progress("Scansione avanzata in corso...")
                img = scan_via_wia_common_dialog()
                raw_pages = [img]
            else:
                if cfg["scan_mode"] == "Pagina singola":
                    self.show_scan_progress("Scansione pagina singola in corso...")
                    raw_pages = [scan_page_wia(cfg["device"]["id"], cfg["dpi"], cfg["color"], use_feeder)]
                elif cfg["scan_mode"] == "Più pagine (manuale)":
                    idx = 1
                    while True:
                        self.show_scan_progress(f"Scansione pagina {idx} in corso...")
                        raw_pages.append(scan_page_wia(cfg["device"]["id"], cfg["dpi"], cfg["color"], False))
                        self.hide_scan_progress(f"Pagina {idx} acquisita")
                        if not messagebox.askyesno("Aggiungere pagina", f"Pagina {idx} acquisita.\n\nVuoi acquisire un'altra pagina?"):
                            break
                        self.show_scan_progress(f"Preparazione scansione pagina {idx+1}...")
                        idx += 1
                else:
                    idx = 1
                    while True:
                        try:
                            self.show_scan_progress(f"Scansione pagina {idx} da caricatore automatico...")
                            raw_pages.append(scan_page_wia(cfg["device"]["id"], cfg["dpi"], cfg["color"], True))
                            idx += 1
                            if idx > 200:
                                break
                        except Exception:
                            break
                    if not raw_pages:
                        raise RuntimeError("Nessuna pagina acquisita dal caricatore automatico. Controlla driver e caricamento fogli.")

            if not raw_pages:
                self.hide_scan_progress("Nessuna pagina acquisita")
                messagebox.showwarning("Scanner", "Nessuna pagina acquisita.")
                return

            total = len(raw_pages)
            for idx, img in enumerate(raw_pages, start=1):
                self.set_scan_progress_value(idx / max(1, total), f"Elaborazione pagina {idx}/{total}...")
                final_pages.append(self.maybe_process_scanned(img, cfg["process"], cfg.get("color_policy", "Segui modalità principale dell'app"), cfg.get("color", "Color")))

            self.hide_scan_progress("Revisione pagine...")
            if cfg["allow_delete"] or cfg["allow_reorder"]:
                review = PageReviewDialog(self.root, final_pages)
                self.root.wait_window(review)
                if review.result is None:
                    self.hide_scan_progress("Operazione annullata")
                    return
                final_pages = review.result
                if not final_pages:
                    messagebox.showwarning("Scanner", "Tutte le pagine sono state eliminate.")
                    self.hide_scan_progress("Nessuna pagina da salvare")
                    return

            self.show_scan_progress("Salvataggio in corso...")
            if cfg["output"] == "PDF unico":
                save_images_as_pdf(final_pages, str(dest))
                self.hide_scan_progress(f"PDF scanner salvato: {dest}")
                messagebox.showinfo("OK", f"PDF creato con {len(final_pages)} pagina/e:\n{dest}")
            else:
                saved = save_images_as_jpg(final_pages, Path(dest), stem)
                self.hide_scan_progress(f"JPG scanner salvati in: {dest}")
                messagebox.showinfo("OK", f"Salvate {len(saved)} immagini JPG.\n\nCartella:\n{dest}")

            preview = final_pages[0]
            temp_path = Path(tempfile.gettempdir()) / f"windocscan_preview_{time.time_ns()}.jpg"
            cv2.imwrite(str(temp_path), preview)
            self.load_single_image(str(temp_path))
            try:
                temp_path.unlink()
            except Exception:
                pass

        except Exception as e:
            self.hide_scan_progress(f"Errore scanner: {e}")
            messagebox.showerror("Scanner", f"Errore durante la scansione:\n\n{e}")

def start_app():
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    splash = SplashScreen()
    for val, text in [
        (18, "Caricamento interfaccia"),
        (42, "Inizializzazione motore immagini"),
        (66, "Preparazione strumenti PDF"),
        (84, "Inizializzazione supporto scanner"),
        (100, "Avvio applicazione"),
    ]:
        splash.update(val, text)
        splash.win.update()
        time.sleep(0.18)
    splash.close()
    root = ctk.CTk()
    DocumentScannerApp(root)
    root.mainloop()


if __name__ == '__main__':
    start_app()
