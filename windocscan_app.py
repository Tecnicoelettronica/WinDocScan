
import os
import sys
import time
import json
import tempfile
import threading
import time as thread_time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk
try:
    import fitz
except Exception:
    fitz = None
import customtkinter as ctk
from tkinter import filedialog, messagebox, Canvas

APP_TITLE = "WINDOCSCAN"
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
PDF_EXTS = {'.pdf'}

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
APP_VERSION = "8.0.0"
SETTINGS_DIR = Path.home() / ".windocscan"
SETTINGS_FILE = SETTINGS_DIR / "settings.json"


def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), relative_path)


def now_date_string():
    return datetime.now().strftime('%Y-%m-%d')

def now_timestamp_string():
    return datetime.now().strftime('%Y-%m-%d_%H-%M')


def default_scan_filename(ext=".pdf"):
    return f"scan_{now_timestamp_string()}{ext}"


def load_app_settings():
    try:
        if SETTINGS_FILE.exists():
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_app_settings(data):
    try:
        SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def default_documents_dir():
    docs = Path.home() / "Documents"
    return docs if docs.exists() else Path.home()



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


def render_pdf_to_bgr_images(pdf_path, dpi=160):
    if fitz is None:
        raise RuntimeError('Supporto PDF non disponibile. Installa PyMuPDF (fitz).')
    pages = []
    zoom = max(1.0, float(dpi) / 72.0)
    mat = fitz.Matrix(zoom, zoom)
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            mode = 'RGB' if pix.n < 4 else 'RGBA'
            pil = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            if mode == 'RGBA':
                pil = pil.convert('RGB')
            arr = np.array(pil)
            pages.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    finally:
        doc.close()
    return pages


def make_copy_output_path(original_path, page_num=None):
    p = Path(original_path)
    if page_num is None:
        return p.with_name(f'{p.stem}_copia.jpg')
    return p.with_name(f'{p.stem}_pag{page_num:03d}_copia.jpg')


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



def apply_rotation(image, angle_deg):
    try:
        angle = float(angle_deg)
    except Exception:
        angle = 0.0
    if abs(angle) < 0.01:
        return image.copy()
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(m[0, 0])
    sin = abs(m[0, 1])
    new_w = max(1, int((h * sin) + (w * cos)))
    new_h = max(1, int((h * cos) + (w * sin)))
    m[0, 2] += (new_w / 2.0) - center[0]
    m[1, 2] += (new_h / 2.0) - center[1]
    if len(image.shape) == 2:
        border = 255
    else:
        border = (255, 255, 255)
    return cv2.warpAffine(image, m, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


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
    if mode == 'Ripristina':
        return image_bgr.copy()
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


def apply_sharpen(image_bgr, amount=0):
    try:
        amount = float(amount)
    except Exception:
        amount = 0.0
    if amount <= 0.01:
        return image_bgr

    # Unsharp mask più evidente del precedente, pensato per documenti e scrittura.
    strength = np.clip(amount / 100.0, 0.0, 2.5)
    sigma = 1.0 + (0.6 * min(strength, 1.5))
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(image_bgr, 1.0 + 1.8 * strength, blurred, -1.8 * strength, 0)

    # Piccolo rinforzo locale dei contorni per rendere visibile l'effetto anche su testo fine.
    if len(image_bgr.shape) == 3 and image_bgr.shape[2] == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr.copy()
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    edge_boost = np.clip((0.35 * strength) * lap, -35, 35)
    if len(sharp.shape) == 2:
        boosted = sharp.astype(np.float32) - edge_boost
    else:
        boosted = sharp.astype(np.float32)
        for c in range(3):
            boosted[:, :, c] = boosted[:, :, c] - edge_boost
    return np.clip(boosted, 0, 255).astype(np.uint8)


def cv_to_tk(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


def save_images_as_pdf(images_bgr, pdf_path):
    pil_images = []
    for img in images_bgr:
        if img is None:
            continue
        if len(img.shape) == 2:
            pil_images.append(Image.fromarray(img).convert('RGB'))
            continue
        if img.shape[2] == 4:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(rgb).convert('RGB'))
    if not pil_images:
        raise ValueError("Nessuna immagine da salvare")
    pil_images[0].save(str(pdf_path), save_all=True, append_images=pil_images[1:], resolution=200.0)


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
        self.geometry("360x820")
        self.minsize(340, 700)
        self.configure(fg_color=COLORS["bg"])
        self.transient(parent)
        self.grab_set()
        self.result = None
        self.devices = []
        self.settings = load_app_settings()
        scanner_cfg = self.settings.get('scanner_settings', {})

        self.method_var = ctk.StringVar(value=scanner_cfg.get('method', 'Automatica integrata'))
        self.device_var = ctk.StringVar(value=scanner_cfg.get('device_name', ''))
        self.scan_mode_var = ctk.StringVar(value=scanner_cfg.get('scan_mode', 'Pagina singola'))
        self.output_var = ctk.StringVar(value=scanner_cfg.get('output', 'PDF unico'))
        self.color_var = ctk.StringVar(value=scanner_cfg.get('color', 'Color'))
        self.dpi_var = ctk.StringVar(value=str(scanner_cfg.get('dpi', '300')))
        self.source_var = ctk.StringVar(value=scanner_cfg.get('source', 'Vetro scanner'))
        self.process_var = ctk.BooleanVar(value=scanner_cfg.get('process', True))
        self.raw_var = ctk.BooleanVar(value=scanner_cfg.get('save_raw', False))
        self.color_policy_var = ctk.StringVar(value=scanner_cfg.get('color_policy', 'Mantieni il colore scelto nello scanner'))
        self.allow_delete_var = ctk.BooleanVar(value=scanner_cfg.get('allow_delete', True))
        self.allow_reorder_var = ctk.BooleanVar(value=scanner_cfg.get('allow_reorder', True))

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
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold")
        )
        opt.pack(fill="x", pady=(6, 0))
        if title == "Scanner":
            self.device_option = opt

    def persist_settings(self):
        self.settings["last_used_dir"] = self.last_used_dir
        save_app_settings(self.settings)

    def update_last_used_dir(self, path_like):
        if not path_like:
            return
        p = Path(path_like)
        folder = p if p.is_dir() else p.parent
        if folder and folder.exists():
            self.last_used_dir = str(folder)
            self.persist_settings()

    def get_initial_dir(self):
        if self.last_used_dir and Path(self.last_used_dir).exists():
            return self.last_used_dir
        if self.current_file:
            try:
                p = Path(self.current_file)
                folder = p if p.is_dir() else p.parent
                if folder.exists():
                    return str(folder)
            except Exception:
                pass
        return str(default_documents_dir())

    def get_default_save_name(self, ext='.pdf', prefix='scan'):
        return f"{prefix}_{now_timestamp_string()}{ext}"

    def setup_drag_and_drop(self):
        self.drop_enabled = False
        try:
            import windnd
            self._windnd = windnd

            def callback(files):
                self.root.after(0, lambda f=files: self.handle_drop_files(f))

            for widget in (self.root, self.before_canvas, self.after_canvas):
                try:
                    windnd.hook_dropfiles(widget, func=callback)
                except Exception:
                    pass
            self.drop_enabled = True
            self.set_status("Pronto • puoi anche trascinare file o cartelle dentro l'app")
        except Exception:
            self._windnd = None

    def handle_drop_files(self, files):
        paths = []
        for item in files:
            if isinstance(item, bytes):
                try:
                    paths.append(item.decode('utf-8'))
                except Exception:
                    paths.append(item.decode('mbcs', errors='ignore'))
            else:
                paths.append(str(item))
        paths = [p.strip().strip('{}') for p in paths if p]
        if not paths:
            return
        if self.documents and not self.confirm_save_before_switch('file'):
            return
        added = self.import_paths(paths, reset_existing=True)
        if not added:
            messagebox.showwarning('Drag&Drop', 'Trascina immagini, PDF oppure cartelle con immagini.')

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
        grid.grid_columnconfigure(0, weight=1)

        self.make_option(grid, "Scanner", self.device_var, ["Rilevamento..."], 0, 0)
        ctk.CTkButton(grid, text="Aggiorna elenco", width=120, height=34, corner_radius=12,
                      fg_color=COLORS["secondary"], hover_color=COLORS["secondary_hover"],
                      text_color=COLORS["text"], command=self.refresh_devices).grid(row=1, column=0, sticky="e", pady=(0,8))

        self.make_option(grid, "Metodo di acquisizione", self.method_var,
                         ["Automatica integrata", "Avanzata con interfaccia scanner"], 2, 0)

        self.make_option(grid, "Modalità scansione", self.scan_mode_var,
                         ["Pagina singola", "Più pagine (manuale)", "Più pagine (caricatore automatico)"], 3, 0)
        self.make_option(grid, "Output finale", self.output_var,
                         ["PDF unico", "JPG separati"], 4, 0)
        self.make_option(grid, "Origine foglio", self.source_var,
                         ["Vetro scanner", "Caricatore fogli automatico"], 5, 0)
        self.make_option(grid, "Colore", self.color_var,
                         ["Color", "Grayscale", "BlackWhite"], 6, 0)
        self.make_option(grid, "Risoluzione DPI", self.dpi_var,
                         ["150", "200", "300", "600"], 7, 0)
        self.make_option(grid, "Colore finale scansione", self.color_policy_var,
                         ["Mantieni il colore scelto nello scanner", "Segui modalità principale dell'app"], 8, 0)

        tips = ctk.CTkFrame(grid, corner_radius=16, fg_color=COLORS["secondary"])
        tips.grid(row=11, column=0, sticky="ew", pady=(16, 8))
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
        explain.grid(row=9, column=0, columnspan=1, sticky="ew", pady=(8, 12))
        ctk.CTkLabel(explain, text="Spiegazione rapida", font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
                     text_color=COLORS["warn_text"]).pack(anchor="w", padx=14, pady=(12, 4))
        ctk.CTkLabel(explain,
                     text="Vetro scanner = appoggi il foglio sul piano.\nCaricatore fogli automatico = inserisci un fascio di fogli nell'alimentatore.",
                     wraplength=300, justify="left", font=ctk.CTkFont(family="Segoe UI", size=12),
                     text_color=COLORS["warn_text"]).pack(anchor="w", padx=14, pady=(0, 10))

        warn = ctk.CTkFrame(grid, corner_radius=16, fg_color=COLORS["warn_bg"], border_width=1, border_color=COLORS["warn_border"])
        warn.grid(row=10, column=0, columnspan=1, sticky="ew", pady=(0, 8))
        ctk.CTkLabel(warn,
                     text="Nota: la compatibilità WIA dipende dai driver dello scanner. Pagina singola e multipagina manuale sono le modalità più affidabili. Il caricatore automatico dipende dal modello.",
                     wraplength=300, justify="left", font=ctk.CTkFont(family="Segoe UI", size=12),
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
        vals = [d['name'] for d in devices] if devices else ['Nessuno scanner WIA trovato']
        self.device_option.configure(values=vals)
        preferred = self.device_var.get()
        self.device_var.set(preferred if preferred in vals else vals[0])

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
        self.settings['scanner_settings'] = {
            'method': self.method_var.get(),
            'device_name': self.device_var.get(),
            'scan_mode': self.scan_mode_var.get(),
            'output': self.output_var.get(),
            'source': self.source_var.get(),
            'color': self.color_var.get(),
            'dpi': self.dpi_var.get(),
            'color_policy': self.color_policy_var.get(),
            'process': bool(self.process_var.get()),
            'save_raw': bool(self.raw_var.get()),
            'allow_delete': bool(self.allow_delete_var.get()),
            'allow_reorder': bool(self.allow_reorder_var.get()),
        }
        save_app_settings(self.settings)
        self.destroy()

    def cancel(self):
        self.result = None
        self.destroy()


class PageReviewDialog(ctk.CTkToplevel):
    def __init__(self, parent, images_bgr, ask_continue_on_confirm=False, page_number=None):
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
        self.ask_continue_on_confirm = bool(ask_continue_on_confirm)
        self.page_number = page_number
        self.transient(parent)
        self.grab_set()
        self.build_ui()
        self.refresh_thumbnails()
        self.refresh_view()

    def persist_settings(self):
        self.settings["last_used_dir"] = self.last_used_dir
        save_app_settings(self.settings)

    def update_last_used_dir(self, path_like):
        if not path_like:
            return
        p = Path(path_like)
        folder = p if p.is_dir() else p.parent
        if folder and folder.exists():
            self.last_used_dir = str(folder)
            self.persist_settings()

    def get_initial_dir(self):
        if self.last_used_dir and Path(self.last_used_dir).exists():
            return self.last_used_dir
        if self.current_file:
            try:
                p = Path(self.current_file)
                folder = p if p.is_dir() else p.parent
                if folder.exists():
                    return str(folder)
            except Exception:
                pass
        return str(default_documents_dir())

    def get_default_save_name(self, ext='.pdf', prefix='scan'):
        return f"{prefix}_{now_timestamp_string()}{ext}"

    def setup_drag_and_drop(self):
        self.drop_enabled = False
        try:
            import windnd
            self._windnd = windnd

            def callback(files):
                self.root.after(0, lambda f=files: self.handle_drop_files(f))

            for widget in (self.root, self.before_canvas, self.after_canvas):
                try:
                    windnd.hook_dropfiles(widget, func=callback)
                except Exception:
                    pass
            self.drop_enabled = True
            self.set_status("Pronto • puoi anche trascinare file o cartelle dentro l'app")
        except Exception:
            self._windnd = None

    def handle_drop_files(self, files):
        paths = []
        for item in files:
            if isinstance(item, bytes):
                try:
                    paths.append(item.decode('utf-8'))
                except Exception:
                    paths.append(item.decode('mbcs', errors='ignore'))
            else:
                paths.append(str(item))
        paths = [p.strip().strip('{}') for p in paths if p]
        if not paths:
            return
        if self.documents and not self.confirm_save_before_switch('file'):
            return
        added = self.import_paths(paths, reset_existing=True)
        if not added:
            messagebox.showwarning('Drag&Drop', 'Trascina immagini, PDF oppure cartelle con immagini.')

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
        if self.ask_continue_on_confirm:
            btn("Termina scansione e salva", self.finish_and_save, COLORS["primary"], COLORS["primary_hover"], "white", 190).pack(side="right", padx=(0,8))
            btn("Scansiona nuova pagina", self.confirm_and_add_more, COLORS["secondary"], COLORS["secondary_hover"], COLORS["text"], 200).pack(side="right", padx=(0,8))
            btn("Rifai questa pagina", self.rescan_current, COLORS["secondary"], COLORS["secondary_hover"], COLORS["text"], 170).pack(side="right", padx=(0,8))
        else:
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
        self.result = list(self.images)
        self.destroy()

    def confirm_and_add_more(self):
        self.result = {"images": list(self.images), "action": "add_more"}
        self.destroy()

    def finish_and_save(self):
        self.result = {"images": list(self.images), "action": "finish"}
        self.destroy()

    def rescan_current(self):
        self.result = {"images": list(self.images), "action": "rescan_current", "index": self.index}
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
        self.root.geometry("1840x1080")
        self.root.minsize(1600, 920)
        self.root.configure(fg_color=COLORS["bg"])
        try:
            ico = resource_path("windocscan.ico")
            if os.path.exists(ico):
                self.root.iconbitmap(ico)
        except Exception:
            pass

        self.current_file = None
        self.documents = []
        self.current_index = -1
        self.doc_thumb_refs = []
        self.current_image = None
        self.current_corners = None
        self.processed_image = None
        self.batch_files = []
        self.before_tk = None
        self.after_tk = None
        self.before_display_scale = 1.0
        self.before_offset = (0, 0)
        self.drag_idx = None
        self.current_source_type = None
        self.has_unsaved_changes = False
        self.settings = load_app_settings()
        self.last_used_dir = self.settings.get("last_used_dir")
        if not self.last_used_dir or not Path(self.last_used_dir).exists():
            self.last_used_dir = str(default_documents_dir())

        self.mode_var = ctk.StringVar(value='Originale')
        self.brightness_var = ctk.DoubleVar(value=0)
        self.contrast_var = ctk.DoubleVar(value=1.00)
        self.white_var = ctk.DoubleVar(value=215)
        self.lens_var = ctk.DoubleVar(value=0)
        self.flatten_var = ctk.DoubleVar(value=0)
        self.rotation_var = ctk.DoubleVar(value=0)
        self.sharpness_var = ctk.DoubleVar(value=0)
        self.auto_lens_var = ctk.BooleanVar(value=False)
        self.auto_flatten_var = ctk.BooleanVar(value=False)
        self.status_var = ctk.StringVar(value='Pronto')

        self.build_ui()
        self.mode_var.set('Originale')
        self.setup_drag_and_drop()
        self.refresh_controls()

    def persist_settings(self):
        self.settings["last_used_dir"] = self.last_used_dir
        save_app_settings(self.settings)

    def update_last_used_dir(self, path_like):
        if not path_like:
            return
        p = Path(path_like)
        folder = p if p.is_dir() else p.parent
        if folder and folder.exists():
            self.last_used_dir = str(folder)
            self.persist_settings()

    def get_initial_dir(self):
        if self.last_used_dir and Path(self.last_used_dir).exists():
            return self.last_used_dir
        if self.current_file:
            try:
                p = Path(self.current_file)
                folder = p if p.is_dir() else p.parent
                if folder.exists():
                    return str(folder)
            except Exception:
                pass
        return str(default_documents_dir())

    def get_default_save_name(self, ext='.pdf', prefix='scan'):
        return f"{prefix}_{now_timestamp_string()}{ext}"

    def setup_drag_and_drop(self):
        self.drop_enabled = False
        try:
            import windnd
            self._windnd = windnd

            def callback(files):
                self.root.after(0, lambda f=files: self.handle_drop_files(f))

            for widget in (self.root, self.before_canvas, self.after_canvas):
                try:
                    windnd.hook_dropfiles(widget, func=callback)
                except Exception:
                    pass
            self.drop_enabled = True
            self.set_status("Pronto • puoi anche trascinare file o cartelle dentro l'app")
        except Exception:
            self._windnd = None

    def handle_drop_files(self, files):
        paths = []
        for item in files:
            if isinstance(item, bytes):
                try:
                    paths.append(item.decode('utf-8'))
                except Exception:
                    paths.append(item.decode('mbcs', errors='ignore'))
            else:
                paths.append(str(item))
        paths = [p.strip().strip('{}') for p in paths if p]
        if not paths:
            return
        if self.documents and not self.confirm_save_before_switch('file'):
            return
        added = self.import_paths(paths, reset_existing=True)
        if not added:
            messagebox.showwarning('Drag&Drop', 'Trascina immagini, PDF oppure cartelle con immagini.')

    def build_ui(self):
        outer = ctk.CTkFrame(self.root, fg_color="transparent")
        outer.pack(fill="both", expand=True, padx=18, pady=16)

        header = ctk.CTkFrame(outer, corner_radius=20, fg_color=COLORS["card"], border_width=1, border_color=COLORS["border"])
        header.pack(fill="x", pady=(0, 14))

        title_wrap = ctk.CTkFrame(header, fg_color="transparent")
        title_wrap.pack(side="left", padx=18, pady=16)
        ctk.CTkLabel(title_wrap, text="WINDOCSCAN", font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w")
        ctk.CTkLabel(title_wrap, text=f" V{APP_VERSION} ®Tecnicoelettronica",
                     font=ctk.CTkFont(family="Segoe UI", size=12), text_color=COLORS["muted"]).pack(anchor="w", pady=(2, 0))

        actions = ctk.CTkFrame(header, fg_color="transparent")
        actions.pack(side="left", padx=14, pady=12)

        btn_primary = dict(height=34, corner_radius=11, font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                           fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"], text_color="white")
        btn_secondary = dict(height=34, corner_radius=11, font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                             fg_color=COLORS["secondary"], hover_color=COLORS["secondary_hover"], text_color=COLORS["text"])

        ctk.CTkButton(actions, text="Apri e modifica", command=self.open_file, width=138, **btn_primary).grid(row=0, column=0, padx=6, pady=4)
        ctk.CTkButton(actions, text="Esporta PDF unico", command=self.export_all_to_pdf, width=146, **btn_secondary).grid(row=0, column=1, padx=6, pady=4)
        ctk.CTkButton(actions, text="Scansiona da scanner", command=self.scan_from_scanner, width=162, **btn_primary).grid(row=0, column=2, padx=6, pady=4)
        ctk.CTkButton(actions, text="Auto rileva angoli", command=self.auto_detect_corners, width=138, **btn_secondary).grid(row=0, column=3, padx=6, pady=4)
        ctk.CTkButton(actions, text="Salva PDF", command=self.export_all_to_pdf, width=104, **btn_primary).grid(row=0, column=4, padx=6, pady=4)
        ctk.CTkButton(actions, text="Salva JPG pulito", command=self.save_current_jpg, width=122, **btn_secondary).grid(row=0, column=5, padx=6, pady=4)

        right_mode = ctk.CTkFrame(header, fg_color="transparent")
        right_mode.pack(side="right", padx=18, pady=16)
        ctk.CTkLabel(right_mode, text="Modalità", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                     text_color=COLORS["muted"]).pack(anchor="e")
        self.mode_menu = ctk.CTkOptionMenu(
            right_mode, values=['Originale', 'Ripristina', 'Documento a colori migliorato', 'Documento scala di grigi', 'Documento B/N pulito'],
            variable=self.mode_var, width=235, height=34, corner_radius=11,
            fg_color=COLORS["secondary"], button_color=COLORS["primary"], button_hover_color=COLORS["primary_hover"],
            dropdown_fg_color="white", dropdown_text_color=COLORS["text"], text_color=COLORS["text"],
            command=self.on_mode_menu_change, font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold")
        )
        self.mode_menu.pack(anchor="e", pady=(4, 0))

        main = ctk.CTkFrame(outer, fg_color="transparent")
        main.pack(fill="both", expand=True)

        left_card = ctk.CTkFrame(main, width=308, corner_radius=22, fg_color=COLORS["card"], border_width=1, border_color=COLORS["border"])
        left_card.pack(side="left", fill="y", padx=(0, 14))
        left_card.pack_propagate(False)
        ctk.CTkLabel(left_card, text="Controlli immagine", font=ctk.CTkFont(family="Segoe UI", size=19, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=18, pady=(18, 4))
        ctk.CTkLabel(left_card, text="Regola l'elaborazione e la geometria del documento",
                     font=ctk.CTkFont(family="Segoe UI", size=12), text_color=COLORS["muted"]).pack(anchor="w", padx=18, pady=(0, 14))

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
        self.s_rotation = self.make_slider(left_card, "Rotazione", self.rotation_var, -180, 180, 1)
        self.s_sharpness = self.make_slider(left_card, "Definizione / sharpen", self.sharpness_var, 0, 200, 1)

        self.info_panel = ctk.CTkFrame(left_card, corner_radius=16, fg_color=COLORS["secondary"], height=174)
        self.info_panel.pack(fill="x", padx=16, pady=(18, 16))
        self.info_panel.pack_propagate(False)
        ctk.CTkLabel(
            self.info_panel,
            text="Info pagina",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=COLORS["text"]
        ).pack(anchor="w", padx=14, pady=(12, 4))
        self.info_label = ctk.CTkLabel(
            self.info_panel,
            text="Nessuna pagina caricata",
            justify="left",
            anchor="nw",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLORS["muted"],
            wraplength=272,
        )
        self.info_label.pack(fill="both", expand=True, padx=14, pady=(2, 12))

        right = ctk.CTkFrame(main, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True)
        row = ctk.CTkFrame(right, fg_color="transparent")
        row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(row, text="Anteprime", font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
                     text_color=COLORS["text"]).pack(side="left")
        ctk.CTkLabel(row, text="prima e dopo in tempo reale", font=ctk.CTkFont(family="Segoe UI", size=12),
                     text_color=COLORS["muted"]).pack(side="left", padx=(10,0), pady=(4,0))

        previews = ctk.CTkFrame(right, fg_color="transparent")
        previews.pack(fill="both", expand=True)
        previews.grid_rowconfigure(0, weight=1)
        previews.grid_columnconfigure(0, weight=1, uniform="p")
        previews.grid_columnconfigure(1, weight=1, uniform="p")
        previews.grid_columnconfigure(2, weight=1, uniform="p")

        self.docs_card = ctk.CTkFrame(previews, width=320, corner_radius=22, fg_color=COLORS["card"], border_width=1, border_color=COLORS["border"])
        self.docs_card.grid_rowconfigure(1, weight=1)
        self.docs_card.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        self.docs_card.grid_propagate(False)
        ctk.CTkLabel(self.docs_card, text="Pagine / miniature", font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"), text_color=COLORS["text"]).pack(anchor="w", padx=14, pady=(14,6))
        self.docs_scroll = ScrollableFrame(self.docs_card, fg_color=COLORS["canvas_bg"], border_width=1, border_color=COLORS["border"])
        self.docs_scroll.pack(fill="both", expand=True, padx=12, pady=(0,12))
        thumbs_bar = ctk.CTkFrame(self.docs_card, fg_color="transparent")
        thumbs_bar.pack(side="bottom", fill="x", padx=10, pady=(0,10))
        ctk.CTkButton(thumbs_bar, text="Nuova scan", command=self.scan_add_page_from_scanner, width=88, **btn_primary).pack(side="left", padx=(0,6))
        ctk.CTkButton(thumbs_bar, text="Rifai", command=self.rescan_selected_page_from_scanner, width=60, **btn_secondary).pack(side="left", padx=4)
        ctk.CTkButton(thumbs_bar, text="Su", command=self.move_current_doc_up, width=46, **btn_secondary).pack(side="left", padx=4)
        ctk.CTkButton(thumbs_bar, text="Giù", command=self.move_current_doc_down, width=46, **btn_secondary).pack(side="left", padx=4)
        ctk.CTkButton(thumbs_bar, text="Elimina", command=self.delete_current_doc, width=80, fg_color="#FEE2E2", hover_color="#FECACA", text_color=COLORS["danger"], height=34, corner_radius=11, font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold")).pack(side="right")

        self.before_card = self.make_preview_card(previews, "Prima", "angoli modificabili")
        self.before_card.grid(row=0, column=1, sticky="nsew", padx=(8,8))
        self.after_card = self.make_preview_card(previews, "Dopo", "anteprima live")
        self.after_card.grid(row=0, column=2, sticky="nsew", padx=(8,0))

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
        ctk.CTkLabel(head, text=title, font=ctk.CTkFont(family="Segoe UI", size=17, weight="bold"),
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
        for w in [self.s_brightness, self.s_contrast, self.s_white, self.s_lens, self.s_flatten, self.s_rotation, self.s_sharpness, self.auto_lens_chk, self.auto_flatten_chk]:
            w.configure(state=state)
        self.update_info_panel()

    def update_info_panel(self):
        if not hasattr(self, 'info_label'):
            return
        doc = self.current_doc()
        if doc is None or not self.documents:
            self.info_label.configure(text='Nessuna pagina caricata')
            return
        idx = self.current_index + 1
        total = len(self.documents)
        source_type = doc.get('source_type', 'file')
        origine = 'scanner' if source_type == 'scanner' else 'file'
        mode = self.mode_var.get() if hasattr(self, 'mode_var') else 'Originale'
        dpi = doc.get('dpi')
        dpi_text = f'{int(dpi)} DPI' if isinstance(dpi, (int, float)) else '—'
        try:
            rotation = float(self.rotation_var.get())
        except Exception:
            rotation = 0.0
        stato = 'modificata' if self._doc_is_modified(doc) else 'originale'
        text = (
            f'Pagina: {idx} / {total}\n'
            f'Origine: {origine}\n'
            f'Modalità: {mode}\n'
            f'Risoluzione: {dpi_text}\n'
            f'Rotazione: {rotation:.0f}°\n'
            f'Stato: {stato}'
        )
        self.info_label.configure(text=text)
        self.info_label.update_idletasks()

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

    def ui_call_and_wait(self, func):
        holder = {"done": False, "result": None, "error": None}
        def wrapper():
            try:
                holder["result"] = func()
            except Exception as e:
                holder["error"] = e
            finally:
                holder["done"] = True
        self.root.after(0, wrapper)
        while not holder["done"]:
            thread_time.sleep(0.02)
        if holder["error"] is not None:
            raise holder["error"]
        return holder["result"]

    def current_doc(self):
        if 0 <= self.current_index < len(self.documents):
            return self.documents[self.current_index]
        return None

    def mark_dirty(self):
        doc = self.current_doc()
        if doc is not None:
            doc['dirty'] = True
        self.has_unsaved_changes = True

    def _geometry_is_default(self):
        try:
            return (
                abs(float(self.brightness_var.get())) < 0.01 and
                abs(float(self.contrast_var.get()) - 1.0) < 0.01 and
                int(round(float(self.white_var.get()))) == 215 and
                abs(float(self.lens_var.get())) < 0.01 and
                abs(float(self.flatten_var.get())) < 0.01 and
                abs(float(self.rotation_var.get())) < 0.01 and
                abs(float(self.sharpness_var.get())) < 0.01 and
                not bool(self.auto_lens_var.get()) and
                not bool(self.auto_flatten_var.get()) and
                self.mode_var.get() == 'Originale'
            )
        except Exception:
            return False

    def _doc_is_modified(self, doc=None):
        doc = doc or self.current_doc()
        if doc is None:
            return False
        if doc is not self.current_doc():
            return bool(doc.get('dirty'))
        orig = doc.get('original_corners')
        cur = self.current_corners
        corners_changed = False
        if orig is None and cur is None:
            corners_changed = False
        elif orig is None or cur is None:
            corners_changed = True
        else:
            try:
                corners_changed = not np.allclose(np.asarray(orig, dtype=np.float32), np.asarray(cur, dtype=np.float32), atol=1.0)
            except Exception:
                corners_changed = True
        return corners_changed or (not self._geometry_is_default())

    def _refresh_unsaved_flag(self):
        self.has_unsaved_changes = any(bool(doc.get('dirty')) for doc in self.documents)

    def persist_current_document_state(self):
        doc = self.current_doc()
        if doc is None or self.current_image is None:
            return
        doc['current_image'] = self.current_image.copy()
        doc['corners'] = self.current_corners.copy() if self.current_corners is not None else None
        doc['processed'] = self.processed_image.copy() if self.processed_image is not None else None

    def sync_from_document(self):
        doc = self.current_doc()
        if doc is None:
            self.current_file = None
            self.current_image = None
            self.current_corners = None
            self.processed_image = None
            self.current_source_type = None
            self.refresh_controls()
            self.redraw_before_canvas()
            self.redraw_after_canvas()
            self.refresh_doc_thumbnails()
            self.update_info_panel()
            return
        self.current_file = doc.get('file_path') or doc.get('name')
        self.current_image = doc['current_image'].copy()
        self.current_corners = doc['corners'].copy() if doc.get('corners') is not None else None
        self.processed_image = doc['processed'].copy() if doc.get('processed') is not None else None
        self.current_source_type = doc.get('source_type', 'file')
        self.refresh_controls()
        self.redraw_before_canvas()
        self.redraw_after_canvas()
        self.refresh_doc_thumbnails()
        self._refresh_unsaved_flag()
        self.update_info_panel()

    def add_document(self, img, source_label, file_path=None, source_type='file', corners=None, copy_path=None, dpi=None):
        if img is None:
            return
        if corners is None:
            corners = detect_document_corners_scanner_safe(img) if source_type == 'scanner' else detect_document_corners(img)
        doc = {
            'name': source_label,
            'file_path': file_path,
            'source_type': source_type,
            'original_image': img.copy(),
            'current_image': img.copy(),
            'original_corners': corners.copy() if corners is not None else None,
            'corners': corners.copy() if corners is not None else None,
            'processed': None,
            'copy_path': str(copy_path) if copy_path else None,
            'selected_for_pdf': False,
            'dirty': False,
            'dpi': dpi,
        }
        self.documents.append(doc)
        self.current_index = len(self.documents) - 1
        self.sync_from_document()
        self.reprocess_preview()

    def scan_add_page_from_scanner(self):
        ok, err = import_wia_modules()
        if not ok:
            messagebox.showwarning('Scanner', f'Scanner non disponibile.\n\n{err}')
            return
        dlg = ScannerDialog(self.root)
        self.root.wait_window(dlg)
        cfg = dlg.result
        if not cfg:
            return
        cfg = dict(cfg)
        cfg['scan_mode'] = 'Pagina singola'
        threading.Thread(target=self._scan_single_into_collection_worker, args=(cfg, None), daemon=True).start()

    def rescan_selected_page_from_scanner(self):
        if self.current_index < 0:
            messagebox.showinfo('Scanner', 'Seleziona prima una pagina da rifare.')
            return
        ok, err = import_wia_modules()
        if not ok:
            messagebox.showwarning('Scanner', f'Scanner non disponibile.\n\n{err}')
            return
        dlg = ScannerDialog(self.root)
        self.root.wait_window(dlg)
        cfg = dlg.result
        if not cfg:
            return
        cfg = dict(cfg)
        cfg['scan_mode'] = 'Pagina singola'
        threading.Thread(target=self._scan_single_into_collection_worker, args=(cfg, self.current_index), daemon=True).start()

    def _scan_single_into_collection_worker(self, cfg, replace_index=None):
        try:
            self.root.after(0, lambda: self.show_scan_progress('Scansione pagina in corso...'))
            advanced = cfg.get('method') == 'Avanzata con interfaccia scanner'
            use_feeder = cfg.get('source') == 'Caricatore fogli automatico'
            raw_img = scan_via_wia_common_dialog() if advanced else scan_page_wia(cfg['device']['id'], cfg['dpi'], cfg['color'], use_feeder)
            processed_img = self.maybe_process_scanned(raw_img, cfg['process'], cfg.get('color_policy', "Segui modalità principale dell'app"), cfg.get('color', 'Color'))
            corners = detect_document_corners_scanner_safe(raw_img)
            def inject():
                if replace_index is not None and 0 <= replace_index < len(self.documents):
                    doc = self.documents[replace_index]
                    doc['original_image'] = raw_img.copy()
                    doc['current_image'] = raw_img.copy()
                    doc['original_corners'] = corners.copy()
                    doc['corners'] = corners.copy()
                    doc['processed'] = processed_img.copy()
                    doc['source_type'] = 'scanner'
                    doc['dirty'] = False
                    self.current_index = replace_index
                else:
                    self.add_document(raw_img, f'scanner pagina {len(self.documents)+1}', None, 'scanner', corners, dpi=cfg.get('dpi'))
                    self.documents[-1]['processed'] = processed_img.copy()
                    self.current_index = len(self.documents)-1
                self.sync_from_document()
                self.refresh_doc_thumbnails()
                self.hide_scan_progress('Scansione integrata completata')
            self.root.after(0, inject)
        except Exception as e:
            self.root.after(0, lambda: self.hide_scan_progress(f'Errore scanner: {e}'))
            self.root.after(0, lambda: messagebox.showerror('Scanner', f'Errore durante la scansione:\n\n{e}'))

    def refresh_doc_thumbnails(self):
        if not hasattr(self, 'docs_scroll'):
            return
        for w in self.docs_scroll.inner.winfo_children():
            w.destroy()
        self.doc_thumb_refs = []
        self.doc_select_vars = []
        if not self.documents:
            ctk.CTkLabel(self.docs_scroll.inner, text='Nessuna pagina caricata', text_color=COLORS['muted']).pack(anchor='w', padx=10, pady=10)
            return
        any_marked = any(doc.get('selected_for_pdf') for doc in self.documents)
        for idx, doc in enumerate(self.documents):
            current = idx == self.current_index
            marked = bool(doc.get('selected_for_pdf'))
            fg = COLORS['secondary']
            border_color = COLORS['border']
            border_width = 1
            if marked and current:
                fg = '#DBEAFE'
                border_color = '#16A34A'
                border_width = 2
            elif current:
                fg = '#DBEAFE'
                border_color = COLORS['primary']
                border_width = 2
            elif marked:
                fg = '#ECFDF5'
                border_color = '#16A34A'
                border_width = 2
            card = ctk.CTkFrame(self.docs_scroll.inner, corner_radius=14, fg_color=fg, border_width=border_width, border_color=border_color)
            card.pack(fill='x', padx=6, pady=6)

            top = ctk.CTkFrame(card, fg_color='transparent')
            top.pack(fill='x', padx=10, pady=(8,0))
            ctk.CTkLabel(top, text=f'Pagina {idx+1}', font=ctk.CTkFont(family='Segoe UI', size=11, weight='bold'), text_color=COLORS['muted']).pack(side='left')
            var = ctk.BooleanVar(value=marked)
            self.doc_select_vars.append(var)
            cb = ctk.CTkCheckBox(top, text='PDF', variable=var, width=72, checkbox_width=18, checkbox_height=18,
                                 border_width=2, corner_radius=5,
                                 command=lambda i=idx, v=var: self.toggle_document_pdf_selection(i, bool(v.get())),
                                 text_color=COLORS['text'], font=ctk.CTkFont(family='Segoe UI', size=11, weight='bold'))
            cb.pack(side='right')

            thumb_src = doc.get('processed') if doc.get('processed') is not None else doc.get('current_image')
            thumb, _ = resize_to_fit(thumb_src, 180, 120)
            tk_img = cv_to_tk(thumb)
            self.doc_thumb_refs.append(tk_img)
            cv = Canvas(card, width=180, height=120, bg=COLORS['canvas_bg'], highlightthickness=1, highlightbackground=COLORS['canvas_border'], bd=0)
            cv.pack(padx=10, pady=(6, 6))
            h, w = thumb.shape[:2]
            cv.create_image((180-w)//2, (120-h)//2, image=tk_img, anchor='nw')
            label = doc['name']
            if len(label) > 24:
                label = label[:21] + '...'
            ctk.CTkLabel(card, text=f'{idx+1}. {label}', font=ctk.CTkFont(family='Segoe UI', size=12, weight='bold'), text_color=COLORS['text']).pack(anchor='w', padx=10, pady=(0,2))
            if marked:
                note = 'selezionata per PDF'
            elif any_marked:
                note = 'non inclusa nel PDF'
            else:
                note = 'tutte le pagine saranno incluse'
            ctk.CTkLabel(card, text=note, font=ctk.CTkFont(family='Segoe UI', size=11), text_color=COLORS['muted']).pack(anchor='w', padx=10, pady=(0,10))
            for widget in (card, cv):
                widget.bind('<Button-1>', lambda e, i=idx: self.select_document(i))

    def toggle_document_pdf_selection(self, idx, selected):
        if idx < 0 or idx >= len(self.documents):
            return
        self.documents[idx]['selected_for_pdf'] = bool(selected)
        marked = sum(1 for doc in self.documents if doc.get('selected_for_pdf'))
        if marked:
            self.set_status(f"Selezione PDF: {marked} pagina/e marcate. Salva PDF userà solo quelle nell'ordine mostrato.")
        else:
            self.set_status("Nessuna miniatura marcata: Salva PDF userà tutte le pagine nell'ordine mostrato.")
        self.refresh_doc_thumbnails()

    def select_document(self, idx):
        if idx < 0 or idx >= len(self.documents):
            return
        self.persist_current_document_state()
        self.current_index = idx
        self.sync_from_document()
        self.set_status(f'Pagina selezionata: {self.documents[idx]["name"]}')

    def move_current_doc_up(self):
        if self.current_index > 0:
            self.persist_current_document_state()
            i = self.current_index
            self.documents[i-1], self.documents[i] = self.documents[i], self.documents[i-1]
            self.current_index -= 1
            self.sync_from_document()

    def move_current_doc_down(self):
        if 0 <= self.current_index < len(self.documents)-1:
            self.persist_current_document_state()
            i = self.current_index
            self.documents[i+1], self.documents[i] = self.documents[i], self.documents[i+1]
            self.current_index += 1
            self.sync_from_document()

    def delete_current_doc(self):
        if not self.documents or self.current_index < 0:
            return
        del self.documents[self.current_index]
        if self.current_index >= len(self.documents):
            self.current_index = len(self.documents)-1
        self.sync_from_document()
        self.set_status('Pagina eliminata')

    def autosave_current_copy(self):
        doc = self.current_doc()
        if doc is None or self.processed_image is None:
            return
        copy_path = doc.get('copy_path')
        if not copy_path and doc.get('file_path'):
            src = Path(doc['file_path'])
            if src.suffix.lower() in PDF_EXTS:
                page_num = int(doc.get('page_num', self.current_index+1))
                copy_path = str(make_copy_output_path(src, page_num=page_num))
            else:
                copy_path = str(make_copy_output_path(src))
            doc['copy_path'] = copy_path
        if copy_path:
            Path(copy_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(copy_path), self.processed_image, [int(cv2.IMWRITE_JPEG_QUALITY),95])

    def clear_all_documents(self):
        self.documents = []
        self.current_index = -1
        self.sync_from_document()

    def import_paths(self, paths, reset_existing=True):
        if reset_existing:
            self.clear_all_documents()
        added = 0
        first_dir = None
        for path in paths:
            p = Path(path)
            if not p.exists():
                continue
            if first_dir is None:
                first_dir = str(p.parent if p.is_file() else p)
            if p.is_dir():
                for sub in list_images_in_folder(str(p)):
                    self.import_paths([sub], reset_existing=False)
                continue
            ext = p.suffix.lower()
            if ext in SUPPORTED_EXTS:
                img = cv2.imread(str(p))
                if img is not None:
                    self.add_document(img, p.name, str(p), 'file')
                    added += 1
            elif ext in PDF_EXTS:
                pages = render_pdf_to_bgr_images(str(p))
                for page_no, img in enumerate(pages, start=1):
                    self.add_document(img, f'{p.stem} p.{page_no}', str(p), 'file', copy_path=make_copy_output_path(p, page_no))
                    self.documents[-1]['page_num'] = page_no
                    added += 1
        if first_dir:
            self.update_last_used_dir(first_dir)
        if added:
            self.current_index = 0
            self.sync_from_document()
            self.set_status(f'Caricate {added} pagine da file/PDF. Puoi anche marcare singole miniature per il PDF finale.')
        return added


    def open_file(self):
        paths = filedialog.askopenfilenames(
            title='Seleziona 1 o più immagini o PDF',
            initialdir=self.get_initial_dir(),
            filetypes=[('Immagini e PDF', '*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.pdf'), ('Tutti i file supportati', '*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.pdf')]
        )
        if not paths:
            return
        if self.documents and not self.confirm_save_before_switch('file'):
            return
        added = self.import_paths(list(paths), reset_existing=True)
        if not added:
            messagebox.showwarning('Importazione', 'Nessun file valido selezionato.')

    def open_folder(self):
        self.open_file()

    def open_folder_path(self, folder):
        self.import_paths([folder], reset_existing=True)

    def _load_image_common(self, img, source_label, file_path=None, corners=None):
        self.clear_all_documents()
        self.add_document(img, source_label, file_path, 'file', corners)

    def load_single_image(self, path):
        self.import_paths([path], reset_existing=True)

    def load_single_image_from_array(self, img, label='anteprima_scanner.jpg', corners=None):
        self.clear_all_documents()
        self.add_document(img, label, None, 'scanner', corners)

    def auto_detect_corners(self):
        if self.current_image is None:
            messagebox.showinfo('Info', 'Apri prima una foto.')
            return
        self.current_corners = detect_document_corners(self.current_image)
        self.redraw_before_canvas()
        self.reprocess_preview()
        self.set_status('Auto-rilevamento angoli eseguito.')

    def confirm_save_before_switch(self, incoming_source_type):
        if not self.documents:
            return True
        answer = messagebox.askyesnocancel(
            'Salvare modifiche',
            "C'è un lavoro in corso.\n\nVuoi esportare prima il PDF unico del lavoro attuale?"
        )
        if answer is None:
            return False
        if answer is False:
            return True
        saved = self.export_all_to_pdf()
        return bool(saved)

    def on_before_double_click(self, event):
        self.auto_detect_corners()

    def get_geometry_settings(self):
        lens = float(self.lens_var.get())
        flat = float(self.flatten_var.get())
        rotation = float(self.rotation_var.get())
        if self.auto_lens_var.get() and abs(lens) < 0.01:
            lens = 10.0
        if self.auto_flatten_var.get() and flat < 0.01:
            flat = 35.0
        return lens, flat, rotation

    def process_image_object(self, img, corners=None, mode_override=None):
        mode = mode_override or self.mode_var.get()
        if mode not in {'Originale', 'Ripristina', 'Documento a colori migliorato', 'Documento scala di grigi', 'Documento B/N pulito'}:
            mode = 'Originale'
        corners = detect_document_corners(img) if corners is None else corners
        warped = four_point_transform(img, corners)
        lens, flat, rotation = self.get_geometry_settings()
        corrected = apply_lens_correction(warped, lens)
        flattened = apply_page_flatten(corrected, flat)
        rotated = apply_rotation(flattened, rotation)
        sharpened = apply_sharpen(rotated, self.sharpness_var.get())
        if mode == 'Ripristina':
            return apply_rotation(warped, rotation)
        if mode == 'Originale':
            return cv2.convertScaleAbs(sharpened, alpha=float(self.contrast_var.get()), beta=float(self.brightness_var.get()))
        if mode == 'Documento a colori migliorato':
            return apply_enhancement(sharpened, 'Documento a colori migliorato', self.brightness_var.get(), self.contrast_var.get(), int(self.white_var.get()))
        if mode == 'Documento scala di grigi':
            return apply_enhancement(sharpened, 'Documento scala di grigi', self.brightness_var.get(), self.contrast_var.get(), int(self.white_var.get()))
        return apply_enhancement(sharpened, 'Documento B/N pulito', self.brightness_var.get(), self.contrast_var.get(), int(self.white_var.get()))



    def reset_controls_to_defaults(self):
        self.brightness_var.set(0)
        self.contrast_var.set(1.00)
        self.white_var.set(215)
        self.lens_var.set(0)
        self.flatten_var.set(0)
        self.rotation_var.set(0)
        self.sharpness_var.set(0)
        self.auto_lens_var.set(False)
        self.auto_flatten_var.set(False)

    def reset_current_document_to_original(self):
        doc = self.current_doc()
        if doc is None:
            return
        base_img = doc.get('original_image') if doc.get('original_image') is not None else doc.get('current_image')
        if base_img is None:
            return
        self.current_image = base_img.copy()
        base_corners = doc.get('original_corners')
        if base_corners is None:
            if doc.get('source_type') == 'scanner':
                base_corners = detect_document_corners_scanner_safe(self.current_image)
            else:
                base_corners = detect_document_corners(self.current_image)
        self.current_corners = base_corners.copy() if base_corners is not None else None
        self.reset_controls_to_defaults()
        # Ripristina deve tornare davvero allo stato originale, anche per il pannello info.
        self.mode_var.set('Originale')
        self.processed_image = self.process_image_object(self.current_image, self.current_corners, mode_override='Originale')
        doc['current_image'] = self.current_image.copy()
        doc['corners'] = self.current_corners.copy() if self.current_corners is not None else None
        doc['processed'] = self.processed_image.copy() if self.processed_image is not None else None
        doc['dirty'] = False
        self._refresh_unsaved_flag()
        self.redraw_before_canvas()
        self.redraw_after_canvas()
        self.refresh_doc_thumbnails()
        self.set_status('Pagina ripristinata ai valori originali')
        self.update_info_panel()

    def on_mode_menu_change(self, selected_mode):
        if selected_mode == 'Ripristina':
            self.reset_current_document_to_original()
            self.mode_var.set('Originale')
            return
        self.reprocess_preview()

    def maybe_process_scanned(self, img, do_process=True, color_policy="Segui modalità principale dell'app", scanner_color="Color"):
        if not do_process:
            return img

        corners = detect_document_corners_scanner_safe(img)

        if color_policy == "Segui modalità principale dell'app":
            return self.process_image_object(img, corners)

        warped = four_point_transform(img, corners)
        lens, flat, rotation = self.get_geometry_settings()
        corrected = apply_lens_correction(warped, lens)
        flattened = apply_page_flatten(corrected, flat)
        rotated = apply_rotation(flattened, rotation)
        sharpened = apply_sharpen(rotated, self.sharpness_var.get())

        if scanner_color == "BlackWhite":
            gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
            bw = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
            )
            return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

        if scanner_color == "Grayscale":
            gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
            gray = cv2.convertScaleAbs(
                gray, alpha=float(self.contrast_var.get()), beta=float(self.brightness_var.get())
            )
            gray[gray > int(self.white_var.get())] = 255
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        color = cv2.convertScaleAbs(
            sharpened, alpha=float(self.contrast_var.get()), beta=float(self.brightness_var.get())
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
            doc = self.current_doc()
            if doc is not None:
                doc['current_image'] = self.current_image.copy()
                doc['corners'] = self.current_corners.copy()
                doc['processed'] = self.processed_image.copy()
                doc['dirty'] = self._doc_is_modified(doc)
            self._refresh_unsaved_flag()
            self.redraw_before_canvas()
            self.redraw_after_canvas()
            self.refresh_doc_thumbnails()
            if doc is not None and doc.get('dirty'):
                self.autosave_current_copy()
            lens, flat, rotation = self.get_geometry_settings()
            sharp = float(self.sharpness_var.get()) if hasattr(self, 'sharpness_var') else 0.0
            self.set_status(f'Anteprima aggiornata | lente: {lens:.0f} | appiattimento: {flat:.0f} | rotazione: {rotation:.0f}° | definizione: {sharp:.0f}')
            self.update_info_panel()
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
            self.has_unsaved_changes = True
            self.reprocess_preview()

    def on_before_release(self, event):
        self.drag_idx = None

    def on_canvas_resize(self, event):
        if self.current_image is not None:
            self.redraw_before_canvas()
        if self.processed_image is not None:
            self.redraw_after_canvas()

    def save_processed_to_path(self, out_path):
        out_path = str(out_path)
        ext = Path(out_path).suffix.lower()
        if ext == '.pdf':
            save_images_as_pdf([self.processed_image], out_path)
        elif ext in {'.jpg', '.jpeg'}:
            cv2.imwrite(out_path, self.processed_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            raise ValueError('Formato di salvataggio non supportato.')
        self.has_unsaved_changes = False
        self.update_last_used_dir(out_path)
        return out_path

    def save_current_pdf(self, prompt_on_missing=True):
        if self.processed_image is None:
            if prompt_on_missing:
                messagebox.showinfo('Info', 'Apri prima una pagina.')
            return None
        base_dir = self.get_initial_dir()
        current_name = 'pagina'
        doc = self.current_doc()
        if doc is not None:
            current_name = Path(doc.get('name','pagina')).stem
        suggested = f'{current_name}_copia.pdf'
        out = filedialog.asksaveasfilename(title='Salva PDF pagina corrente', defaultextension='.pdf', initialfile=suggested, initialdir=base_dir, filetypes=[('PDF', '*.pdf')])
        if out:
            self.save_processed_to_path(out)
            self.set_status(f'PDF pagina salvato: {out}')
            messagebox.showinfo('OK', f'PDF salvato:\n{out}')
            return out
        return None

    def save_current_jpg(self):
        if self.processed_image is None:
            messagebox.showinfo('Info', 'Apri prima una pagina.')
            return None
        base_dir = self.get_initial_dir()
        doc = self.current_doc()
        suggested = Path(doc.get('copy_path') or f"{Path(doc.get('name','pagina')).stem}_copia.jpg").name if doc else self.get_default_save_name('.jpg', 'scan')
        out = filedialog.asksaveasfilename(title='Salva JPG pulito', defaultextension='.jpg', initialfile=suggested, initialdir=base_dir, filetypes=[('JPEG', '*.jpg')])
        if out:
            self.save_processed_to_path(out)
            self.set_status(f'JPG salvato: {out}')
            messagebox.showinfo('OK', f'JPG salvato:\n{out}')
            return out
        return None

    def export_all_to_pdf(self):
        if not self.documents:
            messagebox.showinfo('Info', 'Carica prima una o più immagini o PDF.')
            return None
        self.persist_current_document_state()

        marked_docs = [doc for doc in self.documents if doc.get('selected_for_pdf')]
        docs_to_export = marked_docs if marked_docs else list(self.documents)

        images = []
        for doc in docs_to_export:
            img = doc.get('processed')
            if img is None:
                img = doc.get('current_image')
            if img is not None:
                images.append(img)

        if not images:
            messagebox.showwarning('Errore', 'Nessuna pagina valida da esportare.')
            return None

        out = filedialog.asksaveasfilename(
            title='Esporta PDF unico',
            defaultextension='.pdf',
            initialfile=self.get_default_save_name('.pdf', 'scan'),
            initialdir=self.get_initial_dir(),
            filetypes=[('PDF', '*.pdf')]
        )
        if not out:
            return None

        try:
            save_images_as_pdf(images, str(out))
        except Exception as e:
            messagebox.showerror('Errore', f'Impossibile salvare il PDF.\n\n{e}')
            return None

        self.has_unsaved_changes = False
        self.update_last_used_dir(out)
        count = len(images)
        if marked_docs:
            self.set_status(f'PDF salvato con {count} pagina/e selezionate: {out}')
        else:
            self.set_status(f'PDF salvato con tutte le {count} pagine: {out}')
        messagebox.showinfo('OK', f'PDF multipagina salvato:\n{out}')
        return out

    def save_batch_folder_pdf(self):
        return self.export_all_to_pdf()

    def ask_output_destination(self, output_mode, stem_prefix="scanner_scan"):
        initialdir = self.get_initial_dir()
        if output_mode == 'PDF unico':
            out = filedialog.asksaveasfilename(
                title='Scegli nome e cartella del PDF scanner',
                defaultextension='.pdf',
                initialfile=self.get_default_save_name('.pdf', 'scan'),
                initialdir=initialdir,
                filetypes=[('PDF', '*.pdf')]
            )
            if not out:
                return None, None
            self.update_last_used_dir(out)
            return Path(out), Path(out).stem

        folder = filedialog.askdirectory(title="Scegli la cartella di salvataggio JPG", initialdir=initialdir)
        if not folder:
            return None, None
        folder_path = Path(folder)
        self.update_last_used_dir(folder_path)
        stem = self.get_default_save_name('', 'scan')
        return folder_path, stem

    def scan_from_scanner(self):
        if not self.confirm_save_before_switch("scanner"):
            return
        ok, err = import_wia_modules()
        if not ok:
            messagebox.showwarning(
                "Scanner",
                "Per usare la scansione diretta serve Windows con WIA disponibile.\n\n"
                "Installa pywin32 e usa uno scanner con driver WIA.\n\n"
                f"Dettaglio: {err}"
            )
            return

        dlg = ScannerDialog(self.root)
        self.root.wait_window(dlg)
        cfg = dlg.result
        if not cfg:
            return

        threading.Thread(
            target=self._scan_worker,
            args=(cfg, None, None),
            daemon=True
        ).start()

    def _scan_worker(self, cfg, dest, stem):
        try:
            raw_pages = []
            final_pages = []
            use_feeder = cfg["source"] == "Caricatore fogli automatico" or cfg["scan_mode"] == "Più pagine (caricatore automatico)"
            advanced = cfg.get("method") == "Avanzata con interfaccia scanner"

            self.root.after(0, lambda: self.show_scan_progress("Scansione in corso..."))

            if advanced:
                img = scan_via_wia_common_dialog()
                raw_pages = [img]
            else:
                if cfg["scan_mode"] == "Pagina singola":
                    raw_pages = [scan_page_wia(cfg["device"]["id"], cfg["dpi"], cfg["color"], use_feeder)]
                elif cfg["scan_mode"] == "Più pagine (manuale)":
                    idx = 1
                    while True:
                        self.root.after(0, lambda i=idx: self.show_scan_progress(f"Scansione pagina {i} in corso..."))
                        raw_img = scan_page_wia(cfg["device"]["id"], cfg["dpi"], cfg["color"], False)
                        raw_pages.append(raw_img)
                        ask_more = self.ui_call_and_wait(lambda i=idx: messagebox.askyesno('Scanner', f'Pagina {i} acquisita.\n\nVuoi scansionare una nuova pagina?'))
                        idx += 1
                        if not ask_more:
                            break
                else:
                    idx = 1
                    while True:
                        try:
                            raw_pages.append(scan_page_wia(cfg["device"]["id"], cfg["dpi"], cfg["color"], True))
                            idx += 1
                            if idx > 200:
                                break
                        except Exception:
                            break
                    if not raw_pages:
                        raise RuntimeError("Nessuna pagina acquisita dal caricatore automatico. Controlla driver e caricamento fogli.")

            if not raw_pages:
                self.root.after(0, lambda: self.hide_scan_progress("Nessuna pagina acquisita"))
                return

            for idx, img in enumerate(raw_pages, start=1):
                self.root.after(0, lambda i=idx, t=len(raw_pages): self.set_scan_progress_value(i / max(1, t), f"Elaborazione pagina {i}/{len(raw_pages)}..."))
                final_pages.append(self.maybe_process_scanned(img, cfg["process"], cfg.get("color_policy", "Segui modalità principale dell'app"), cfg.get("color", "Color")))

            def inject_pages():
                self.clear_all_documents()
                for i, raw_img in enumerate(raw_pages, start=1):
                    corners = detect_document_corners_scanner_safe(raw_img)
                    self.add_document(raw_img, f'scanner pagina {i}', None, 'scanner', corners, dpi=cfg.get('dpi'))
                    if self.documents:
                        self.documents[-1]['processed'] = final_pages[i-1].copy()
                if self.documents:
                    self.current_index = 0
                    self.sync_from_document()
                    self.refresh_doc_thumbnails()
                    self.set_status('Scansione completata. Ora puoi rivedere miniature, ordine, pagina da rifare o esportare in PDF.')
                self.hide_scan_progress('Scansione completata')
                messagebox.showinfo('Scanner', 'Scansione completata. Le pagine sono state caricate nella colonna miniature a sinistra. Ora puoi riordinarle, modificarle ed esportare il PDF unico.')
            self.root.after(0, inject_pages)

        except Exception as e:
            self.root.after(0, lambda: self.hide_scan_progress(f"Errore scanner: {e}"))
            self.root.after(0, lambda: messagebox.showerror("Scanner", f"Errore durante la scansione:\n\n{e}"))

    def _open_review_dialog(self, pages, ask_continue_on_confirm=False, page_number=None):
        return {'images': list(pages), 'action': 'finish'}


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