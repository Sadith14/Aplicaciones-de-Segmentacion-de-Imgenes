"""
Analizador de Documentos (OCR) con Canny explicable  ·  versión con tema moderno
================================================================================

Aplicación de escritorio (Tkinter) que permite:
 1) Subir una imagen de un documento (foto de una hoja, recibo, etc.)
 2) Ejecutar un pipeline de Canny IMPLEMENTADO PASO A PASO (no es una caja
    negra: se muestran las etapas clásicas del algoritmo) para encontrar
    el contorno del documento y enderezarlo (perspectiva).
 3) Ejecutar un pipeline alternativo basado SOLO en thresholding
    (Otsu / adaptativo) para el mismo objetivo.
 4) Ejecutar un pipeline COMBINADO: Canny para la geometría (encontrar y
    enderezar el documento) + thresholding para la binarización final
    que se le entrega al OCR.
 5) Comparar el texto y la confianza de OCR (Tesseract) obtenidos por
    cada pipeline.

Requisitos:
    pip install -r requirements.txt
    Tesseract-OCR instalado en el sistema (ver README.md)

Ejecutar:
    python app.py
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext

import cv2
import numpy as np
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import pytesseract
    TESSERACT_OK = True
except ImportError:
    TESSERACT_OK = False


# ============================================================
# 1. CANNY IMPLEMENTADO PASO A PASO (para que sea "explicable")
# ============================================================
# cv2.Canny() internamente hace estos mismos pasos, pero no expone los
# resultados intermedios. Aquí los recalculamos a mano para poder
# mostrarlos y explicarlos en la interfaz.

def resize_max(img, max_dim=700):
    """Reduce la imagen si es muy grande, para que la app responda rápido."""
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
    return img


def gaussian_blur(gray, ksize=5, sigma=1.4):
    return cv2.GaussianBlur(gray, (ksize, ksize), sigma)


def sobel_gradients(blurred):
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max() * 255.0
    direction = np.arctan2(gy, gx)
    return magnitude, direction


def non_max_suppression(magnitude, direction):
    """Adelgaza los bordes: en cada píxel, sólo se conserva si es el máximo
    local en la dirección del gradiente."""
    M, N = magnitude.shape
    Z = np.zeros((M, N), dtype=np.float64)
    angle = direction * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            a = angle[i, j]
            q, r = 255.0, 255.0
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q, r = magnitude[i, j + 1], magnitude[i, j - 1]
            elif 22.5 <= a < 67.5:
                q, r = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
            elif 67.5 <= a < 112.5:
                q, r = magnitude[i + 1, j], magnitude[i - 1, j]
            elif 112.5 <= a < 157.5:
                q, r = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                Z[i, j] = magnitude[i, j]
    return Z


def double_threshold(img, low_ratio=0.05, high_ratio=0.15):
    """Clasifica cada píxel como borde fuerte (255), débil (75) o no-borde (0)."""
    high_thresh = img.max() * high_ratio
    low_thresh = high_thresh * low_ratio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.uint8)
    weak, strong = np.uint8(75), np.uint8(255)

    strong_i, strong_j = np.where(img >= high_thresh)
    weak_i, weak_j = np.where((img >= low_thresh) & (img < high_thresh))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res, weak, strong


def hysteresis(img, weak, strong=255):
    """Un borde débil sobrevive sólo si está conectado a un borde fuerte."""
    M, N = img.shape
    out = img.copy()
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if out[i, j] == weak:
                neighborhood = out[i - 1:i + 2, j - 1:j + 2]
                if strong in neighborhood:
                    out[i, j] = strong
                else:
                    out[i, j] = 0
    return out


def canny_steps(gray, low_ratio=0.05, high_ratio=0.15, ksize=5, sigma=1.4):
    """Ejecuta las etapas clásicas de Canny y regresa cada resultado
    intermedio en una lista ordenada, con una explicación breve."""
    blurred = gaussian_blur(gray, ksize, sigma)
    magnitude, direction = sobel_gradients(blurred)
    nms = non_max_suppression(magnitude, direction)
    thresholded, weak, strong = double_threshold(nms, low_ratio, high_ratio)
    final = hysteresis(thresholded, weak, strong)

    steps = [
        ("1. Escala de grises", gray,
         "Canny trabaja sobre intensidad, no sobre color: se descarta el "
         "canal de color para quedarnos solo con el brillo de cada píxel."),
        ("2. Suavizado Gaussiano", blurred,
         "Se difumina la imagen para reducir ruido; si no se hiciera, el "
         "ruido generaría miles de falsos bordes en el siguiente paso."),
        ("3. Magnitud del gradiente (Sobel)", magnitude.astype(np.uint8),
         "Se calcula cuánto cambia la intensidad en cada píxel (en X e Y). "
         "Zonas muy brillantes aquí = cambios bruscos = posibles bordes."),
        ("4. Supresión de no-máximos", nms.astype(np.uint8),
         "Se adelgazan los bordes: en la dirección del gradiente, sólo se "
         "conserva el píxel que es máximo local (bordes de 1 píxel de grosor)."),
        ("5. Umbral doble", thresholded,
         "Cada píxel se clasifica como borde fuerte (blanco), débil (gris) "
         "o no-borde (negro), según dos umbrales."),
        ("6. Histéresis (Canny final)", final,
         "Un borde débil sólo se conserva si toca a un borde fuerte; así se "
         "eliminan bordes débiles aislados producidos por ruido."),
    ]
    return steps


# ============================================================
# 2. DETECCIÓN DEL CONTORNO DEL DOCUMENTO Y ENDEREZADO (warp)
# ============================================================

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]          # arriba-izquierda
    rect[2] = pts[np.argmax(s)]          # abajo-derecha
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]       # arriba-derecha
    rect[3] = pts[np.argmax(diff)]       # abajo-izquierda
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b), 1)

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b), 1)

    dst = np.array([
        [0, 0], [max_width - 1, 0],
        [max_width - 1, max_height - 1], [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def find_quad_contour(binary_img):
    """Busca, entre los 5 contornos más grandes, uno que se aproxime a un
    cuadrilátero (4 puntos): asumimos que es el documento."""
    contours, _ = cv2.findContours(binary_img.copy(), cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.05 * binary_img.size:
            return approx.reshape(4, 2)
    return None


def draw_quad_overlay(bgr, quad):
    """Dibuja, sobre la imagen ORIGINAL, el cuadrilátero de 4 puntos que
    Canny logró extraer (la silueta del documento) antes de enderezarlo.
    Esto es lo que responde a "¿qué extrajo Canny?": no solo el mapa de
    bordes, sino el contorno concreto que se usó para el warp."""
    overlay = bgr.copy()
    if quad is None:
        cv2.putText(overlay, "No se detecto un contorno de 4 puntos",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
                    cv2.LINE_AA)
        return overlay

    rect = order_points(quad.astype("float32")).astype(int)
    pts_closed = rect.reshape((-1, 1, 2))
    cv2.polylines(overlay, [pts_closed], isClosed=True,
                  color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

    labels = ["TL", "TR", "BR", "BL"]
    for (x, y), lab in zip(rect, labels):
        cv2.circle(overlay, (int(x), int(y)), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(overlay, lab, (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA)
    return overlay


# ============================================================
# 3. TRES PIPELINES COMPLETOS
# ============================================================

def pipeline_canny(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    steps = canny_steps(gray)
    edges = steps[-1][1]
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    quad = find_quad_contour(edges_dilated)
    contour_overlay = draw_quad_overlay(bgr, quad)
    warped = four_point_transform(bgr, quad) if quad is not None else bgr.copy()
    return steps, quad, warped, contour_overlay


def pipeline_threshold(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 25, 15)
    quad = find_quad_contour(otsu)
    warped = four_point_transform(bgr, quad) if quad is not None else bgr.copy()

    steps = [
        ("1. Escala de grises", gray,
         "Igual que en Canny: se trabaja sólo con intensidad."),
        ("2. Suavizado", blur,
         "Reduce ruido antes de binarizar."),
        ("3. Umbral de Otsu (global)", otsu,
         "Un único umbral, calculado automáticamente, separa toda la "
         "imagen en blanco/negro. Usado para hallar el contorno."),
        ("4. Umbral adaptativo (local)", adaptive,
         "El umbral se recalcula por regiones; compensa mejor la "
         "iluminación desigual, útil para el texto."),
    ]
    return steps, quad, warped


def pipeline_combined(bgr):
    """Canny para la GEOMETRÍA (encontrar y enderezar el documento) +
    thresholding para la BINARIZACIÓN final que recibe el OCR."""
    _, quad, warped, _ = pipeline_canny(bgr)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_bin = cv2.adaptiveThreshold(
        warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )
    return quad, warped, warped_bin


# ============================================================
# 4. OCR
# ============================================================

def run_ocr(img):
    if not TESSERACT_OK:
        return "[pytesseract no está instalado]", 0.0
    try:
        text = pytesseract.image_to_string(img, lang="spa+eng")
    except Exception:
        text = pytesseract.image_to_string(img)  # fallback sin idioma spa
    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in data["conf"] if str(c) not in ("-1", "")]
        mean_conf = sum(confs) / len(confs) if confs else 0.0
    except Exception:
        mean_conf = 0.0
    return text.strip(), mean_conf


# ============================================================
# 5. TEMA VISUAL (paleta, estilos y componentes reutilizables)
# ============================================================

FONT = "Segoe UI"
MONO = "Consolas"

# Base
BG = "#f3f4fb"          # fondo general (lavanda muy suave)
CARD = "#ffffff"
BORDER = "#e2e5f3"
INK = "#1e2340"         # texto principal
MUTED = "#6b7194"       # texto secundario
NAVY = "#141833"        # cabecera
NAVY_SOFT = "#a9b0d9"

# Un color por pipeline
VIOLET, VIOLET_D = "#6c5ce7", "#4c3fc4"     # Canny
AMBER, AMBER_D = "#f59e0b", "#b45309"       # Thresholding
EMERALD, EMERALD_D = "#10b981", "#047857"   # Combinado
BLUE = "#3b82f6"

# Semáforo de confianza
GOOD, MID, BAD = "#12a150", "#d98200", "#dc3545"

PIPE = {
    "canny":  dict(c=VIOLET,  d=VIOLET_D,  title="Canny",         name="Solo Canny"),
    "thresh": dict(c=AMBER,   d=AMBER_D,   title="Thresholding",  name="Solo Thresholding"),
    "combo":  dict(c=EMERALD, d=EMERALD_D, title="Combinado",     name="Canny + Thresholding"),
}


def apply_theme(root):
    root.configure(bg=BG)
    s = ttk.Style(root)
    s.theme_use("clam")

    s.configure(".", background=BG, foreground=INK, font=(FONT, 10))
    s.configure("TFrame", background=BG)
    s.configure("TLabel", background=BG, foreground=INK)
    s.configure("Muted.TLabel", background=BG, foreground=MUTED)

    # Pestañas
    s.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 6, 0, 0))
    s.configure("TNotebook.Tab", padding=(20, 10), font=(FONT, 10, "bold"),
                background="#e4e7f6", foreground=MUTED, borderwidth=0)
    s.map("TNotebook.Tab",
          background=[("selected", BG), ("active", "#d6dbf2")],
          foreground=[("selected", VIOLET), ("active", INK)])
    s.layout("TNotebook.Tab", [("Notebook.tab", {"sticky": "nswe", "children": [
        ("Notebook.padding", {"side": "top", "sticky": "nswe", "children": [
            ("Notebook.label", {"side": "top", "sticky": ""})]})]})])

    # Botones
    s.configure("Accent.TButton", background=VIOLET, foreground="white",
                font=(FONT, 10, "bold"), padding=(18, 9), borderwidth=0,
                focusthickness=0, focuscolor=VIOLET)
    s.map("Accent.TButton",
          background=[("disabled", "#c9c5f2"), ("active", VIOLET_D)],
          foreground=[("disabled", "#ffffff")])
    s.configure("Ghost.TButton", background=CARD, foreground=VIOLET,
                font=(FONT, 10, "bold"), padding=(16, 8), bordercolor=BORDER,
                lightcolor=CARD, darkcolor=CARD, focusthickness=0, focuscolor=CARD)
    s.map("Ghost.TButton", background=[("active", "#eeebfd")],
          bordercolor=[("active", VIOLET)])

    # Scrollbar y progreso
    s.configure("Vertical.TScrollbar", background="#cfd4ee", troughcolor=BG,
                bordercolor=BG, arrowcolor=MUTED, relief="flat")
    s.map("Vertical.TScrollbar", background=[("active", "#b5bce6")])
    s.configure("Accent.Horizontal.TProgressbar", troughcolor="#e4e7f6",
                background=VIOLET, borderwidth=0, thickness=6)


def conf_palette(conf):
    """(color de texto, color de fondo suave) según la confianza del OCR."""
    if conf >= 70:
        return GOOD, "#dcf6e6"
    if conf >= 40:
        return MID, "#fff1d6"
    return BAD, "#fde2e4"


def make_card(parent, title=None, accent=VIOLET, pad=14):
    """Tarjeta blanca con borde suave y una barrita de color en el título."""
    outer = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
    if title:
        head = tk.Frame(outer, bg=CARD)
        head.pack(fill="x")
        tk.Frame(head, bg=accent, width=5).pack(side="left", fill="y")
        tk.Label(head, text=title, bg=CARD, fg=INK, font=(FONT, 11, "bold"),
                 anchor="w").pack(side="left", padx=12, pady=10)
    body = tk.Frame(outer, bg=CARD)
    body.pack(fill="both", expand=True, padx=pad, pady=(6 if title else pad, pad))
    return outer, body


def make_badge(parent, text, fg, bg):
    return tk.Label(parent, text=text, bg=bg, fg=fg, font=(FONT, 9, "bold"),
                    padx=10, pady=3)


class ConfBar(tk.Canvas):
    """Barra horizontal que muestra la confianza del OCR (0-100)."""
    def __init__(self, parent, value, width=200, height=10):
        super().__init__(parent, width=width, height=height, bg=CARD,
                         highlightthickness=0)
        self.create_rectangle(0, 0, width, height, fill="#e9ecf7", outline="")
        fill_w = max(0.0, min(100.0, value)) / 100.0 * width
        self.create_rectangle(0, 0, fill_w, height, fill=conf_palette(value)[0], outline="")


def styled_text(parent, height=10):
    return scrolledtext.ScrolledText(
        parent, height=height, wrap="word", font=(MONO, 10),
        bg="#fafbff", fg=INK, relief="flat", borderwidth=0,
        highlightthickness=1, highlightbackground=BORDER, highlightcolor=VIOLET,
        padx=12, pady=10, insertbackground=INK,
        selectbackground="#d9d4fb", selectforeground=INK)


def new_figure(width, height):
    return Figure(figsize=(width, height), dpi=90, facecolor=CARD)


def show_img(ax, img, title, accent):
    """Dibuja una imagen (gris o BGR) con marco del color del pipeline."""
    if img.ndim == 2:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=9, color=INK, fontweight="bold", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(accent)
        sp.set_linewidth(2)


def embed_figure(fig, parent, fill=None):
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    widget = canvas.get_tk_widget()
    widget.configure(bg=CARD, highlightthickness=0)
    if fill:
        widget.pack(fill=fill, expand=True)
    else:
        widget.pack()
    return canvas


# Contenido de la pestaña de teoría, estructurado para poder estilizarlo.
THEORY = [
    ("h", "¿En qué se diferencia Canny de una solución basada en thresholding?"),
    ("bullet", AMBER, "Thresholding",
     "clasifica cada píxel según su intensidad: ¿es más claro o más oscuro que un "
     "umbral? Es sensible a la iluminación: una sombra sobre el documento puede "
     "quedar clasificada como «fondo» y arruinar el resultado."),
    ("bullet", VIOLET, "Canny",
     "no mira el nivel de gris absoluto, sino el cambio de intensidad (el gradiente) "
     "entre píxeles vecinos. Por eso detecta bien la silueta del documento incluso "
     "con iluminación despareja, y produce contornos de 1 píxel de grosor gracias a "
     "la supresión de no-máximos y la histéresis."),
    ("bullet", BLUE, "La contrapartida",
     "Canny es más sensible al ruido y no sirve por sí solo para «limpiar» el texto "
     "interno para el OCR: para eso conviene una imagen binaria (texto negro sobre "
     "fondo blanco), que es justamente lo que entrega el thresholding."),

    ("h", "¿Qué es exactamente lo que «extrae» Canny en esta app?"),
    ("step", VIOLET, "1",
     "El mapa de bordes final (paso 6, histéresis): píxeles blancos = borde."),
    ("step", VIOLET, "2",
     "A partir de ese mapa se buscan los 5 contornos más grandes y se queda con el "
     "primero que se aproxima a un cuadrilátero de 4 puntos: ese es el contorno del "
     "documento que Canny «extrajo»."),
    ("step", VIOLET, "3",
     "Con esos 4 puntos se corrige la perspectiva (warp) y se obtiene el documento "
     "enderezado, que es la extracción final lista para OCR."),
    ("p", "Los tres resultados se muestran en la pestaña «Canny»: el contorno "
          "detectado dibujado sobre la foto original y el documento ya enderezado. "
          "El mapa de bordes está en «Ver proceso paso a paso»."),

    ("h", "¿Se pueden combinar Canny y thresholding?  Sí."),
    ("step", EMERALD, "1",
     "Canny encuentra el contorno del documento (paso geométrico) y se usa para "
     "enderezar la imagen (corrección de perspectiva)."),
    ("step", EMERALD, "2",
     "Sobre el documento ya enderezado se aplica un umbral ADAPTATIVO (thresholding "
     "local) para binarizar el texto antes de pasarlo al OCR."),
    ("p", "Esta combinación suele dar mejor texto reconocido que usar cualquiera de "
          "las dos técnicas por separado, porque cada una resuelve un problema "
          "distinto: Canny corrige la geometría, thresholding mejora la legibilidad."),
]


class ScrollableFrame(ttk.Frame):
    """Frame con scroll vertical, usado para alojar figuras grandes."""
    def __init__(self, parent):
        super().__init__(parent)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg=BG)
        vbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)

        self.inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        win = canvas.create_window((0, 0), window=self.inner, anchor="nw")
        # que el contenido siempre ocupe todo el ancho disponible
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.configure(yscrollcommand=vbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)


# ============================================================
# 6. INTERFAZ GRÁFICA (Tkinter)
# ============================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Analizador de Documentos — Canny explicable + OCR")
        self.geometry("1200x840")
        self.minsize(980, 680)
        apply_theme(self)

        self.bgr_image = None
        self.image_path = None

        self._build_header()
        self._build_toolbar()
        self._build_tabs()
        self._show_placeholders()

        if not TESSERACT_OK:
            self._log_warning(
                "pytesseract no está instalado (pip install pytesseract). "
                "La app funcionará, pero sin texto OCR."
            )

    # ---------------- Cabecera ----------------
    def _build_header(self):
        header = tk.Frame(self, bg=NAVY)
        header.pack(side="top", fill="x")

        left = tk.Frame(header, bg=NAVY)
        left.pack(side="left", padx=24, pady=16)
        tk.Label(left, text="Analizador de Documentos", bg=NAVY, fg="white",
                 font=(FONT, 19, "bold")).pack(anchor="w")
        tk.Label(left, text="Canny explicable  ·  Thresholding  ·  OCR con Tesseract",
                 bg=NAVY, fg=NAVY_SOFT, font=(FONT, 10)).pack(anchor="w", pady=(2, 0))

        legend = tk.Frame(header, bg=NAVY)
        legend.pack(side="right", padx=24)
        for key in ("canny", "thresh", "combo"):
            p = PIPE[key]
            chip = tk.Frame(legend, bg=NAVY)
            chip.pack(side="left", padx=10)
            tk.Label(chip, text="●", bg=NAVY, fg=p["c"], font=(FONT, 14)).pack(side="left")
            tk.Label(chip, text=p["title"], bg=NAVY, fg="white",
                     font=(FONT, 10)).pack(side="left", padx=(4, 0))

        # franja tricolor
        strip = tk.Frame(self)
        strip.pack(side="top", fill="x")
        for i, color in enumerate((VIOLET, AMBER, EMERALD)):
            strip.columnconfigure(i, weight=1)
            tk.Frame(strip, bg=color, height=4).grid(row=0, column=i, sticky="ew")

    # ---------------- Toolbar ----------------
    def _build_toolbar(self):
        bar = ttk.Frame(self, padding=(20, 14, 20, 6))
        bar.pack(side="top", fill="x")

        ttk.Button(bar, text="📂  Subir imagen", style="Ghost.TButton",
                   command=self.on_load).pack(side="left")
        self.btn_run = ttk.Button(bar, text="▶  Ejecutar análisis", style="Accent.TButton",
                                   command=self.on_run, state="disabled")
        self.btn_run.pack(side="left", padx=10)

        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=120,
                                         style="Accent.Horizontal.TProgressbar")

        self.status_var = tk.StringVar(value="Sube una imagen de un documento para empezar.")
        ttk.Label(bar, textvariable=self.status_var, style="Muted.TLabel").pack(side="left", padx=12)

        self.thumb_label = tk.Label(bar, bg=CARD, highlightthickness=2,
                                    highlightbackground=BORDER)

    # ---------------- Tabs ----------------
    def _build_tabs(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=20, pady=(4, 16))

        self.tab_canny = ttk.Frame(self.nb)
        self.tab_thresh = ttk.Frame(self.nb)
        self.tab_combo = ttk.Frame(self.nb)
        self.tab_compare = ttk.Frame(self.nb)
        self.tab_steps = ScrollableFrame(self.nb)
        self.tab_theory = ttk.Frame(self.nb)

        self.nb.add(self.tab_canny, text="1) Canny")
        self.nb.add(self.tab_thresh, text="2) Thresholding")
        self.nb.add(self.tab_combo, text="3) Combinado + OCR")
        self.nb.add(self.tab_compare, text="4) Comparación OCR")
        self.nb.add(self.tab_steps, text="Proceso paso a paso")
        self.nb.add(self.tab_theory, text="¿Canny vs Thresholding?")

        self._build_theory_tab()

    def _build_theory_tab(self):
        wrap = ttk.Frame(self.tab_theory, padding=16)
        wrap.pack(fill="both", expand=True)
        outer, body = make_card(wrap, pad=6)
        outer.pack(fill="both", expand=True)

        txt = tk.Text(body, wrap="word", bg=CARD, fg=INK, relief="flat", borderwidth=0,
                      highlightthickness=0, padx=22, pady=14, cursor="arrow",
                      font=(FONT, 11), spacing2=3)
        sb = ttk.Scrollbar(body, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True)

        txt.tag_configure("h", font=(FONT, 14, "bold"), foreground=INK,
                          spacing1=18, spacing3=8)
        txt.tag_configure("item", lmargin1=18, lmargin2=44, spacing3=8)
        txt.tag_configure("plain", spacing1=6, spacing3=8)
        txt.tag_configure("lead", font=(FONT, 11, "bold"))

        for block in THEORY:
            kind = block[0]
            if kind == "h":
                txt.insert("end", block[1] + "\n", "h")
            elif kind == "bullet":
                _, color, lead, body_text = block
                tag = f"c{color}"
                txt.tag_configure(tag, foreground=color, font=(FONT, 11, "bold"))
                txt.insert("end", "●  ", ("item", tag))
                txt.insert("end", lead + ": ", ("item", "lead", tag))
                txt.insert("end", body_text + "\n", "item")
            elif kind == "step":
                _, color, num, body_text = block
                tag = f"c{color}"
                txt.tag_configure(tag, foreground=color, font=(FONT, 11, "bold"))
                txt.insert("end", f"{num}   ", ("item", tag))
                txt.insert("end", body_text + "\n", "item")
            else:
                txt.insert("end", block[1] + "\n", "plain")
        txt.configure(state="disabled")

    def _log_warning(self, msg):
        messagebox.showwarning("Aviso", msg)

    # ---------------- Estados vacíos ----------------
    def _placeholder(self, frame, text):
        self._clear_frame(frame)
        box = ttk.Frame(frame)
        box.pack(expand=True, pady=110)
        tk.Label(box, text="🗎", font=(FONT, 46), bg=BG, fg="#c4c9e8").pack()
        ttk.Label(box, text=text, style="Muted.TLabel",
                  font=(FONT, 11)).pack(pady=(6, 0))

    def _show_placeholders(self):
        msg = "Sube una imagen y pulsa «Ejecutar análisis» para ver los resultados."
        for f in (self.tab_canny, self.tab_thresh, self.tab_combo, self.tab_compare,
                  self.tab_steps.inner):
            self._placeholder(f, msg)

    # ---------------- Cargar imagen ----------------
    @staticmethod
    def _read_image_any_path(path):
        """cv2.imread() falla silenciosamente (devuelve None) con rutas que
        tienen tildes/ñ/espacios raros en Windows. Leer los bytes primero y
        decodificarlos con cv2.imdecode evita ese problema."""
        try:
            with open(path, "rb") as f:
                data = f.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception:
            pass
        # Último intento, por si acaso
        return cv2.imread(path)

    def on_load(self):
        path = filedialog.askopenfilename(
            title="Selecciona una imagen de documento",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not path:
            return
        bgr = self._read_image_any_path(path)
        if bgr is None:
            messagebox.showerror(
                "Error",
                "No se pudo leer esa imagen.\n\n"
                "Posibles causas:\n"
                "- El archivo está dañado o no es realmente una imagen.\n"
                "- La ruta tiene tildes/ñ/caracteres raros (prueba moverla a una "
                "carpeta simple, ej. C:\\docs\\imagen.jpg).\n"
                "- El formato no es JPG/PNG/BMP/TIFF."
            )
            return

        self.bgr_image = resize_max(bgr, 700)
        self.image_path = path
        self.status_var.set(f"Imagen cargada: {os.path.basename(path)} "
                             f"({self.bgr_image.shape[1]}x{self.bgr_image.shape[0]})")
        self.btn_run.configure(state="normal")

        # miniatura que conserva la proporción
        thumb = Image.fromarray(cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2RGB))
        thumb.thumbnail((84, 84))
        self._thumb_imgtk = ImageTk.PhotoImage(thumb)
        self.thumb_label.configure(image=self._thumb_imgtk, highlightbackground=VIOLET)
        self.thumb_label.pack(side="right")

    # ---------------- Ejecutar análisis ----------------
    def on_run(self):
        if self.bgr_image is None:
            return
        self.btn_run.configure(state="disabled")
        self.progress.pack(side="left", padx=(6, 0))
        self.progress.start(12)
        self.status_var.set("Procesando... (la supresión de no-máximos y la "
                             "histéresis son pixel-a-pixel y pueden tardar unos segundos)")
        threading.Thread(target=self._run_pipelines_thread, daemon=True).start()

    def _stop_progress(self):
        self.progress.stop()
        self.progress.pack_forget()

    def _on_error(self, msg):
        self._stop_progress()
        self.btn_run.configure(state="normal")
        self.status_var.set("Ocurrió un error durante el procesamiento.")
        messagebox.showerror("Error durante el procesamiento", msg)

    def _run_pipelines_thread(self):
        try:
            bgr = self.bgr_image.copy()

            canny_result_steps, canny_quad, canny_warped, canny_contour = pipeline_canny(bgr)
            thresh_steps, thresh_quad, thresh_warped = pipeline_threshold(bgr)
            combo_quad, combo_warped, combo_bin = pipeline_combined(bgr)

            # OCR de los tres enfoques
            text_canny, conf_canny = run_ocr(cv2.cvtColor(canny_warped, cv2.COLOR_BGR2RGB))
            thresh_warped_gray = cv2.cvtColor(thresh_warped, cv2.COLOR_BGR2GRAY)
            _, thresh_warped_bin = cv2.threshold(
                thresh_warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text_thresh, conf_thresh = run_ocr(thresh_warped_bin)
            text_combo, conf_combo = run_ocr(combo_bin)

            results = dict(
                canny_steps=canny_result_steps, canny_quad=canny_quad,
                canny_warped=canny_warped, canny_contour=canny_contour,
                thresh_steps=thresh_steps, thresh_quad=thresh_quad, thresh_warped=thresh_warped,
                combo_quad=combo_quad, combo_warped=combo_warped, combo_bin=combo_bin,
                text_canny=text_canny, conf_canny=conf_canny,
                text_thresh=text_thresh, conf_thresh=conf_thresh,
                text_combo=text_combo, conf_combo=conf_combo,
            )
            self.after(0, lambda: self._render_results(results))
        except Exception as e:
            msg = str(e)  # se copia: `e` no existe fuera del bloque except
            self.after(0, lambda: self._on_error(msg))

    # ---------------- Render ----------------
    def _clear_frame(self, frame):
        for w in frame.winfo_children():
            w.destroy()

    def _ocr_header(self, parent, text, conf, bar_width=200, compact=False):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", pady=(0, 10))
        fg, bg = conf_palette(conf)
        make_badge(row, f"Confianza {conf:.1f}%", fg, bg).pack(side="left")
        ConfBar(row, conf, width=bar_width).pack(side="left", padx=12)
        if not compact:
            tk.Label(row, text=f"{len(text or '')} caracteres detectados", bg=CARD,
                     fg=MUTED, font=(FONT, 9)).pack(side="left")
        return row

    def _render_result_tab(self, container, key, images, ocr_text, ocr_conf, subtitle):
        """Muestra SOLO el resultado final de un pipeline: las imágenes clave
        y el texto que realmente extrajo el OCR, con su confianza. El detalle
        paso a paso del algoritmo vive en la pestaña "Proceso paso a paso"."""
        p = PIPE[key]
        self._clear_frame(container)
        wrap = ttk.Frame(container, padding=16)
        wrap.pack(fill="both", expand=True)

        top = ttk.Frame(wrap)
        top.pack(fill="x", pady=(0, 10))
        tk.Label(top, text=p["title"], font=(FONT, 17, "bold"), bg=BG,
                 fg=p["c"]).pack(anchor="w")
        ttk.Label(top, text=subtitle, style="Muted.TLabel").pack(anchor="w")

        outer, body = make_card(wrap, "Imágenes clave", p["c"])
        outer.pack(fill="x")
        fig = new_figure(5.4 * len(images), 3.5)
        for i, (name, img) in enumerate(images, start=1):
            show_img(fig.add_subplot(1, len(images), i), img, name, p["c"])
        fig.tight_layout()
        embed_figure(fig, body)

        outer2, body2 = make_card(wrap, "Texto extraído por OCR", p["c"])
        outer2.pack(fill="both", expand=True, pady=(12, 0))
        self._ocr_header(body2, ocr_text, ocr_conf)
        box = styled_text(body2, height=9)
        box.pack(fill="both", expand=True)
        box.insert("1.0", ocr_text or "(sin texto detectado)")
        box.configure(state="disabled")

    def _build_step_figure(self, parent, steps, accent, title):
        """Dibuja una grilla de imágenes + su explicación técnica dentro de
        `parent`. Usado únicamente en la pestaña "Proceso paso a paso"."""
        outer, body = make_card(parent, title, accent)
        outer.pack(fill="x", padx=16, pady=(6, 10))

        n = len(steps)
        cols = 3
        rows = (n + cols - 1) // cols
        fig = new_figure(11, 3.3 * rows)
        for idx, (name, img, _expl) in enumerate(steps, start=1):
            show_img(fig.add_subplot(rows, cols, idx), img, name, accent)
        fig.tight_layout()
        embed_figure(fig, body, fill="both")

        expl = tk.Text(body, height=10, wrap="word", bg="#fafbff", fg=INK, relief="flat",
                       borderwidth=0, highlightthickness=1, highlightbackground=BORDER,
                       padx=14, pady=10, font=(FONT, 10), spacing3=4)
        expl.pack(fill="x", pady=(10, 0))
        expl.tag_configure("name", font=(FONT, 10, "bold"), foreground=accent)
        expl.tag_configure("body", foreground=INK, spacing3=10)
        for name, _, e in steps:
            expl.insert("end", name + "\n", "name")
            expl.insert("end", e + "\n", "body")
        expl.configure(state="disabled")

    def _render_steps_tab(self, r):
        host = self.tab_steps.inner
        self._clear_frame(host)

        tk.Label(host, text="Canny — proceso paso a paso", bg=BG, fg=VIOLET,
                 font=(FONT, 15, "bold")).pack(anchor="w", padx=18, pady=(16, 2))
        canny_full = r["canny_steps"] + [
            ("7. Contorno extraído (4 puntos)", r["canny_contour"],
             "A partir del mapa de bordes se buscan los contornos más grandes "
             "y se toma el primero que se aproxima a un cuadrilátero: esa es "
             "la extracción geométrica del documento (esquinas TL/TR/BR/BL)."),
            ("8. Documento enderezado", r["canny_warped"],
             "Con esas 4 esquinas se aplica una transformación de perspectiva "
             "(warp) que endereza el documento, listo para OCR."),
        ]
        self._build_step_figure(host, canny_full, VIOLET, "Las 8 etapas de Canny")

        tk.Label(host, text="Thresholding — proceso paso a paso", bg=BG, fg=AMBER_D,
                 font=(FONT, 15, "bold")).pack(anchor="w", padx=18, pady=(20, 2))
        thresh_full = r["thresh_steps"] + [
            ("5. Documento enderezado", r["thresh_warped"],
             "Resultado final tras localizar el contorno con Otsu y aplicar el warp."),
        ]
        self._build_step_figure(host, thresh_full, AMBER, "Las etapas de Thresholding")

    def _render_results(self, r):
        self._render_result_tab(
            self.tab_canny, "canny",
            images=[
                ("Contorno extraído (4 puntos)", r["canny_contour"]),
                ("Documento enderezado", r["canny_warped"]),
            ],
            ocr_text=r["text_canny"], ocr_conf=r["conf_canny"],
            subtitle="Bordes por gradiente → contorno de 4 puntos → perspectiva corregida.",
        )
        self._render_result_tab(
            self.tab_thresh, "thresh",
            images=[("Documento enderezado (Otsu)", r["thresh_warped"])],
            ocr_text=r["text_thresh"], ocr_conf=r["conf_thresh"],
            subtitle="Umbral global de Otsu para localizar el documento y binarizar.",
        )
        self._render_result_tab(
            self.tab_combo, "combo",
            images=[
                ("Paso 1 (Canny): documento enderezado", r["combo_warped"]),
                ("Paso 2 (Umbral adaptativo): binarizado", r["combo_bin"]),
            ],
            ocr_text=r["text_combo"], ocr_conf=r["conf_combo"],
            subtitle="Canny corrige la geometría; el thresholding adaptativo mejora la legibilidad.",
        )
        self._render_steps_tab(r)
        self._render_compare_tab(r)

        self._stop_progress()
        self.status_var.set("✔ Análisis completo.")
        self.btn_run.configure(state="normal")

    def _render_compare_tab(self, r):
        self._clear_frame(self.tab_compare)
        wrap = ttk.Frame(self.tab_compare, padding=16)
        wrap.pack(fill="both", expand=True)

        tk.Label(wrap, text="Comparación de OCR", font=(FONT, 17, "bold"),
                 bg=BG, fg=INK).pack(anchor="w")
        ttk.Label(wrap, text="Mismo documento, tres pipelines distintos → tres resultados de OCR.",
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 10))

        items = [
            ("canny", "A · Solo Canny", "Documento enderezado, sin binarizar.",
             r["text_canny"], r["conf_canny"]),
            ("thresh", "B · Solo Thresholding", "Otsu global, sin corregir perspectiva.",
             r["text_thresh"], r["conf_thresh"]),
            ("combo", "C · Combinado", "Canny (geometría) + umbral adaptativo.",
             r["text_combo"], r["conf_combo"]),
        ]
        best = max(items, key=lambda it: it[4])[0]

        grid = ttk.Frame(wrap)
        grid.pack(fill="both", expand=True)
        grid.rowconfigure(0, weight=1)
        for col, (key, title, desc, text, conf) in enumerate(items):
            grid.columnconfigure(col, weight=1, uniform="cmp")
            p = PIPE[key]
            outer, body = make_card(grid, title, p["c"])
            outer.grid(row=0, column=col, sticky="nsew", padx=6)

            if key == best and conf > 0:
                make_badge(body, "★ Mayor confianza media", p["d"],
                           {"canny": "#ece9fd", "thresh": "#fef3dc",
                            "combo": "#dcf6ec"}[key]).pack(anchor="w", pady=(0, 8))
            tk.Label(body, text=desc, bg=CARD, fg=MUTED, font=(FONT, 9),
                     wraplength=300, justify="left").pack(anchor="w", pady=(0, 8))
            self._ocr_header(body, text, conf, bar_width=110, compact=True)
            tk.Label(body, text=f"{len(text or '')} caracteres detectados", bg=CARD,
                     fg=MUTED, font=(FONT, 9)).pack(anchor="w", pady=(0, 6))
            box = styled_text(body, height=12)
            box.pack(fill="both", expand=True)
            box.insert("1.0", text or "(sin texto detectado)")
            box.configure(state="disabled")

        ttk.Label(wrap, text="Nota: la confianza media es la que reporta Tesseract; "
                             "una confianza alta no siempre significa que el texto sea correcto.",
                  style="Muted.TLabel", font=(FONT, 9)).pack(anchor="w", pady=(10, 0))


if __name__ == "__main__":
    app = App()
    app.mainloop()