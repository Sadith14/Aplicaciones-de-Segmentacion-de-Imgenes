"""
Segmentación de huesos en radiografías (FracAtlas) mediante THRESHOLDING
=========================================================================
Requisitos:  pip install opencv-python pillow numpy
Ejecutar:    python segmentacion_rayosx.py

Métodos de umbralización incluidos:
  1. Global fijo
  2. Otsu (1 umbral automático)
  3. Triangle
  4. Multi-Otsu (2 umbrales -> 3 clases: fondo / tejido blando / hueso)
  5. Adaptativo (media)
  6. Adaptativo (gaussiano)

Pipeline: gris -> (invertir) -> CLAHE -> suavizado gaussiano -> threshold
          -> morfología (apertura/cierre) -> filtro por área -> rellenar huecos

MÉTRICAS DE EVALUACIÓN (nuevo)
--------------------------------
Intrínsecas (no requieren máscara de referencia, se calculan siempre):
  - Separabilidad (eta): 0-1, qué tan bien el umbral separa dos clases de
    intensidad. Es la misma medida que usa Otsu internamente para elegir
    su umbral; aquí se reporta para CUALQUIER método, como indicador de
    calidad estadística del corte usado.
  - Solidez (compacidad): area_region / area_envolvente_convexa. Detecta
    máscaras "deshilachadas" o con bordes muy irregulares (valor cercano
    a 1 = forma compacta y sólida; valores bajos = fragmentado/ruidoso).

Contra máscara de referencia (ground truth), OPCIONAL:
  Si cargas una máscara binaria real (botón "Cargar máscara GT"), se
  calculan además:
  - IoU (Intersection over Union)
  - Dice (F1 de segmentación)
  - Precisión (de lo marcado como hueso, cuánto era realmente hueso)
  - Sensibilidad / Recall (de todo el hueso real, cuánto se detectó)
"""
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

METODOS = [
    "Global fijo",
    "Otsu",
    "Triangle",
    "Multi-Otsu (3 clases)",
    "Otsu en 2 etapas (cuerpo→hueso)",
    "Adaptativo (media)",
    "Adaptativo (gaussiano)",
]
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ----------------------------------------------------------------------------
#  NÚCLEO DE PROCESAMIENTO (independiente de la interfaz)
# ----------------------------------------------------------------------------
def multi_otsu_2(gray):
    """Otsu con 2 umbrales (3 clases) maximizando la varianza entre clases."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    P = np.cumsum(p)                      # probabilidad acumulada
    S = np.cumsum(p * np.arange(256))     # media acumulada
    mu_t = S[-1]
    best, best_t = -1.0, (85, 170)
    for t1 in range(1, 254):
        t2 = np.arange(t1 + 1, 255)
        w0 = P[t1]
        w1 = P[t2] - P[t1]
        w2 = 1.0 - P[t2]
        valid = (w0 > 1e-9) & (w1 > 1e-9) & (w2 > 1e-9)
        if not valid.any():
            continue
        m0 = S[t1] / w0
        m1 = (S[t2] - S[t1]) / np.where(w1 > 1e-9, w1, 1)
        m2 = (mu_t - S[t2]) / np.where(w2 > 1e-9, w2, 1)
        sigma_b = w0 * m0**2 + w1 * m1**2 + w2 * m2**2
        sigma_b = np.where(valid, sigma_b, -1)
        i = int(np.argmax(sigma_b))
        if sigma_b[i] > best:
            best, best_t = sigma_b[i], (t1, int(t2[i]))
    return best_t


def normalizar_iluminacion(g, sigma):
    """Resta el fondo de baja frecuencia (grosor de tejido) para que el hueso
    destaque localmente. Se calcula en baja resolución para ser rápido."""
    f = g.astype(np.float32)
    h, w = f.shape
    s = 4
    small = cv2.resize(f, (max(1, w // s), max(1, h // s)), interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(small, (0, 0), max(1.0, sigma / s))
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(f - bg + 110, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
#  MÉTRICAS
# ----------------------------------------------------------------------------
def separabilidad_otsu(g, umbral):
    """Eta = sigma_between / sigma_total, la medida de separabilidad de Otsu.
    Va de 0 (las dos clases son indistinguibles) a 1 (separación perfecta).
    Se calcula para CUALQUIER umbral dado (no solo el elegido por Otsu),
    así sirve para comparar la calidad estadística del corte entre métodos.
    Para métodos con 2 umbrales se usa el último (el que define hueso)."""
    if umbral is None:
        return None
    t = int(round(umbral))
    t = max(0, min(254, t))
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    niveles = np.arange(256)
    mu_t = np.sum(p * niveles)
    sigma_t = np.sum(p * (niveles - mu_t) ** 2)
    if sigma_t <= 1e-9:
        return 0.0
    w0 = p[:t + 1].sum()
    w1 = 1.0 - w0
    if w0 <= 1e-9 or w1 <= 1e-9:
        return 0.0
    m0 = np.sum(p[:t + 1] * niveles[:t + 1]) / w0
    m1 = np.sum(p[t + 1:] * niveles[t + 1:]) / w1
    sigma_b = w0 * w1 * (m0 - m1) ** 2
    return float(np.clip(sigma_b / sigma_t, 0.0, 1.0))


def solidez_mascara(mask):
    """Solidez = area_region / area_envolvente_convexa, promediada sobre las
    regiones (ponderada por área). 1.0 = forma compacta/convexa (ideal para
    huesos, que son estructuras alargadas pero sin bordes deshilachados);
    valores bajos indican una máscara ruidosa, fragmentada o con muchas
    protuberancias espurias."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    area_tot, sol_pond = 0.0, 0.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1:
            continue
        hull = cv2.convexHull(c)
        area_hull = cv2.contourArea(hull)
        if area_hull < 1:
            continue
        sol = area / area_hull
        sol_pond += sol * area
        area_tot += area
    if area_tot == 0:
        return None
    return float(sol_pond / area_tot)


def metricas_gt(mask_pred, mask_gt):
    """Compara la máscara predicha contra una máscara de referencia (ambas
    binarias, mismo tamaño) y devuelve IoU, Dice, precisión y sensibilidad."""
    p = mask_pred > 0
    g = mask_gt > 0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    iou = inter / union if union > 0 else (1.0 if p.sum() == g.sum() == 0 else 0.0)
    dice = (2 * inter) / (p.sum() + g.sum()) if (p.sum() + g.sum()) > 0 else 1.0
    precision = inter / p.sum() if p.sum() > 0 else (1.0 if g.sum() == 0 else 0.0)
    recall = inter / g.sum() if g.sum() > 0 else (1.0 if p.sum() == 0 else 0.0)
    return dict(iou=float(iou), dice=float(dice), precision=float(precision), recall=float(recall))


def procesar(gray, prm, gt=None):
    """Devuelve dict con imágenes intermedias, máscara y métricas.
    gt: máscara binaria de referencia opcional (mismo tamaño que gray)."""
    g = 255 - gray if prm["invertir"] else gray.copy()

    # 0) Corrección de iluminación / grosor de tejido
    if prm["norm"] > 0:
        g = normalizar_iluminacion(g, prm["norm"])

    # 1) Realce de contraste local (CLAHE)
    if prm["clahe"] > 0:
        g = cv2.createCLAHE(clipLimit=prm["clahe"], tileGridSize=(8, 8)).apply(g)

    # 2) Suavizado (reduce ruido antes de umbralizar)
    k = prm["blur"]
    if k > 1:
        k = k if k % 2 == 1 else k + 1
        g = cv2.GaussianBlur(g, (k, k), 0)

    # 3) Thresholding
    metodo = prm["metodo"]
    umbrales = []
    if metodo == "Global fijo":
        t, m = cv2.threshold(g, prm["umbral"], 255, cv2.THRESH_BINARY)
        umbrales = [t]
    elif metodo == "Otsu":
        t, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        umbrales = [t]
    elif metodo == "Triangle":
        t, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        umbrales = [t]
    elif metodo.startswith("Otsu en 2"):
        t1, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cuerpo = g[g > t1]                       # solo píxeles del cuerpo
        if cuerpo.size > 0:
            t2, _ = cv2.threshold(cuerpo.reshape(-1, 1), 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            t2 = t1
        m = np.where(g > t2, 255, 0).astype(np.uint8)
        umbrales = [t1, t2]
    elif metodo.startswith("Multi-Otsu"):
        t1, t2 = multi_otsu_2(g)
        m = np.where(g > t2, 255, 0).astype(np.uint8)   # clase 3 = hueso
        umbrales = [t1, t2]
    else:
        bs = prm["bloque"]
        bs = bs if bs % 2 == 1 else bs + 1
        tipo = (cv2.ADAPTIVE_THRESH_MEAN_C if "media" in metodo
                else cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        m = cv2.adaptiveThreshold(g, 255, tipo, cv2.THRESH_BINARY, max(bs, 3), prm["C"])

    # --- Separabilidad ANTES de morfología/filtros: mide la calidad pura del
    #     corte de intensidad (no la limpieza posterior).
    umbral_ref = umbrales[-1] if umbrales else None
    eta = separabilidad_otsu(g, umbral_ref)

    # 4) Post-procesado morfológico
    if prm["apertura"] > 0:
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (prm["apertura"] * 2 + 1,) * 2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ke)
    if prm["cierre"] > 0:
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (prm["cierre"] * 2 + 1,) * 2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ke)

    # 5) Eliminar regiones pequeñas (área mínima y/o relativa al objeto mayor)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        lim = max(prm["area_min"], (0.10 * areas.max()) if prm["quitar_peq"] else 0)
        keep = np.where(areas >= lim)[0] + 1
        m = (np.isin(lab, keep) * 255).astype(np.uint8)

    # 6) Rellenar huecos (canal medular, ruido interno)
    if prm["rellenar"]:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        m = np.zeros_like(m)
        cv2.drawContours(m, cnts, -1, 255, thickness=cv2.FILLED)

    # Superposición
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    rojo = np.zeros_like(color)
    rojo[:] = (0, 0, 255)
    over = np.where(m[..., None] > 0, cv2.addWeighted(color, 0.6, rojo, 0.4, 0), color)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, cnts, -1, (0, 255, 0), 2)

    n_comp = len(cnts)
    pct = 100.0 * np.count_nonzero(m) / m.size
    solidez = solidez_mascara(m)

    metricas = dict(area_pct=pct, n_regiones=n_comp, separabilidad=eta, solidez=solidez)
    if gt is not None and gt.shape == m.shape:
        metricas.update(metricas_gt(m, gt))

    return dict(pre=g, mask=m, overlay=over, umbrales=umbrales, pct=pct, n=n_comp,
                metricas=metricas)


def dibujar_histograma(g, umbrales, w=300, h=150):
    hist = cv2.calcHist([g], [0], None, [256], [0, 256]).ravel()
    hist = np.sqrt(hist)                      # compresión para ver colas
    hist = hist / (hist.max() + 1e-9)
    img = np.full((h, w, 3), 30, np.uint8)
    for x in range(256):
        xx = int(x * (w - 1) / 255)
        cv2.line(img, (xx, h - 1), (xx, h - 1 - int(hist[x] * (h - 12))), (200, 200, 200), 1)
    for t in umbrales:
        xx = int(t * (w - 1) / 255)
        cv2.line(img, (xx, 0), (xx, h), (0, 80, 255), 2)
        cv2.putText(img, str(int(t)), (min(xx + 3, w - 28), 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    return img


def ajustar(img, size):
    """Escala manteniendo proporción y centra en un lienzo size x size."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    s = min(size[0] / w, size[1] / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    lienzo = np.full((size[1], size[0], 3), 20, np.uint8)
    y0, x0 = (size[1] - nh) // 2, (size[0] - nw) // 2
    lienzo[y0:y0 + nh, x0:x0 + nw] = r
    return ImageTk.PhotoImage(Image.fromarray(lienzo))


# ----------------------------------------------------------------------------
#  INTERFAZ
# ----------------------------------------------------------------------------
COMPARAR = [  # (método interno, nombre corto en ASCII para cv2.putText)
    ("Global fijo", "Global fijo"),
    ("Otsu", "Otsu (1 umbral)"),
    ("Triangle", "Triangle"),
    ("Multi-Otsu (3 clases)", "Multi-Otsu (2 umbrales)"),
    ("Otsu en 2 etapas (cuerpo→hueso)", "Otsu 2 etapas"),
    ("Adaptativo (gaussiano)", "Adaptativo gaussiano"),
]


def comparar_metodos(gray, prm, gt=None, tam=300, cols=3):
    """Compone una figura con la misma imagen segmentada por varios métodos,
    mostrando en cada tile el umbral, el área y las métricas de calidad."""
    tiles = []
    for metodo, nombre in COMPARAR:
        r = procesar(gray, {**prm, "metodo": metodo}, gt=gt)
        h, w = r["overlay"].shape[:2]
        e = min(tam / w, tam / h)
        nw, nh = int(w * e), int(h * e)
        im = cv2.resize(r["overlay"], (nw, nh), interpolation=cv2.INTER_AREA)
        alto_extra = 46 + (18 if gt is not None else 0)
        lienzo = np.full((tam + alto_extra, tam, 3), 25, np.uint8)
        y0, x0 = alto_extra + (tam - nh) // 2, (tam - nw) // 2
        lienzo[y0:y0 + nh, x0:x0 + nw] = im
        umb = "/".join(str(int(t)) for t in r["umbrales"]) or "local"
        met = r["metricas"]
        eta_txt = f"{met['separabilidad']:.2f}" if met["separabilidad"] is not None else "n/a"
        sol_txt = f"{met['solidez']:.2f}" if met["solidez"] is not None else "n/a"
        cv2.putText(lienzo, nombre, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(lienzo, f"umbral:{umb} area:{r['pct']:.1f}% eta:{eta_txt} sol:{sol_txt}",
                    (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
        if gt is not None:
            cv2.putText(lienzo, f"IoU:{met['iou']:.2f} Dice:{met['dice']:.2f} "
                                 f"Prec:{met['precision']:.2f} Rec:{met['recall']:.2f}",
                        (6, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 255, 120), 1, cv2.LINE_AA)
        tiles.append(lienzo)
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 25))
    filas = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(filas)


class App:
    PANEL = (330, 330)

    def __init__(self, root):
        self.root = root
        root.title("Segmentación de rayos X por thresholding – OpenCV")
        self.gray = None
        self.res = None
        self.gt = None            # máscara de referencia (opcional), np.uint8 0/255
        self.gt_path = None
        self.archivos, self.idx = [], -1
        self._job = None
        self._construir()

    def _construir(self):
        ctrl = ttk.Frame(self.root, padding=8)
        ctrl.grid(row=0, column=0, sticky="n")
        ctrl2 = ttk.Frame(self.root, padding=8)      # columna de sliders
        ctrl2.grid(row=0, column=1, sticky="n")
        vis = ttk.Frame(self.root, padding=8)
        vis.grid(row=0, column=2, sticky="n")

        # Botones de archivo
        f = ttk.Frame(ctrl)
        f.pack(fill="x")
        ttk.Button(f, text="Abrir imagen", command=self.abrir_imagen).grid(row=0, column=0, sticky="ew")
        ttk.Button(f, text="Abrir carpeta", command=self.abrir_carpeta).grid(row=0, column=1, sticky="ew")
        ttk.Button(f, text="◀ Anterior", command=lambda: self.mover(-1)).grid(row=1, column=0, sticky="ew")
        ttk.Button(f, text="Siguiente ▶", command=lambda: self.mover(1)).grid(row=1, column=1, sticky="ew")
        f.columnconfigure((0, 1), weight=1)
        self.lbl_arch = ttk.Label(ctrl, text="(sin imagen)", wraplength=280)
        self.lbl_arch.pack(anchor="w", pady=(4, 8))

        # Máscara de referencia (ground truth), opcional
        fgt = ttk.Frame(ctrl)
        fgt.pack(fill="x", pady=(0, 8))
        ttk.Button(fgt, text="Cargar máscara GT (opcional)", command=self.cargar_gt).grid(row=0, column=0, sticky="ew")
        ttk.Button(fgt, text="Quitar GT", command=self.quitar_gt).grid(row=0, column=1, sticky="ew")
        fgt.columnconfigure((0, 1), weight=1)
        self.lbl_gt = ttk.Label(ctrl, text="Sin máscara de referencia cargada", wraplength=280, foreground="#888")
        self.lbl_gt.pack(anchor="w", pady=(0, 8))

        # Método
        ttk.Label(ctrl, text="Método de thresholding").pack(anchor="w")
        self.metodo = tk.StringVar(value=METODOS[4])
        cb = ttk.Combobox(ctrl, textvariable=self.metodo, values=METODOS, state="readonly")
        cb.pack(fill="x")
        cb.bind("<<ComboboxSelected>>", lambda e: self.actualizar())

        # Sliders
        self.vars = {}
        self._slider(ctrl2, "umbral", "Umbral (solo global fijo)", 0, 255, 127)
        self._slider(ctrl2, "norm", "Normalizar iluminación (sigma, 0 = off)", 0, 250, 0, res=5)
        self._slider(ctrl2, "clahe", "CLAHE clipLimit (0 = off)", 0, 10, 2, res=0.5)
        self._slider(ctrl2, "blur", "Suavizado gaussiano (kernel)", 1, 21, 5)
        self._slider(ctrl2, "bloque", "Adaptativo: tamaño de bloque", 3, 201, 51)
        self._slider(ctrl2, "C", "Adaptativo: constante C", -30, 30, -5)
        self._slider(ctrl2, "apertura", "Morfología: apertura (radio)", 0, 15, 2)
        self._slider(ctrl2, "cierre", "Morfología: cierre (radio)", 0, 30, 7)
        self._slider(ctrl2, "area_min", "Área mínima (px)", 0, 20000, 500, res=50)

        self.invertir = tk.BooleanVar(value=False)
        self.rellenar = tk.BooleanVar(value=True)
        self.quitar_peq = tk.BooleanVar(value=True)
        for txt, v in (("Invertir imagen (hueso oscuro)", self.invertir),
                       ("Rellenar huecos", self.rellenar),
                       ("Quitar objetos pequeños (marcadores)", self.quitar_peq)):
            ttk.Checkbutton(ctrl, text=txt, variable=v, command=self.actualizar).pack(anchor="w")

        ttk.Label(ctrl, text="Histograma (líneas rojas = umbrales)").pack(anchor="w", pady=(8, 0))
        self.lbl_hist = ttk.Label(ctrl)
        self.lbl_hist.pack()
        self.lbl_info = ttk.Label(ctrl, text="", justify="left")
        self.lbl_info.pack(anchor="w", pady=4)
        ttk.Button(ctrl, text="Guardar máscara y superposición", command=self.guardar).pack(fill="x")
        ttk.Button(ctrl, text="Comparar métodos (figura)", command=self.comparar).pack(fill="x", pady=(4, 0))

        # Paneles de imagen 2x2
        self.paneles = {}
        for i, (clave, titulo) in enumerate([("orig", "Original"), ("pre", "Preprocesada (CLAHE + blur)"),
                                             ("mask", "Máscara binaria"), ("over", "Superposición")]):
            fr = ttk.LabelFrame(vis, text=titulo)
            fr.grid(row=i // 2, column=i % 2, padx=3, pady=3)
            lb = ttk.Label(fr)
            lb.pack()
            self.paneles[clave] = lb

    def _slider(self, parent, clave, texto, a, b, ini, res=1):
        v = tk.DoubleVar(value=ini)
        self.vars[clave] = v
        tk.Scale(parent, from_=a, to=b, resolution=res, orient="horizontal", variable=v,
                 label=texto, length=280,
                 command=lambda _=None: self.actualizar_diferido()).pack(fill="x")

    # ---- archivos
    def abrir_imagen(self):
        p = filedialog.askopenfilename(title="Selecciona una radiografía",
                                       filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if p:
            self.archivos, self.idx = [p], 0
            self.cargar(p)

    def abrir_carpeta(self):
        d = filedialog.askdirectory(title="Carpeta con radiografías (p. ej. FracAtlas/images)")
        if not d:
            return
        self.archivos = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(d)
                               for f in fs if f.lower().endswith(EXT))
        if not self.archivos:
            messagebox.showwarning("Sin imágenes", "No se encontraron imágenes en esa carpeta.")
            return
        self.idx = 0
        self.cargar(self.archivos[0])

    def mover(self, d):
        if self.archivos:
            self.idx = (self.idx + d) % len(self.archivos)
            self.cargar(self.archivos[self.idx])

    def cargar(self, ruta):
        # imdecode + fromfile: funciona con rutas con tildes/espacios en Windows
        img = cv2.imdecode(np.fromfile(ruta, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("Error", f"No se pudo leer:\n{ruta}")
            return
        h0, w0 = img.shape
        esc = 1024 / max(h0, w0)          # se procesa a máx. 1024 px (más rápido)
        if esc < 1:
            img = cv2.resize(img, (int(w0 * esc), int(h0 * esc)), interpolation=cv2.INTER_AREA)
        self.gray = img
        # Al cambiar de imagen, la GT cargada (si había) ya no corresponde: se descarta.
        if self.gt is not None:
            self.quitar_gt(aviso=False)
        self.lbl_arch.config(text=f"[{self.idx + 1}/{len(self.archivos)}] {os.path.basename(ruta)}"
                                  f"\noriginal {w0}x{h0} px (procesada a {img.shape[1]}x{img.shape[0]})")
        self.actualizar()

    # ---- ground truth
    def cargar_gt(self):
        if self.gray is None:
            messagebox.showinfo("Máscara GT", "Primero carga una radiografía.")
            return
        p = filedialog.askopenfilename(title="Selecciona la máscara de referencia (binaria)",
                                       filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if not p:
            return
        m = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE)
        if m is None:
            messagebox.showerror("Error", f"No se pudo leer la máscara:\n{p}")
            return
        # Redimensionar exactamente al tamaño con el que se procesa la radiografía actual
        m = cv2.resize(m, (self.gray.shape[1], self.gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)  # binarizar por si viene en grises
        self.gt = m
        self.gt_path = p
        self.lbl_gt.config(text=f"GT: {os.path.basename(p)}", foreground="#0a0")
        self.actualizar()

    def quitar_gt(self, aviso=True):
        self.gt = None
        self.gt_path = None
        self.lbl_gt.config(text="Sin máscara de referencia cargada", foreground="#888")
        if aviso and self.gray is not None:
            self.actualizar()

    # ---- procesamiento
    def actualizar_diferido(self):
        if self._job:
            self.root.after_cancel(self._job)
        self._job = self.root.after(120, self.actualizar)

    def parametros(self):
        g = lambda k: self.vars[k].get()
        return dict(metodo=self.metodo.get(), umbral=int(g("umbral")), clahe=float(g("clahe")),
                    blur=int(g("blur")), bloque=int(g("bloque")), C=int(g("C")),
                    apertura=int(g("apertura")), cierre=int(g("cierre")),
                    area_min=int(g("area_min")), invertir=self.invertir.get(),
                    rellenar=self.rellenar.get(), norm=int(g("norm")),
                    quitar_peq=self.quitar_peq.get())

    def actualizar(self):
        if self.gray is None:
            return
        self.res = procesar(self.gray, self.parametros(), gt=self.gt)
        r = self.res
        self._fotos = {
            "orig": ajustar(self.gray, self.PANEL), "pre": ajustar(r["pre"], self.PANEL),
            "mask": ajustar(r["mask"], self.PANEL), "over": ajustar(r["overlay"], self.PANEL)}
        for k, foto in self._fotos.items():
            self.paneles[k].config(image=foto)
        self._hist = ImageTk.PhotoImage(Image.fromarray(
            cv2.cvtColor(dibujar_histograma(r["pre"], r["umbrales"]), cv2.COLOR_BGR2RGB)))
        self.lbl_hist.config(image=self._hist)
        umb = ", ".join(str(int(t)) for t in r["umbrales"]) or "local (adaptativo)"
        met = r["metricas"]
        eta_txt = f"{met['separabilidad']:.3f}" if met["separabilidad"] is not None else "n/a (umbral local)"
        sol_txt = f"{met['solidez']:.3f}" if met["solidez"] is not None else "n/a (sin regiones)"
        texto = (f"Umbral(es): {umb}\nÁrea segmentada: {r['pct']:.2f} %\n"
                 f"Regiones: {r['n']}\n"
                 f"Separabilidad (eta): {eta_txt}\n"
                 f"Solidez: {sol_txt}")
        if "iou" in met:
            texto += (f"\n--- vs. máscara GT ---\n"
                      f"IoU: {met['iou']:.3f}  Dice: {met['dice']:.3f}\n"
                      f"Precisión: {met['precision']:.3f}  Recall: {met['recall']:.3f}")
        self.lbl_info.config(text=texto)

    def comparar(self):
        if self.gray is None:
            messagebox.showinfo("Comparar", "Primero carga una imagen.")
            return
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            self.fig = comparar_metodos(self.gray, self.parametros(), gt=self.gt)
        finally:
            self.root.config(cursor="")
        win = tk.Toplevel(self.root)
        win.title("Comparación de métodos de thresholding (mismos parámetros de preprocesado)")
        rgb = cv2.cvtColor(self.fig, cv2.COLOR_BGR2RGB)
        win._foto = ImageTk.PhotoImage(Image.fromarray(rgb))
        ttk.Label(win, image=win._foto).pack()

        def guardar_fig():
            ruta = filedialog.asksaveasfilename(parent=win, defaultextension=".png",
                                                initialfile="comparacion_metodos.png",
                                                filetypes=[("PNG", "*.png")])
            if ruta:
                cv2.imencode(".png", self.fig)[1].tofile(ruta)
                messagebox.showinfo("Guardado", ruta, parent=win)
        ttk.Button(win, text="Guardar figura para el informe", command=guardar_fig).pack(pady=6)

    def guardar(self):
        if self.res is None:
            return
        base = filedialog.asksaveasfilename(defaultextension=".png", initialfile="resultado.png",
                                            filetypes=[("PNG", "*.png")])
        if not base:
            return
        raiz, ext = os.path.splitext(base)
        cv2.imencode(".png", self.res["mask"])[1].tofile(raiz + "_mascara.png")
        cv2.imencode(".png", self.res["overlay"])[1].tofile(raiz + "_superposicion.png")
        messagebox.showinfo("Guardado", f"Se guardaron:\n{raiz}_mascara.png\n{raiz}_superposicion.png")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
