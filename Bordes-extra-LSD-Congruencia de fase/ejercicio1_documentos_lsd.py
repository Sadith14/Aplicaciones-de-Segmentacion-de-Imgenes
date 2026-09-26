"""
Ejercicio 1 - OCR / Análisis de documentos con LSD (Line Segment Detector)
==========================================================================
Problema: "escanear" un documento fotografiado con el celular (fondo parecido al
papel, perspectiva inclinada y sombra) para luego pasarlo por OCR.

Detector de bordes: LSD (Grompone von Gioi et al., 2010). En vez de marcar
píxeles de borde, detecta SEGMENTOS DE RECTA validados estadísticamente
("a contrario"), casi sin parámetros. El borde de una hoja = 4 rectas.

Se comparan 3 pipelines:
  A) Solo thresholding (Otsu)           -> estilo Actividad 04
  B) Solo bordes (LSD)                  -> 4 rectas del borde -> esquinas por intersección
  C) Híbrido: LSD + umbral adaptativo   -> LSD para la GEOMETRÍA,
                                           thresholding para el CONTENIDO (texto)

Uso:
    python ejercicio1_documentos_lsd.py                  # foto sintética con ground truth
    python ejercicio1_documentos_lsd.py --img foto.jpg   # tu propia foto de un documento
    python ejercicio1_documentos_lsd.py --dedo           # un "dedo" tapa parte del borde de la hoja
    python ejercicio1_documentos_lsd.py --exportar       # guarda la imagen de prueba y su GT
                                                         # (para correr Canny sobre la MISMA imagen)

Requisitos: opencv-python (>= 4.5.1, trae LSD), numpy, matplotlib,
            (opcional) pytesseract + tesseract-ocr
"""
import argparse
import difflib

import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    import os
    import pytesseract
    # En Windows, si Tesseract no quedó en el PATH, se busca en la ruta de instalación por defecto
    _ruta_win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.name == "nt" and os.path.exists(_ruta_win):
        pytesseract.pytesseract.tesseract_cmd = _ruta_win
    pytesseract.get_tesseract_version()
    HAY_OCR, ERROR_OCR = True, ""
except ImportError:
    HAY_OCR, ERROR_OCR = False, "Falta pytesseract: python -m pip install pytesseract"
except Exception as e:
    HAY_OCR, ERROR_OCR = False, f"Tesseract no responde: {e}"

ANCHO_HOJA, ALTO_HOJA = 700, 950

TEXTO_GT = [
    "UNIVERSIDAD NACIONAL DEL ALTIPLANO",
    "CONSTANCIA DE MATRICULA 2026-II",
    "El estudiante se encuentra matriculado",
    "en la Escuela Profesional de Ingenieria",
    "de Sistemas durante el semestre vigente.",
    "Codigo: 204571   Creditos: 22",
    "Puno, 22 de setiembre de 2026",
]


# ---------------------------------------------------------------------------
# 1. Imagen de prueba sintética (con ground truth conocido)
# ---------------------------------------------------------------------------
def generar_foto_sintetica(seed=0):
    """Crea una 'foto' de un documento: hoja blanca con texto, en perspectiva,
    sobre una mesa clara con textura, con sombra y ruido de sensor."""
    rng = np.random.default_rng(seed)

    # Hoja con texto
    hoja = np.full((ALTO_HOJA, ANCHO_HOJA), 235, np.uint8)
    y = 110
    for i, linea in enumerate(TEXTO_GT):
        escala = 0.95 if i < 2 else 0.85
        grosor = 2
        cv2.putText(hoja, linea, (45, y), cv2.FONT_HERSHEY_SIMPLEX, escala, 25, grosor, cv2.LINE_AA)
        y += 110 if i == 1 else 80

    # Fondo: mesa clara con vetas (similar en brillo al papel -> difícil para Otsu)
    H, W = 1100, 1400
    xs = np.linspace(0, 1, W)
    vetas = 12 * np.sin(2 * np.pi * (xs * 9))[None, :] + rng.normal(0, 6, (H, W))
    fondo = np.clip(190 + vetas, 0, 255).astype(np.float32)

    # Perspectiva (esquinas destino = ground truth)
    src = np.float32([[0, 0], [ANCHO_HOJA, 0], [ANCHO_HOJA, ALTO_HOJA], [0, ALTO_HOJA]])
    gt = np.float32([[420, 90], [1080, 160], [1010, 1030], [330, 960]])
    M = cv2.getPerspectiveTransform(src, gt)
    hoja_w = cv2.warpPerspective(hoja, M, (W, H)).astype(np.float32)
    mascara = cv2.warpPerspective(np.ones_like(hoja) * 255, M, (W, H))
    foto = np.where(mascara > 0, hoja_w, fondo)

    # Sombra/iluminación no uniforme (gradiente diagonal + mancha oscura)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ilum = 1.0 - 0.45 * (xx / W) * (yy / H)
    ilum -= 0.25 * np.exp(-(((xx - 1000) ** 2) + ((yy - 800) ** 2)) / (2 * 220 ** 2))
    foto = foto * ilum
    foto = cv2.GaussianBlur(foto, (3, 3), 0) + rng.normal(0, 4, foto.shape)
    foto = np.clip(foto, 0, 255).astype(np.uint8)
    return cv2.cvtColor(foto, cv2.COLOR_GRAY2BGR), gt


# ---------------------------------------------------------------------------
# 2. Utilidades geométricas
# ---------------------------------------------------------------------------
def ordenar_esquinas(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def cuadrilatero_mayor(binaria, area_min_frac=0.10):
    """Busca el contorno externo más grande que se pueda aproximar a 4 lados."""
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_img = binaria.shape[0] * binaria.shape[1]
    for c in sorted(contornos, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < area_min_frac * area_img:
            break
        peri = cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(aprox) == 4:
            return ordenar_esquinas(aprox)
    return None


def cuadrilatero_desde_bordes(bordes, area_min_frac=0.10):
    cnts, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_img = bordes.shape[0] * bordes.shape[1]
    for c in sorted(cnts, key=lambda c: cv2.arcLength(c, False), reverse=True)[:5]:
        hull = cv2.convexHull(c)
        if cv2.contourArea(hull) < area_min_frac * area_img:
            continue
        peri = cv2.arcLength(hull, True)
        aprox = cv2.approxPolyDP(hull, 0.02 * peri, True)
        if len(aprox) == 4:
            return ordenar_esquinas(aprox)
    return None


def corregir_perspectiva(img, esquinas, lado_max=1600):
    """Rectifica la hoja RESPETANDO su proporción (ancho/alto medidos en la foto)
    y la escala para que el lado mayor mida ~lado_max px: Tesseract lee mejor
    letras grandes, y una imagen pequeña (p. ej. un recibo de 400 px) se agranda."""
    tl, tr, br, bl = esquinas
    ancho = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    alto = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    escala = lado_max / max(ancho, alto)
    W, H = int(round(ancho * escala)), int(round(alto * escala))
    dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    M = cv2.getPerspectiveTransform(esquinas, dst)
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC)


def iou_cuadrilateros(q1, q2, shape):
    if q1 is None or q2 is None:
        return 0.0
    m1 = np.zeros(shape[:2], np.uint8)
    m2 = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m1, q1.astype(np.int32), 1)
    cv2.fillConvexPoly(m2, q2.astype(np.int32), 1)
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------
# 3. Detector LSD y ajuste de las 4 rectas de la hoja
# ---------------------------------------------------------------------------
def detectar_segmentos(gris):
    """LSD sobre la imagen sin texto. Devuelve array (N, 4) con x1, y1, x2, y2."""
    # Cierre morfológico: borra el texto (trazos oscuros finos) para que LSD
    # no gaste segmentos en las letras.
    sin_texto = cv2.morphologyEx(gris, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    segs = lsd.detect(sin_texto)[0]
    return np.empty((0, 4), np.float32) if segs is None else segs.reshape(-1, 4)


def _recta_normal(seg):
    """Segmento -> recta en forma normal (theta, rho): x*cos(t) + y*sin(t) = rho."""
    x1, y1, x2, y2 = seg
    t = np.arctan2(x1 - x2, y2 - y1)  # ángulo de la normal
    rho = x1 * np.cos(t) + y1 * np.sin(t)
    if rho < 0:
        t, rho = t + np.pi, -rho
    return t % (2 * np.pi), rho


def _interseccion(l1, l2):
    (t1, r1), (t2, r2) = l1, l2
    A = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    return np.linalg.solve(A, [r1, r2])


def cuadrilatero_desde_lsd(segs, shape, largo_min_frac=0.05):
    """1) Se quedan solo los segmentos largos.
    2) Se agrupan por orientación en 2 familias (≈horizontales / ≈verticales de la hoja).
    3) En cada familia se toman las 2 rectas más externas (bordes opuestos de la hoja),
       uniendo segmentos colineales (así un borde cortado en pedazos cuenta como uno).
    4) Esquinas = intersecciones de las 4 rectas (funciona aunque falte un trozo de borde)."""
    h, w = shape[:2]
    largos = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    segs, largos = segs[largos > largo_min_frac * min(h, w)], largos[largos > largo_min_frac * min(h, w)]
    if len(segs) < 4:
        return None, segs

    # Orientación de cada segmento (0..pi) y agrupamiento en 2 familias con k-means sobre 2*ángulo
    ang = np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]) % np.pi
    feats = np.float32(np.c_[np.cos(2 * ang), np.sin(2 * ang)])
    _, grupos, _ = cv2.kmeans(feats, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3),
                              5, cv2.KMEANS_PP_CENTERS)
    grupos = grupos.ravel()

    centro = np.array([w / 2, h / 2])
    rectas = []
    for g in (0, 1):
        idx = np.where(grupos == g)[0]
        if len(idx) < 2:
            return None, segs
        # Distancia firmada de cada segmento al centro de la imagen, en la dirección normal media
        a_med = np.arctan2(np.sum(largos[idx] * np.sin(2 * ang[idx])), np.sum(largos[idx] * np.cos(2 * ang[idx]))) / 2
        normal = np.array([-np.sin(a_med), np.cos(a_med)])
        medios = (segs[idx, :2] + segs[idx, 2:]) / 2
        d = (medios - centro) @ normal
        # Los dos bordes opuestos: el grupo de segmentos más negativo y el más positivo
        for lado in (d < 0, d >= 0):
            sel = idx[lado]
            if len(sel) == 0:
                return None, segs
            dl = d[lado]
            extremo = sel[np.argmax(np.abs(dl))]
            # unir segmentos colineales con el extremo (misma recta ±8 px)
            colineales = sel[np.abs(dl - d[np.where(idx == extremo)[0][0]]) < 8]
            pts = np.vstack([segs[colineales, :2], segs[colineales, 2:]]).astype(np.float32)
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            rectas.append(_recta_normal((x0, y0, x0 + vx, y0 + vy)))

    esquinas = [_interseccion(rectas[i], rectas[j]) for i in (0, 1) for j in (2, 3)]
    if any(e is None for e in esquinas):
        return None, segs
    return ordenar_esquinas(np.array(esquinas)), segs


# ---------------------------------------------------------------------------
# 4. Pipelines
# ---------------------------------------------------------------------------
def pipeline_thresholding(img):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(gris, (7, 7), 0)
    _, bin_ = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    esquinas = cuadrilatero_mayor(bin_)
    detectada = esquinas is not None
    if esquinas is None:  # fallback típico: toda la imagen
        h, w = gris.shape
        esquinas = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    hoja = corregir_perspectiva(gris, esquinas)
    _, texto = cv2.threshold(hoja, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return dict(mapa=bin_, esquinas=esquinas, hoja=hoja, para_ocr=texto, detectada=detectada)


def pipeline_lsd(img):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    segs = detectar_segmentos(gris)
    esquinas, largos = cuadrilatero_desde_lsd(segs, gris.shape)
    detectada = esquinas is not None
    if esquinas is None:  # p. ej. un documento ESCANEADO: no hay fondo, no hay borde de hoja
        h, w = gris.shape
        esquinas = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    # Mapa para visualizar: todos los segmentos (gris) y los largos usados (blanco)
    mapa = np.zeros_like(gris)
    for x1, y1, x2, y2 in segs.astype(int):
        cv2.line(mapa, (x1, y1), (x2, y2), 90, 2)
    for x1, y1, x2, y2 in largos.astype(int):
        cv2.line(mapa, (x1, y1), (x2, y2), 255, 4)
    hoja = corregir_perspectiva(gris, esquinas)
    m = int(0.02 * min(hoja.shape))  # mismo recorte de margen que el híbrido (comparación justa)
    return dict(mapa=mapa, esquinas=esquinas, hoja=hoja, para_ocr=hoja[m:-m, m:-m], detectada=detectada)


def pipeline_hibrido(img):
    """LSD para encontrar la hoja + umbral ADAPTATIVO para el texto."""
    r = pipeline_lsd(img)
    hoja = r["hoja"]
    # Normalizar iluminación (divide por el fondo estimado) y luego umbral adaptativo
    suave = cv2.GaussianBlur(hoja, (5, 5), 0)          # quita ruido de sensor antes de binarizar
    fondo = cv2.medianBlur(suave, 81)
    norm = cv2.divide(suave, fondo, scale=235)
    texto = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 61, 20)
    m = int(0.02 * min(texto.shape))  # recortar un margen fino: evita que el borde de la hoja se lea como "|"
    texto[:m, :], texto[-m:, :], texto[:, :m], texto[:, -m:] = 255, 255, 255, 255
    r = dict(r)
    r["para_ocr"] = texto
    return r


# ---------------------------------------------------------------------------
# 5. OCR y métricas
# ---------------------------------------------------------------------------
def hacer_ocr(img, idioma="eng"):
    if not HAY_OCR:
        return ""
    return pytesseract.image_to_string(img, lang=idioma, config="--psm 6")


def ocr_con_confianza(img, idioma="eng"):
    """OCR que además devuelve la CONFIANZA de Tesseract (0-100) por palabra.
    Retorna: texto, confianza media (%), lista de palabras (texto, conf, x, y, w, h)."""
    if not HAY_OCR:
        return "", None, []
    d = pytesseract.image_to_data(img, lang=idioma, config="--psm 6",
                                  output_type=pytesseract.Output.DICT)
    palabras, lineas = [], {}
    for i, txt in enumerate(d["text"]):
        conf = float(d["conf"][i])
        if conf < 0 or not txt.strip():   # conf = -1 -> bloque/línea, no es palabra
            continue
        palabras.append((txt, conf, d["left"][i], d["top"][i], d["width"][i], d["height"][i]))
        clave = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        lineas.setdefault(clave, []).append(txt)
    texto = "\n".join(" ".join(v) for _, v in sorted(lineas.items()))
    conf_media = float(np.mean([p[1] for p in palabras])) if palabras else 0.0
    return texto, conf_media, palabras


def dibujar_confianza(img, palabras):
    """Recuadro por palabra: verde >= 80 %, amarillo 60-80 %, rojo < 60 %."""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img.copy()
    for _, conf, x, y, w, h in palabras:
        col = (0, 170, 0) if conf >= 80 else (230, 180, 0) if conf >= 60 else (220, 0, 0)
        cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
    return vis


def precision_texto(ocr, gt_lineas):
    limpio = lambda s: " ".join(s.upper().split())
    return difflib.SequenceMatcher(None, limpio(ocr), limpio(" ".join(gt_lineas))).ratio()


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", help="ruta a una foto de documento (opcional)")
    ap.add_argument("--salida", default="ejercicio1_resultados.png")
    ap.add_argument("--dedo", action="store_true",
                    help="tapar parte del borde inferior con un 'dedo' (prueba de bordes incompletos)")
    ap.add_argument("--exportar", action="store_true",
                    help="guardar la imagen sintética y su ground truth para comparar con Canny")
    args = ap.parse_args()

    if args.img:
        img, gt = cv2.imread(args.img), None
        if img is None:
            raise SystemExit(f"No se pudo leer {args.img}")
    else:
        img, gt = generar_foto_sintetica()
        if args.dedo:
            cv2.ellipse(img, (650, 995), (70, 110), 20, 0, 360, (120, 120, 120), -1)
        if args.exportar:
            cv2.imwrite("doc_prueba.png", img)
            np.save("doc_prueba_esquinas_gt.npy", gt)
            with open("doc_prueba_texto_gt.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(TEXTO_GT))
            print("Exportados: doc_prueba.png, doc_prueba_esquinas_gt.npy, doc_prueba_texto_gt.txt")

    resultados = {
        "A) Solo thresholding (Otsu)": pipeline_thresholding(img),
        "B) Solo bordes (LSD)": pipeline_lsd(img),
        "C) Híbrido LSD + umbral adaptativo": pipeline_hibrido(img),
    }

    print(f"OCR disponible: {HAY_OCR}")
    print(f"{'Pipeline':42s} {'IoU hoja':>9s} {'Precisión OCR':>14s} {'Confianza':>10s}")
    for nombre, r in resultados.items():
        iou = iou_cuadrilateros(r["esquinas"], gt, img.shape) if gt is not None else float("nan")
        texto, conf, _ = ocr_con_confianza(r["para_ocr"])
        r["ocr"] = texto
        acc = precision_texto(texto, TEXTO_GT) if (gt is not None and HAY_OCR) else float("nan")
        conf = conf if conf is not None else float("nan")
        r["iou"], r["acc"] = iou, acc
        print(f"{nombre:42s} {iou:9.3f} {acc:14.3f} {conf:9.1f}%")

    # Figura
    fig, ax = plt.subplots(3, 4, figsize=(20, 15))
    for fila, (nombre, r) in enumerate(resultados.items()):
        vis = img.copy()
        if gt is not None:
            cv2.polylines(vis, [gt.astype(np.int32)], True, (0, 200, 0), 4)
        cv2.polylines(vis, [r["esquinas"].astype(np.int32)], True, (0, 0, 255), 4)
        ax[fila, 0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax[fila, 0].set_title(f"{nombre}\nverde=GT, rojo=detectado  IoU={r['iou']:.2f}")
        ax[fila, 1].imshow(r["mapa"], cmap="gray")
        ax[fila, 1].set_title("Mapa usado para localizar la hoja" if fila == 0 else "LSD: segmentos (gris) y largos (blanco)")
        ax[fila, 2].imshow(r["hoja"], cmap="gray")
        ax[fila, 2].set_title("Hoja con perspectiva corregida")
        ax[fila, 3].imshow(r["para_ocr"], cmap="gray")
        ax[fila, 3].set_title(f"Entrada al OCR  (precisión={r['acc']:.2f})")
        for a in ax[fila]:
            a.axis("off")
    plt.tight_layout()
    plt.savefig(args.salida, dpi=80)
    print(f"Figura guardada en {args.salida}")

    if HAY_OCR:
        print("\n--- Texto OCR del pipeline híbrido ---")
        print(resultados["C) Híbrido LSD + umbral adaptativo"]["ocr"])


if __name__ == "__main__":
    main()
