"""Funciones del pipeline de control de calidad con thresholding (OpenCV)."""
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np


def preprocesar(img, k=5):
    """Escala de grises + filtro Gaussiano para reducir ruido."""
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    return cv2.GaussianBlur(gris, (k, k), 0)


def corregir_iluminacion(gris, sigma=25):
    """Divide por el fondo estimado (blur grande): el fondo queda ~128 y
    se eliminan gradientes de iluminacion."""
    fondo = cv2.GaussianBlur(gris, (0, 0), sigma)
    return cv2.divide(gris, fondo, scale=128)


def umbral_otsu(gris, defecto_oscuro=True):
    """Umbral GLOBAL de Otsu (maximiza la varianza entre clases)."""
    tipo = cv2.THRESH_BINARY_INV if defecto_oscuro else cv2.THRESH_BINARY
    t, mask = cv2.threshold(gris, 0, 255, tipo + cv2.THRESH_OTSU)
    return mask, t


def umbral_adaptativo(gris, bloque=35, C=8, defecto_oscuro=True):
    """Umbral LOCAL: T(x,y) = media gaussiana del vecindario - C.
    OpenCV siempre calcula T = media - C, sin importar el tipo (BINARY o
    BINARY_INV). Para detectar defectos CLAROS (mas brillantes que su
    entorno) hace falta comparar contra T = media + C, lo que se logra
    pasando C en negativo junto con THRESH_BINARY."""
    if defecto_oscuro:
        tipo, c_ajustado = cv2.THRESH_BINARY_INV, C
    else:
        tipo, c_ajustado = cv2.THRESH_BINARY, -C
    return cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 tipo, bloque, c_ajustado)


def morfologia(mask, k_open=3, k_close=7):
    """Apertura (quita ruido) y cierre (une grietas cortadas)."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)))
    return mask


def detectar_defectos(mask, area_min=40):
    """Componentes conexas; descarta las menores a area_min (falsos positivos)."""
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    defectos, limpia = [], np.zeros_like(mask)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= area_min:
            defectos.append((x, y, w, h, int(area)))
            limpia[etiquetas == i] = 255
    return defectos, limpia


def dibujar(img, defectos, veredicto):
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y, w, h, _ in defectos:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
    color = (0, 160, 0) if veredicto == "OK" else (0, 0, 255)
    cv2.putText(out, veredicto, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return out


def procesar(img, metodo, defecto_oscuro=True, area_min=40, area_veredicto=60,
             bloque=35, C=8, sigma=25, k_close=7):
    """Ejecuta el pipeline completo con 'otsu' o 'adaptativo'."""
    bloque = max(3, bloque | 1)  # debe ser impar
    gris = preprocesar(img)
    corr = corregir_iluminacion(gris, sigma=max(1, sigma))
    if metodo == "otsu":
        mask, _ = umbral_otsu(corr, defecto_oscuro)
    else:
        mask = umbral_adaptativo(corr, bloque=bloque, C=C, defecto_oscuro=defecto_oscuro)
    mask = morfologia(mask, k_close=max(1, k_close))
    defectos, limpia = detectar_defectos(mask, area_min)
    area_total = sum(d[4] for d in defectos)
    veredicto = "DEFECTUOSA" if area_total >= area_veredicto else "OK"
    return {"mask": limpia, "defectos": defectos, "area": area_total,
            "veredicto": veredicto, "salida": dibujar(img, defectos, veredicto)}


def recall_cajas(mask, cajas, min_frac=0.05):
    """Para datasets con cajas (NEU-DET): fraccion de cajas reales que
    contienen al menos min_frac de pixeles detectados."""
    if not cajas:
        return None
    ok = 0
    for x1, y1, x2, y2 in cajas:
        roi = mask[y1:y2, x1:x2]
        if roi.size and (roi > 0).mean() >= min_frac:
            ok += 1
    return ok / len(cajas)


def iou_dice(pred, gt):
    p, g = pred > 0, gt > 0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    iou = inter / union if union else 1.0
    dice = 2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) else 1.0
    return float(iou), float(dice)


def cajas_xml(ruta_img):
    """Busca el XML VOC (NEU-DET) en .../annotations/ y devuelve las cajas."""
    base = os.path.splitext(os.path.basename(ruta_img))[0]
    raiz = os.path.dirname(os.path.dirname(ruta_img))
    for cand in (os.path.join(raiz, "annotations", base + ".xml"),
                 os.path.join(os.path.dirname(raiz), "annotations", base + ".xml"),
                 os.path.splitext(ruta_img)[0] + ".xml"):
        if os.path.exists(cand):
            return [tuple(int(float(o.find(t).text)) for t in ("xmin", "ymin", "xmax", "ymax"))
                    for o in ET.parse(cand).getroot().iter("bndbox")]
    return []
