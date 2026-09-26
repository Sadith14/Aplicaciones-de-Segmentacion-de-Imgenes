"""
Ejercicio 2 - Segmentación de imágenes médicas con Congruencia de Fase + Watershed
==================================================================================
Problema: segmentar y contar núcleos celulares en microscopía de fluorescencia.
Dificultades típicas: núcleos que se tocan, brillo MUY desigual entre núcleos
(algunos casi invisibles) y fondo con iluminación no uniforme.

Detector de bordes: CONGRUENCIA DE FASE (Phase Congruency, P. Kovesi).
No mide cuánto cambia el brillo (como los detectores de gradiente), sino dónde
las componentes de frecuencia están en fase, que es lo que el ojo humano percibe
como borde. Es invariante al brillo y al contraste. No usa machine learning.

Pipeline:
  1) Congruencia de fase (filtros log-Gabor) -> mapa de bordes M en [0, 1]
  2) Watershed sobre M desde sus mínimos (zonas planas = interior de un objeto o fondo)
     -> la imagen queda dividida en cuencas separadas por las crestas de borde
  3) Cada cuenca se acepta como núcleo si es más brillante que su entorno inmediato
     (contraste LOCAL: sin umbral global, así la iluminación desigual no afecta)

Imágenes:
  - Por defecto: imagen sintética con ground truth (permite medir Dice, F1 y conteo)
  - --real: skimage.data.human_mitosis() (núcleos reales; se descarga la 1ra vez, requiere 'pooch')
  - --img ruta.png: tu propia imagen (objetos claros sobre fondo oscuro;
    usa --invertir si tus objetos son oscuros sobre fondo claro)
  - --exportar: guarda la imagen sintética y su ground truth (para correr Canny sobre la MISMA imagen)

Requisitos: opencv-python, numpy, scipy, scikit-image, matplotlib
"""
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation, color


# ---------------------------------------------------------------------------
# 1. Imagen sintética de núcleos con ground truth
# ---------------------------------------------------------------------------
def generar_nucleos(n=60, tam=512, seed=1):
    rng = np.random.default_rng(seed)
    gt = np.zeros((tam, tam), np.int32)
    img = np.zeros((tam, tam), np.float32)
    centros, radios = [], []
    intentos = 0
    while len(centros) < n and intentos < 5000:
        intentos += 1
        r = rng.uniform(10, 18)
        c = rng.uniform(r + 5, tam - r - 5, 2)
        # Permitimos que se toquen (factor 0.75) pero sin superponerse demasiado
        if all(np.hypot(*(c - c2)) > 0.75 * (r + r2) for c2, r2 in zip(centros, radios)):
            centros.append(c)
            radios.append(r)
    yy, xx = np.mgrid[0:tam, 0:tam]
    for i, (c, r) in enumerate(zip(centros, radios), start=1):
        ang = rng.uniform(0, np.pi)
        a, b = r * rng.uniform(1.0, 1.3), r * rng.uniform(0.75, 1.0)
        dx, dy = xx - c[0], yy - c[1]
        u = dx * np.cos(ang) + dy * np.sin(ang)
        v = -dx * np.sin(ang) + dy * np.cos(ang)
        dentro = (u / a) ** 2 + (v / b) ** 2 <= 1
        gt[dentro] = i
        img[dentro] = rng.uniform(0.22, 0.9)       # brillo muy variable entre núcleos
    # Textura interna (cromatina), fondo no uniforme, desenfoque óptico y ruido
    img *= 1 + 0.15 * ndi.gaussian_filter(rng.normal(0, 1, img.shape), 2)
    fondo = 0.05 + 0.45 * (xx / tam) ** 1.5         # iluminación que crece hacia la derecha
    img = ndi.gaussian_filter(img, 1.5) + fondo
    img += rng.normal(0, 0.03, img.shape)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8), gt


def quitar_pequenos(binaria, area_min):
    """Elimina componentes con área < area_min (compatible con cualquier versión de skimage)."""
    lab = measure.label(binaria)
    areas = np.bincount(lab.ravel())
    ok = areas >= area_min
    ok[0] = False
    return ok[lab]


# ---------------------------------------------------------------------------
# 2. Detector de bordes: congruencia de fase
# ---------------------------------------------------------------------------

def congruencia_fase(img, nscale=4, norient=6, min_wl=3, mult=2.1, sigma_onf=0.55, k=2.0, cutoff=0.5, g=10):
    """Mapa de bordes por congruencia de fase (Kovesi, 1999/2003; versión 'phasecong3').

    Idea: en un borde, las componentes de Fourier de la imagen están EN FASE.
    Se descompone la imagen con filtros log-Gabor (nscale escalas x norient orientaciones);
    en cada punto se mide qué tan alineadas están las fases:
        PC = energía local / suma de amplitudes   (valor 0..1, sin unidades)
    Como es un COCIENTE, no depende del brillo ni del contraste: un núcleo tenue da
    un borde tan fuerte como uno brillante. El ruido se descuenta con un umbral T
    estimado automáticamente de la escala más fina (k = nº de desviaciones estándar).

    Devuelve el "momento máximo" M de la covarianza de PC entre orientaciones:
    alto en los bordes, cercano a 0 en zonas planas.
    """
    img = img.astype(np.float64)
    rows, cols = img.shape
    F = np.fft.fft2(img)
    x = (np.arange(cols) - cols // 2) / cols
    y = (np.arange(rows) - rows // 2) / rows
    X, Y = np.meshgrid(x, y)
    radio = np.fft.ifftshift(np.hypot(X, Y)); radio[0, 0] = 1
    theta = np.fft.ifftshift(np.arctan2(-Y, X))
    sint, cost = np.sin(theta), np.cos(theta)
    # Filtro pasa-bajos para evitar frecuencias en las esquinas del espectro
    pasabajo = 1 / (1 + (radio / 0.45) ** 30)
    # Filtros log-Gabor radiales (uno por escala)
    loggabor = []
    for s in range(nscale):
        fo = 1 / (min_wl * mult ** s)
        lg = np.exp(-(np.log(radio / fo)) ** 2 / (2 * np.log(sigma_onf) ** 2)) * pasabajo
        lg[0, 0] = 0
        loggabor.append(lg)
    covx = covy = covxy = 0
    # Recorrido por orientaciones: filtro angular * log-Gabor, en el dominio de Fourier
    for o in range(norient):
        ang = o * np.pi / norient
        ds = sint * np.cos(ang) - cost * np.sin(ang)
        dc = cost * np.cos(ang) + sint * np.sin(ang)
        dtheta = np.minimum(np.abs(np.arctan2(ds, dc)) * norient / 2, np.pi)
        spread = (np.cos(dtheta) + 1) / 2
        sumE = sumO = sumAn = 0; maxAn = None; EOs = []
        for s in range(nscale):
            EO = np.fft.ifft2(F * loggabor[s] * spread)
            An = np.abs(EO)
            sumAn = sumAn + An; sumE = sumE + EO.real; sumO = sumO + EO.imag
            maxAn = An if maxAn is None else np.maximum(maxAn, An)
            if s == 0:
                tau = np.median(sumAn) / np.sqrt(np.log(4))
            EOs.append(EO)
        # Energía local: proyección de cada respuesta sobre la dirección de fase media
        XE = np.hypot(sumE, sumO) + 1e-4
        mE, mO = sumE / XE, sumO / XE
        energia = sum(EO.real * mE + EO.imag * mO - np.abs(EO.real * mO - EO.imag * mE) for EO in EOs)
        # Umbral de ruido (distribución de Rayleigh estimada en la escala más fina)
        tau_tot = tau * (1 - (1 / mult) ** nscale) / (1 - 1 / mult)
        T = tau_tot * np.sqrt(np.pi / 2) + k * tau_tot * np.sqrt((4 - np.pi) / 2)
        # Penaliza puntos donde responde una sola escala (poca dispersión de frecuencias)
        ancho = (sumAn / (maxAn + 1e-4) - 1) / (nscale - 1)
        peso = 1 / (1 + np.exp((cutoff - ancho) * g))
        PC = peso * np.maximum(energia - T, 0) / (sumAn + 1e-4)
        covx = covx + (PC * np.cos(ang)) ** 2
        covy = covy + (PC * np.sin(ang)) ** 2
        covxy = covxy + PC ** 2 * np.cos(ang) * np.sin(ang)
    covx = covx / (norient / 2); covy = covy / (norient / 2); covxy = 4 * covxy / norient
    den = np.sqrt(covxy ** 2 + (covx - covy) ** 2) + 1e-9
    return (covy + covx + den) / 2


# ---------------------------------------------------------------------------
# 3. Segmentación: watershed sobre el mapa de congruencia de fase
# ---------------------------------------------------------------------------
def segmentar(gris, suavizado=1.5, h=0.002, contraste_min=1.15, area_min=60, area_max=6000):
    g = cv2.GaussianBlur(gris, (3, 3), 0).astype(np.float64)

    # (1) Mapa de bordes invariante al contraste
    M = congruencia_fase(g)
    Ms = ndi.gaussian_filter(M, suavizado)

    # (2) Marcadores = mínimos regionales de M con profundidad >= h (h-minima):
    #     zonas sin borde. Cada una "inunda" su cuenca hasta chocar con una cresta de borde.
    marcadores = measure.label(morphology.h_minima(Ms, h))
    cuencas = segmentation.watershed(Ms, marcadores)

    # (3) Clasificar cuencas por contraste local: interior vs anillo de 4 px alrededor
    etiquetas = np.zeros_like(cuencas)
    n = 0
    for r in measure.regionprops(cuencas):
        if not (area_min <= r.area <= area_max):
            continue
        region = cuencas == r.label
        anillo = ndi.binary_dilation(region, iterations=4) & ~region
        interior = ndi.binary_erosion(region, iterations=2)
        if interior.sum() == 0:
            continue
        if g[interior].mean() > contraste_min * g[anillo].mean():
            n += 1
            etiquetas[region] = n
    return etiquetas, dict(pc=M, cuencas=cuencas)


# ---------------------------------------------------------------------------
# 4. Métricas
# ---------------------------------------------------------------------------
def dice(pred, gt):
    p, g = pred > 0, gt > 0
    return 2 * np.logical_and(p, g).sum() / (p.sum() + g.sum() + 1e-9)


def f1_objetos(pred, gt, iou_min=0.5):
    """Un núcleo cuenta como acierto si algún objeto predicho lo cubre con IoU >= 0.5."""
    tp = 0
    usados = set()
    for g in range(1, gt.max() + 1):
        mg = gt == g
        candidatos = np.unique(pred[mg])
        candidatos = candidatos[candidatos > 0]
        for p in candidatos:
            if p in usados:
                continue
            mp = pred == p
            iou = np.logical_and(mg, mp).sum() / np.logical_or(mg, mp).sum()
            if iou >= iou_min:
                tp += 1
                usados.add(p)
                break
    n_pred, n_gt = len(np.unique(pred)) - 1, gt.max()
    prec = tp / n_pred if n_pred else 0
    rec = tp / n_gt if n_gt else 0
    return 2 * prec * rec / (prec + rec + 1e-9)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", help="ruta a imagen propia")
    ap.add_argument("--real", action="store_true", help="usar skimage.data.human_mitosis()")
    ap.add_argument("--invertir", action="store_true", help="objetos oscuros sobre fondo claro")
    ap.add_argument("--h", type=float, default=0.002, help="profundidad mínima de los mínimos (h-minima)")
    ap.add_argument("--exportar", action="store_true", help="guardar imagen sintética + ground truth")
    ap.add_argument("--salida", default="ejercicio2_resultados.png")
    args = ap.parse_args()

    gt = None
    if args.img:
        gris = cv2.imread(args.img, cv2.IMREAD_GRAYSCALE)
        if gris is None:
            raise SystemExit(f"No se pudo leer {args.img}")
    elif args.real:
        from skimage import data
        gris = data.human_mitosis()
    else:
        gris, gt = generar_nucleos()
        if args.exportar:
            cv2.imwrite("nucleos_prueba.png", gris)
            cv2.imwrite("nucleos_prueba_gt.png", gt.astype(np.uint16))  # etiquetas: 0 = fondo, 1..N = núcleos
            print("Exportados: nucleos_prueba.png, nucleos_prueba_gt.png")
    if args.invertir:
        gris = 255 - gris

    etiquetas, info = segmentar(gris, h=args.h)

    print(f"{'Método':36s} {'Objetos':>8s} {'Dice':>7s} {'F1@0.5':>7s}")
    if gt is not None:
        print(f"{'Ground truth':36s} {gt.max():8d}")
        print(f"{'Congruencia de fase + watershed':36s} {etiquetas.max():8d} "
              f"{dice(etiquetas, gt):7.3f} {f1_objetos(etiquetas, gt):7.3f}")
    else:
        print(f"{'Congruencia de fase + watershed':36s} {etiquetas.max():8d}")

    # Figura
    ncol = 5 if gt is not None else 4
    fig, ax = plt.subplots(1, ncol, figsize=(5 * ncol, 5.5))
    ax[0].imshow(gris, cmap="gray"); ax[0].set_title("Imagen de entrada")
    ax[1].imshow(info["pc"], cmap="magma", vmax=np.percentile(info["pc"], 99.5))
    ax[1].set_title("Congruencia de fase (bordes)")
    ax[2].imshow(segmentation.mark_boundaries(gris, info["cuencas"], color=(1, 1, 0)))
    ax[2].set_title(f"Cuencas del watershed ({info['cuencas'].max()})")
    ax[3].imshow(color.label2rgb(etiquetas, gris, alpha=0.4, bg_label=0))
    ax[3].set_title(f"Resultado: {etiquetas.max()} núcleos")
    if gt is not None:
        ax[4].imshow(color.label2rgb(gt, gris, alpha=0.4, bg_label=0))
        ax[4].set_title(f"Ground truth: {gt.max()} núcleos")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(args.salida, dpi=80)
    print(f"Figura guardada en {args.salida}")


if __name__ == "__main__":
    main()
