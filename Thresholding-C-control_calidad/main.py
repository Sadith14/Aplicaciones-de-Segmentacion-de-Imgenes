"""Ejecuta Otsu vs Adaptativo sobre imagenes/ y guarda collages + tabla CSV.
Uso: python main.py [carpeta] [--claro]   (usa params.json si existe)
      (--claro si los defectos son mas claros que el fondo)"""
import cv2, csv, glob, json, os, sys, time
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipeline import procesar, iou_dice, recall_cajas

args = [a for a in sys.argv[1:] if not a.startswith("--")]
carpeta = args[0] if args else "imagenes"
params = json.load(open("params.json")) if os.path.exists("params.json") else {}
params.pop("metodo", None)
oscuro = params.pop("defecto_oscuro", True) and "--claro" not in sys.argv


def cajas_xml(ruta_img):
    """Busca el XML VOC (NEU-DET) en .../annotations/ y devuelve las cajas."""
    base = os.path.splitext(os.path.basename(ruta_img))[0]
    raiz = os.path.dirname(os.path.dirname(ruta_img))
    for cand in (os.path.join(raiz, "annotations", base + ".xml"),
                 os.path.join(os.path.dirname(raiz), "annotations", base + ".xml"),
                 os.path.splitext(ruta_img)[0] + ".xml"):
        if os.path.exists(cand):
            cajas = []
            for o in ET.parse(cand).getroot().iter("bndbox"):
                v = [int(float(o.find(t).text)) for t in ("xmin", "ymin", "xmax", "ymax")]
                cajas.append(tuple(v))
            return cajas
    return []

os.makedirs("resultados", exist_ok=True)
filas = []
rutas = sorted(sum([glob.glob(os.path.join(carpeta, "**", e), recursive=True)
                    for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp")], []))
rutas = [r for r in rutas if "masks" not in r.replace("\\", "/").split("/")]
for ruta in rutas:
    nombre = os.path.basename(ruta)
    img = cv2.imread(ruta)
    gt_ruta = os.path.join(carpeta, "masks", os.path.splitext(nombre)[0] + ".png")
    cajas = cajas_xml(ruta)
    gt = cv2.imread(gt_ruta, 0) if os.path.exists(gt_ruta) else None
    res = {}
    for m in ("otsu", "adaptativo"):
        t0 = time.perf_counter()
        res[m] = procesar(img, m, defecto_oscuro=oscuro, **params)
        res[m]["ms"] = (time.perf_counter() - t0) * 1000
    fila = {"imagen": nombre}
    for m in res:
        fila[f"{m}_veredicto"] = res[m]["veredicto"]
        fila[f"{m}_n"] = len(res[m]["defectos"])
        fila[f"{m}_area"] = res[m]["area"]
        fila[f"{m}_ms"] = round(res[m]["ms"], 2)
        if gt is not None:
            fila[f"{m}_iou"], fila[f"{m}_dice"] = [round(v, 3) for v in iou_dice(res[m]["mask"], gt)]
        if cajas:
            fila[f"{m}_recall_cajas"] = round(recall_cajas(res[m]["mask"], cajas), 3)
    fila["real"] = "DEFECTUOSA" if (gt is not None and gt.any()) or cajas else "OK"
    filas.append(fila)

    if len(filas) > 40: continue
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.4))
    paneles = [("Original", img[..., ::-1], None), ("Mascara real", gt, "gray"),
               ("Otsu", res["otsu"]["mask"], "gray"), ("Adaptativo", res["adaptativo"]["mask"], "gray"),
               ("Resultado (adapt.)", res["adaptativo"]["salida"][..., ::-1], None)]
    for a, (t, im, cm) in zip(ax, paneles):
        if im is None: im = np.zeros_like(img[..., 0])
        a.imshow(im, cmap=cm); a.set_title(t, fontsize=9); a.axis("off")
    plt.tight_layout(); plt.savefig(f"resultados/{os.path.splitext(nombre)[0]}_collage.png", dpi=90); plt.close()

with open("resultados/tabla_comparativa.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=filas[0].keys()); w.writeheader(); w.writerows(filas)

# Resumen
print(f"{len(filas)} imagenes procesadas | params: {params or 'por defecto'}")
print(f"{'metodo':11s} {'IoU':>7s} {'Dice':>7s} {'recall cajas':>13s} {'veredicto ok':>13s} {'ms medio':>9s}")
for m in ("otsu", "adaptativo"):
    def media(k):
        v = [f[f"{m}_{k}"] for f in filas if f"{m}_{k}" in f and f["real"] == "DEFECTUOSA"]
        return f"{np.mean(v):.3f}" if v else "  -"
    ac = sum(f[f"{m}_veredicto"] == f["real"] for f in filas)
    ms = np.mean([f[f"{m}_ms"] for f in filas])
    print(f"{m:11s} {media('iou'):>7s} {media('dice'):>7s} {media('recall_cajas'):>13s} {ac:>6d}/{len(filas):<6d} {ms:9.2f}")
