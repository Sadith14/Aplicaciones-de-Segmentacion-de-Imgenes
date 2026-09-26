"""Ajuste interactivo de parametros con trackbars.
Uso:  python tuning.py <carpeta_de_imagenes>
Teclas:  n = siguiente imagen | p = anterior | s = guardar params.json | q = salir"""
import cv2, glob, json, os, sys
import numpy as np
from pipeline import procesar

carpeta = sys.argv[1] if len(sys.argv) > 1 else "imagenes"
rutas = sorted(sum([glob.glob(os.path.join(carpeta, "**", e), recursive=True)
                    for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp")], []))
rutas = [r for r in rutas if "masks" not in r.replace("\\", "/").split("/")]
if not rutas:
    sys.exit(f"No se encontraron imagenes en {carpeta}")

print(f"{len(rutas)} imagenes encontradas en {carpeta}")
print("Version OpenCV:", cv2.__version__)
W = "Tuning (n/p cambiar imagen, s guardar, q salir)"
try:
    cv2.namedWindow(W, cv2.WINDOW_NORMAL)
except cv2.error:
    sys.exit("\nERROR: tu OpenCV no soporta ventanas (opencv-python-headless).\n"
             "Solucion:\n  pip uninstall -y opencv-python-headless opencv-python\n"
             "  pip install opencv-python")
nop = lambda v: None
cv2.createTrackbar("metodo 0=Otsu 1=Adapt", W, 1, 1, nop)
cv2.createTrackbar("defecto 0=oscuro 1=claro", W, 0, 1, nop)
cv2.createTrackbar("bloque (impar)", W, 35, 101, nop)
cv2.createTrackbar("C", W, 8, 40, nop)
cv2.createTrackbar("sigma ilumin.", W, 25, 80, nop)
cv2.createTrackbar("k cierre", W, 7, 21, nop)
cv2.createTrackbar("area minima", W, 40, 500, nop)

i = 0
fallos = 0
while True:
    img = cv2.imread(rutas[i])
    if img is None:
        fallos += 1
        print("No se pudo leer:", rutas[i])
        if fallos >= len(rutas):
            sys.exit("No se pudo leer ninguna imagen (revisa la ruta).")
        i = (i + 1) % len(rutas); continue
    g = lambda n: cv2.getTrackbarPos(n, W)
    p = dict(metodo="otsu" if g("metodo 0=Otsu 1=Adapt") == 0 else "adaptativo",
             defecto_oscuro=g("defecto 0=oscuro 1=claro") == 0,
             bloque=g("bloque (impar)"), C=g("C"), sigma=g("sigma ilumin."),
             k_close=g("k cierre"), area_min=g("area minima"))
    r = procesar(img, **p)
    mask3 = cv2.cvtColor(r["mask"], cv2.COLOR_GRAY2BGR)
    vista = np.hstack([img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), mask3, r["salida"]])
    esc = max(1, 500 // vista.shape[0])
    vista = cv2.resize(vista, None, fx=esc, fy=esc, interpolation=cv2.INTER_NEAREST) if esc > 1 else vista
    cv2.putText(vista, f"{os.path.basename(rutas[i])} [{i+1}/{len(rutas)}] defectos:{len(r['defectos'])}",
                (5, vista.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.imshow(W, vista)
    k = cv2.waitKey(30) & 0xFF
    if k == ord("q"): break
    elif k == ord("n"): i = (i + 1) % len(rutas)
    elif k == ord("p"): i = (i - 1) % len(rutas)
    elif k == ord("s"):
        json.dump(p, open("params.json", "w"), indent=2)
        print("Guardado params.json:", p)
cv2.destroyAllWindows()
