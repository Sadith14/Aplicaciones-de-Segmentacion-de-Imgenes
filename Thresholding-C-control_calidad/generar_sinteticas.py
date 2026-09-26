"""Genera superficies metalicas sinteticas con grietas, rayaduras y manchas
(+ mascara de referencia) para probar el pipeline. Reemplazable por NEU / MVTec AD."""
import cv2, numpy as np, os
rng = np.random.default_rng(7)
H = W = 256
os.makedirs("imagenes/masks", exist_ok=True)

def base():
    img = np.full((H, W), 150, np.float32)
    img += cv2.GaussianBlur(rng.normal(0, 14, (H, W)).astype(np.float32), (0, 0), 1.2)
    img += rng.normal(0, 4, (H, W))
    # gradiente de iluminacion
    gx = np.linspace(-35, 35, W)[None, :]; gy = np.linspace(-15, 15, H)[:, None]
    return img + gx + gy

def grieta(mask, img):
    x, y = rng.integers(30, W - 30), rng.integers(30, H - 30)
    ang = rng.uniform(0, np.pi)
    pts = [(x, y)]
    for _ in range(rng.integers(15, 30)):
        ang += rng.normal(0, 0.4); x += 4 * np.cos(ang); y += 4 * np.sin(ang)
        pts.append((int(x), int(y)))
    tmp = np.zeros((H, W), np.uint8)
    cv2.polylines(tmp, [np.array(pts, np.int32)], False, 255, 2)
    img[tmp > 0] -= 70; mask[tmp > 0] = 255

def rayadura(mask, img):
    x, y = rng.integers(20, W - 80), rng.integers(20, H - 20)
    tmp = np.zeros((H, W), np.uint8)
    cv2.line(tmp, (x, y), (x + rng.integers(40, 90), y + rng.integers(-15, 15)), 255, 2)
    img[tmp > 0] -= 45; mask[tmp > 0] = 255

def mancha(mask, img):
    tmp = np.zeros((H, W), np.uint8)
    cv2.ellipse(tmp, (int(rng.integers(30, W - 30)), int(rng.integers(30, H - 30))),
                (int(rng.integers(6, 14)), int(rng.integers(5, 11))), rng.integers(0, 180), 0, 360, 255, -1)
    img[tmp > 0] -= 40; mask[tmp > 0] = 255

tipos = ["ok", "ok", "ok", "grieta", "grieta", "grieta", "rayadura", "rayadura", "mancha", "mancha", "mixto", "mixto"]
for i, t in enumerate(tipos):
    img, mask = base(), np.zeros((H, W), np.uint8)
    if t in ("grieta", "mixto"): grieta(mask, img)
    if t in ("rayadura", "mixto"): rayadura(mask, img)
    if t == "mancha": mancha(mask, img)
    img = np.clip(img, 0, 255).astype(np.uint8)
    n = f"pieza_{i:02d}_{t}.png"
    cv2.imwrite(f"imagenes/{n}", img)
    cv2.imwrite(f"imagenes/masks/{n}", mask)
print(len(tipos), "imagenes generadas")
