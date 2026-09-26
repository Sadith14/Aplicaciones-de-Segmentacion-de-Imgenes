
import base64
import csv
import glob
import json
import os
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from pipeline import cajas_xml, procesar, recall_cajas

PANEL = 340
EXT = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
# (etiqueta, clave, min, max, defecto)
SLIDERS = [
    ("Sigma de iluminación", "sigma", 1, 80, 25),
    ("Tamaño de bloque (adaptativo)", "bloque", 3, 101, 35),
    ("C - sensibilidad (adaptativo)", "C", 0, 40, 8),
    ("Cierre morfológico", "k_close", 1, 21, 7),
    ("Área mínima de defecto (px)", "area_min", 0, 500, 40),
    ("Área para rechazar pieza (px)", "area_ver", 1, 2000, 60),
]
METODOS = {"otsu": "Otsu (global)", "adaptativo": "Adaptativo (local)", "comparar": "Comparar ambos"}


def leer(ruta):
    try:
        return cv2.imdecode(np.fromfile(ruta, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def a_tk(img, lado=PANEL):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    e = lado / max(h, w)
    interp = cv2.INTER_NEAREST if e > 1 else cv2.INTER_AREA
    img = cv2.resize(img, (max(1, int(w * e)), max(1, int(h * e))), interpolation=interp)
    ok, buf = cv2.imencode(".png", img)
    return tk.PhotoImage(data=base64.b64encode(buf.tobytes()))


class App:
    def __init__(self, root):
        self.root = root
        root.title("Control de calidad industrial - Thresholding (OpenCV)")
        self.rutas, self.idx = [], 0
        self.img, self.cajas = None, []
        self.saved = {}          # params por metodo (de params.json)
        self.fotos, self.job = [], None
        self.ultimo = None       # panel compuesto para exportar
        self._ui()
        if os.path.exists("params.json"):
            self.cargar_params("params.json", silencioso=True)
        root.bind("<Left>", lambda e: self.mover(-1))
        root.bind("<Right>", lambda e: self.mover(1))
        if len(sys.argv) > 1:
            self.abrir_ruta(sys.argv[1])

    # ------------------------------------------------------------- interfaz
    def _ui(self):
        izq = ttk.Frame(self.root, padding=10)
        izq.grid(row=0, column=0, sticky="ns")
        der = ttk.Frame(self.root, padding=10)
        der.grid(row=0, column=1, sticky="nsew")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        ttk.Label(izq, text="1. Imagen", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Button(izq, text="Abrir imagen...", command=self.abrir_imagen).pack(fill="x", pady=1)
        ttk.Button(izq, text="Abrir carpeta...", command=self.abrir_carpeta).pack(fill="x", pady=1)
        nav = ttk.Frame(izq)
        nav.pack(fill="x", pady=2)
        ttk.Button(nav, text="◀", width=4, command=lambda: self.mover(-1)).pack(side="left")
        self.lbl_idx = ttk.Label(nav, text="0 / 0", anchor="center")
        self.lbl_idx.pack(side="left", expand=True, fill="x")
        ttk.Button(nav, text="▶", width=4, command=lambda: self.mover(1)).pack(side="right")

        ttk.Label(izq, text="2. Método de umbralización", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 0))
        self.metodo = tk.StringVar(value="adaptativo")
        for k, txt in METODOS.items():
            ttk.Radiobutton(izq, text=txt, value=k, variable=self.metodo,
                            command=self.cambio_metodo).pack(anchor="w")

        ttk.Label(izq, text="3. Tipo de defecto", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 0))
        self.oscuro = tk.IntVar(value=1)
        ttk.Radiobutton(izq, text="Más oscuro que el fondo (grietas, rayas)", value=1,
                        variable=self.oscuro, command=self.programar).pack(anchor="w")
        ttk.Radiobutton(izq, text="Más claro que el fondo (brillos, óxido claro)", value=0,
                        variable=self.oscuro, command=self.programar).pack(anchor="w")

        ttk.Label(izq, text="4. Parámetros", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 0))
        self.v = {}
        for etiqueta, clave, lo, hi, d in SLIDERS:
            self.v[clave] = tk.IntVar(value=d)
            ttk.Label(izq, text=etiqueta).pack(anchor="w")
            tk.Scale(izq, from_=lo, to=hi, orient="horizontal", variable=self.v[clave],
                     length=260, command=lambda _v: self.programar()).pack(fill="x")

        fila = ttk.Frame(izq)
        fila.pack(fill="x", pady=(8, 0))
        ttk.Button(fila, text="Cargar params.json", command=self.cargar_params).pack(side="left", expand=True, fill="x")
        ttk.Button(fila, text="Guardar params", command=self.guardar_params).pack(side="left", expand=True, fill="x")
        ttk.Button(izq, text="Guardar imagen de resultado...", command=self.exportar_imagen).pack(fill="x", pady=(6, 1))
        ttk.Button(izq, text="Procesar toda la carpeta → CSV", command=self.procesar_lote).pack(fill="x", pady=1)

        # ---- derecha
        self.banner = tk.Label(der, text="Abre una imagen para empezar", font=("Segoe UI", 20, "bold"),
                               bg="#444", fg="white", pady=8)
        self.banner.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        self.tit, self.pan = [], []
        for c in range(3):
            t = ttk.Label(der, text="", font=("Segoe UI", 10, "bold"))
            t.grid(row=1, column=c)
            p = tk.Label(der, bg="#222", width=PANEL // 8, height=PANEL // 16)
            p.grid(row=2, column=c, padx=4, sticky="n")
            self.tit.append(t)
            self.pan.append(p)
        self.info = tk.Text(der, height=9, font=("Consolas", 10), state="disabled", bg="#f4f4f4")
        self.info.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))

    # ------------------------------------------------------------- carga
    def abrir_imagen(self):
        r = filedialog.askopenfilename(filetypes=[("Imágenes", " ".join(EXT)), ("Todos", "*.*")])
        if r:
            self.abrir_ruta(r)

    def abrir_carpeta(self):
        r = filedialog.askdirectory()
        if r:
            self.abrir_ruta(r)

    def abrir_ruta(self, ruta):
        if os.path.isdir(ruta):
            rs = sorted(sum([glob.glob(os.path.join(ruta, "**", e), recursive=True) for e in EXT], []))
            rs = [x for x in rs if "masks" not in x.replace("\\", "/").split("/")]
        else:
            rs = [ruta]
        if not rs:
            messagebox.showwarning("Sin imágenes", "No se encontraron imágenes en esa ruta.")
            return
        self.rutas, self.idx = rs, 0
        self.cargar_actual()

    def mover(self, d):
        if self.rutas:
            self.idx = (self.idx + d) % len(self.rutas)
            self.cargar_actual()

    def cargar_actual(self):
        ruta = self.rutas[self.idx]
        img = leer(ruta)
        if img is None:
            messagebox.showerror("Error", f"No se pudo leer:\n{ruta}")
            return
        self.img, self.cajas = img, cajas_xml(ruta)
        self.lbl_idx.config(text=f"{self.idx + 1} / {len(self.rutas)}")
        self.root.title(f"Control de calidad - {os.path.basename(ruta)}")
        self.actualizar()

    # ------------------------------------------------------------- params
    def params(self):
        return dict(bloque=self.v["bloque"].get(), C=self.v["C"].get(), sigma=self.v["sigma"].get(),
                    k_close=self.v["k_close"].get(), area_min=self.v["area_min"].get(),
                    area_veredicto=self.v["area_ver"].get())

    def aplicar_guardados(self, metodo):
        p = self.saved.get(metodo)
        if not p:
            return
        self.oscuro.set(1 if p.get("defecto_oscuro", True) else 0)
        for k in ("sigma", "bloque", "C", "k_close", "area_min"):
            if k in p:
                self.v[k].set(p[k])

    def cambio_metodo(self):
        m = self.metodo.get()
        if m != "comparar":
            self.aplicar_guardados(m)
        self.actualizar()

    def cargar_params(self, ruta=None, silencioso=False):
        if ruta is None:
            ruta = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
            if not ruta:
                return
        try:
            crudo = json.load(open(ruta))
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        if "otsu" in crudo or "adaptativo" in crudo:
            self.saved = {k: dict(crudo[k]) for k in ("otsu", "adaptativo") if k in crudo}
        else:
            plano = {k: v for k, v in crudo.items() if k != "metodo"}
            self.saved = {"otsu": dict(plano), "adaptativo": dict(plano)}
        if self.metodo.get() != "comparar":
            self.aplicar_guardados(self.metodo.get())
        if not silencioso:
            messagebox.showinfo("Parámetros", f"Parámetros cargados de {os.path.basename(ruta)}")
        self.actualizar()

    def guardar_params(self):
        m = self.metodo.get()
        if m == "comparar":
            messagebox.showinfo("Elige un método", "Selecciona Otsu o Adaptativo para guardar sus parámetros.")
            return
        p = self.params()
        self.saved[m] = {"defecto_oscuro": bool(self.oscuro.get()), "sigma": p["sigma"],
                         "k_close": p["k_close"], "area_min": p["area_min"]}
        if m == "adaptativo":
            self.saved[m].update(bloque=p["bloque"], C=p["C"])
        json.dump(self.saved, open("params.json", "w"), indent=2)
        messagebox.showinfo("Guardado", "params.json actualizado (main.py también lo usa).")

    # ------------------------------------------------------------- proceso
    def programar(self):
        if self.job:
            self.root.after_cancel(self.job)
        self.job = self.root.after(60, self.actualizar)

    def calcular(self, metodo, img=None, p=None, oscuro=None):
        p = p or self.params()
        oscuro = bool(self.oscuro.get()) if oscuro is None else oscuro
        t0 = time.perf_counter()
        r = procesar(self.img if img is None else img, metodo, defecto_oscuro=oscuro, **p)
        r["ms"] = (time.perf_counter() - t0) * 1000
        return r

    def actualizar(self):
        self.job = None
        if self.img is None:
            return
        orig = self.img.copy()
        for x1, y1, x2, y2 in self.cajas:
            cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 200, 0), 2)
        m = self.metodo.get()
        lineas = [f"Imagen: {os.path.basename(self.rutas[self.idx])}  ({self.img.shape[1]}x{self.img.shape[0]} px)"]
        if self.cajas:
            lineas.append(f"Cajas reales (verde): {len(self.cajas)}")

        if m == "comparar":
            res = {k: self.calcular(k) for k in ("otsu", "adaptativo")}
            paneles = [("Original", orig),
                       (f"Otsu: {res['otsu']['veredicto']}", res["otsu"]["salida"]),
                       (f"Adaptativo: {res['adaptativo']['veredicto']}", res["adaptativo"]["salida"])]
            nok = [k for k in res if res[k]["veredicto"] != "OK"]
            self.banner.config(text=f"Otsu: {res['otsu']['veredicto']}   |   Adaptativo: {res['adaptativo']['veredicto']}",
                               bg="#b00020" if nok else "#1b7f3b")
            ks = ("otsu", "adaptativo")
        else:
            r = self.calcular(m)
            res = {m: r}
            paneles = [("Original", orig), (f"Máscara ({METODOS[m]})", r["mask"]), ("Resultado", r["salida"])]
            self.banner.config(text="PIEZA " + ("OK" if r["veredicto"] == "OK" else "DEFECTUOSA"),
                               bg="#1b7f3b" if r["veredicto"] == "OK" else "#b00020")
            ks = (m,)

        for k in ks:
            r = res[k]
            s = (f"{METODOS[k]:20s} defectos: {len(r['defectos']):3d} | área total: {r['area']:6d} px | "
                 f"{r['ms']:.1f} ms | {r['veredicto']}")
            if self.cajas:
                rc = recall_cajas(r["mask"], self.cajas)
                s += f" | recall cajas: {rc:.0%}"
            lineas.append(s)
        self.info.config(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("end", "\n".join(lineas))
        self.info.config(state="disabled")

        self.fotos = []
        for i, (t, im) in enumerate(paneles):
            f = a_tk(im)
            self.fotos.append(f)
            self.pan[i].config(image=f, width=f.width(), height=f.height())
            self.tit[i].config(text=t)
        self.ultimo = np.hstack([cv2.cvtColor(p, cv2.COLOR_GRAY2BGR) if p.ndim == 2 else p for _, p in paneles]) \
            if len({p.shape[0] for _, p in paneles}) == 1 else None

    # ------------------------------------------------------------- exportar
    def exportar_imagen(self):
        if self.ultimo is None:
            messagebox.showinfo("Nada que guardar", "Abre una imagen primero.")
            return
        r = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")],
                                         initialfile=os.path.splitext(os.path.basename(self.rutas[self.idx]))[0] + "_resultado.png")
        if r:
            cv2.imencode(".png", self.ultimo)[1].tofile(r)

    def params_de(self, metodo):
        """Params guardados de ese metodo (si existen) o los de los sliders."""
        p = self.params()
        osc = bool(self.oscuro.get())
        g = self.saved.get(metodo)
        if g:
            osc = g.get("defecto_oscuro", osc)
            for k in ("sigma", "bloque", "C", "k_close", "area_min"):
                if k in g:
                    p[k] = g[k]
        return p, osc

    def procesar_lote(self):
        if len(self.rutas) < 1:
            messagebox.showinfo("Sin imágenes", "Abre una carpeta primero.")
            return
        destino = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="resultados_lote.csv",
                                               filetypes=[("CSV", "*.csv")])
        if not destino:
            return
        filas = []
        self.root.config(cursor="watch")
        for n, ruta in enumerate(self.rutas):
            img = leer(ruta)
            if img is None:
                continue
            cajas = cajas_xml(ruta)
            f = {"imagen": ruta, "cajas_reales": len(cajas)}
            for m in ("otsu", "adaptativo"):
                p, osc = self.params_de(m)
                r = self.calcular(m, img=img, p=p, oscuro=osc)
                f[f"{m}_veredicto"] = r["veredicto"]
                f[f"{m}_defectos"] = len(r["defectos"])
                f[f"{m}_area"] = r["area"]
                f[f"{m}_ms"] = round(r["ms"], 2)
                f[f"{m}_recall_cajas"] = round(recall_cajas(r["mask"], cajas), 3) if cajas else ""
            filas.append(f)
            if n % 20 == 0:
                self.root.update_idletasks()
        self.root.config(cursor="")
        with open(destino, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=filas[0].keys())
            w.writeheader()
            w.writerows(filas)
        resumen = [f"{len(filas)} imágenes procesadas → {os.path.basename(destino)}\n"]
        for m in ("otsu", "adaptativo"):
            nok = sum(f[f"{m}_veredicto"] != "OK" for f in filas)
            rc = [f[f"{m}_recall_cajas"] for f in filas if f[f"{m}_recall_cajas"] != ""]
            ms = np.mean([f[f"{m}_ms"] for f in filas])
            resumen.append(f"{METODOS[m]}: {nok} rechazadas, {len(filas) - nok} OK, {ms:.1f} ms/img"
                           + (f", recall cajas {np.mean(rc):.1%}" if rc else ""))
        messagebox.showinfo("Lote terminado", "\n".join(resumen))


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
