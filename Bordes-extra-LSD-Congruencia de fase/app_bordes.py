
import csv
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from skimage import color, segmentation

import ejercicio1_documentos_lsd as ej1
import ejercicio2_segmentacion_fase as ej2

EXTENSIONES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
PRUEBA = "<imagen de prueba sintética>"


def leer_imagen(ruta, gris=False):
    """cv2.imread no abre rutas con tildes/ñ en Windows; imdecode sí."""
    datos = np.fromfile(ruta, dtype=np.uint8)
    return cv2.imdecode(datos, cv2.IMREAD_GRAYSCALE if gris else cv2.IMREAD_COLOR)


def listar_carpeta(carpeta):
    rutas = []
    for nombre in sorted(os.listdir(carpeta)):
        base = os.path.splitext(nombre)[0]
        # no listar los archivos de ground truth como imágenes a procesar
        if nombre.lower().endswith(EXTENSIONES) and not base.endswith("_gt"):
            rutas.append(os.path.join(carpeta, nombre))
    return rutas


# ---------------------------------------------------------------------------
# Procesamiento (independiente de la interfaz)
# ---------------------------------------------------------------------------
def procesar_ej1(ruta, idioma):
    if ruta == PRUEBA:
        img, gt = ej1.generar_foto_sintetica()
        texto_gt = ej1.TEXTO_GT
    else:
        img = leer_imagen(ruta)
        if img is None:
            raise ValueError(f"No se pudo leer {ruta}")
        base = os.path.splitext(ruta)[0]
        gt = np.load(base + "_esquinas_gt.npy") if os.path.exists(base + "_esquinas_gt.npy") else None
        texto_gt = None
        if os.path.exists(base + "_texto_gt.txt"):
            with open(base + "_texto_gt.txt", encoding="utf-8") as f:
                texto_gt = f.read().splitlines()

    resultados = {
        "A) Otsu": ej1.pipeline_thresholding(img),
        "B) LSD": ej1.pipeline_lsd(img),
        "C) LSD + umbral adaptativo": ej1.pipeline_hibrido(img),
    }
    for r in resultados.values():
        r["ocr"], r["conf"], r["palabras"] = ej1.ocr_con_confianza(r["para_ocr"], idioma)
        r["iou"] = ej1.iou_cuadrilateros(r["esquinas"], gt, img.shape) if gt is not None else None
        r["acc"] = ej1.precision_texto(r["ocr"], texto_gt) if (texto_gt and ej1.HAY_OCR) else None
    return dict(img=img, gt=gt, resultados=resultados)


def dibujar_ej1(fig, datos):
    fig.clear()
    img, gt = datos["img"], datos["gt"]
    axs = fig.subplots(3, 3)
    for fila, (nombre, r) in enumerate(datos["resultados"].items()):
        vis = img.copy()
        grosor = max(2, img.shape[1] // 300)
        if gt is not None:
            cv2.polylines(vis, [gt.astype(np.int32)], True, (0, 200, 0), grosor)
        cv2.polylines(vis, [r["esquinas"].astype(np.int32)], True, (0, 0, 255), grosor)
        extra = f"  IoU={r['iou']:.2f}" if r["iou"] is not None else ""
        if not r.get("detectada", True):
            extra += "\n(sin borde de hoja → imagen completa)"
        axs[fila, 0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axs[fila, 0].set_title(f"{nombre}{extra}", fontsize=9)
        axs[fila, 1].imshow(r["mapa"], cmap="gray")
        axs[fila, 1].set_title("Mapa para localizar la hoja", fontsize=9)
        if not r.get("detectada", True):
            axs[fila, 1].text(0.5, 0.5, "No se encontró\nel borde de la hoja\n(¿imagen escaneada?)",
                              color="white", ha="center", va="center", transform=axs[fila, 1].transAxes)
        acc = f"  precisión={r['acc']:.2f}" if r["acc"] is not None else ""
        conf = f"  confianza={r['conf']:.0f}%" if r["conf"] is not None else ""
        axs[fila, 2].imshow(ej1.dibujar_confianza(r["para_ocr"], r["palabras"]))
        axs[fila, 2].set_title(f"OCR{conf}{acc}", fontsize=9)
        for a in axs[fila]:
            a.axis("off")
    fig.tight_layout()


def texto_ej1(datos):
    lineas = []
    for nombre, r in datos["resultados"].items():
        m = []
        if r["conf"] is not None:
            dudosas = [p[0] for p in r["palabras"] if p[1] < 60]
            m.append(f"confianza media = {r['conf']:.1f}%  ({len(r['palabras'])} palabras, "
                     f"{len(dudosas)} dudosas <60%)")
        if r["iou"] is not None:
            m.append(f"IoU hoja = {r['iou']:.3f}")
        if r["acc"] is not None:
            m.append(f"precisión OCR = {r['acc']:.3f}")
        if not r.get("detectada", True):
            m.append("[no se detectó el borde de la hoja: se usó la imagen completa]")
        lineas.append(f"== {nombre}   {'   '.join(m)}")
        lineas.append(r["ocr"].strip() if ej1.HAY_OCR else "(Tesseract no instalado: sin OCR)")
        dudosas = [f"{p[0]} ({p[1]:.0f}%)" for p in r.get("palabras", []) if p[1] < 60]
        if dudosas:
            lineas.append("Palabras dudosas: " + ", ".join(dudosas[:15]) + (" ..." if len(dudosas) > 15 else ""))
        lineas.append("")
    return "\n".join(lineas)


def procesar_ej2(ruta, h, invertir, contraste=1.15):
    if ruta == PRUEBA:
        gris, gt = ej2.generar_nucleos()
    else:
        gris = leer_imagen(ruta, gris=True)
        if gris is None:
            raise ValueError(f"No se pudo leer {ruta}")
        ruta_gt = os.path.splitext(ruta)[0] + "_gt.png"
        gt = cv2.imread(ruta_gt, cv2.IMREAD_UNCHANGED).astype(np.int32) if os.path.exists(ruta_gt) else None
    if invertir:
        gris = 255 - gris
    etiquetas, info = ej2.segmentar(gris, h=h, contraste_min=contraste)
    met = dict(objetos=int(etiquetas.max()))
    if gt is not None:
        met.update(gt=int(gt.max()), dice=ej2.dice(etiquetas, gt), f1=ej2.f1_objetos(etiquetas, gt))
    return dict(gris=gris, gt=gt, etiquetas=etiquetas, info=info, met=met)


def dibujar_ej2(fig, d):
    fig.clear()
    n = 5 if d["gt"] is not None else 4
    axs = fig.subplots(1, n) if n == 4 else fig.subplots(2, 3).ravel()
    axs[0].imshow(d["gris"], cmap="gray"); axs[0].set_title("Entrada", fontsize=9)
    pc = d["info"]["pc"]
    axs[1].imshow(pc, cmap="magma", vmax=np.percentile(pc, 99.5)); axs[1].set_title("Congruencia de fase", fontsize=9)
    axs[2].imshow(segmentation.mark_boundaries(d["gris"], d["info"]["cuencas"], color=(1, 1, 0)))
    axs[2].set_title("Cuencas del watershed", fontsize=9)
    axs[3].imshow(color.label2rgb(d["etiquetas"], d["gris"], alpha=0.4, bg_label=0))
    axs[3].set_title(f"Resultado: {d['met']['objetos']} objetos", fontsize=9)
    if d["gt"] is not None:
        axs[4].imshow(color.label2rgb(d["gt"], d["gris"], alpha=0.4, bg_label=0))
        axs[4].set_title(f"Ground truth: {d['met']['gt']}", fontsize=9)
    for a in axs:
        a.axis("off")
    fig.tight_layout()


def texto_ej2(d):
    m = d["met"]
    s = f"Objetos detectados: {m['objetos']}"
    if "gt" in m:
        s += f"\nGround truth: {m['gt']}\nDice: {m['dice']:.3f}\nF1@0.5: {m['f1']:.3f}"
    return s


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
class Pestana(ttk.Frame):
    """Panel genérico: lista de imágenes a la izquierda, figura a la derecha, texto abajo."""

    def __init__(self, master, app, titulo):
        super().__init__(master)
        self.app = app
        self.rutas = []
        self.ultimo = None

        izq = ttk.Frame(self, padding=6)
        izq.pack(side="left", fill="y")
        ttk.Label(izq, text=titulo, font=("", 10, "bold"), wraplength=230).pack(anchor="w", pady=(0, 6))
        ttk.Button(izq, text="Cargar imagen(es)...", command=self.cargar_imagenes).pack(fill="x")
        ttk.Button(izq, text="Cargar carpeta...", command=self.cargar_carpeta).pack(fill="x", pady=2)
        ttk.Button(izq, text="Usar imagen de prueba", command=self.usar_prueba).pack(fill="x")
        ttk.Button(izq, text="Limpiar lista", command=self.limpiar).pack(fill="x", pady=(2, 6))

        marco = ttk.Frame(izq)
        marco.pack(fill="both", expand=True)
        self.lista = tk.Listbox(marco, width=34, height=14, exportselection=False)
        sb = ttk.Scrollbar(marco, command=self.lista.yview)
        self.lista.config(yscrollcommand=sb.set)
        self.lista.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.lista.bind("<<ListboxSelect>>", lambda e: self.procesar_seleccion())

        self.opciones = ttk.LabelFrame(izq, text="Opciones", padding=6)
        self.opciones.pack(fill="x", pady=6)

        ttk.Button(izq, text="Procesar seleccionada", command=self.procesar_seleccion).pack(fill="x")
        ttk.Button(izq, text="Procesar toda la lista → CSV", command=self.procesar_lote).pack(fill="x", pady=2)
        ttk.Button(izq, text="Guardar figura actual...", command=self.guardar_figura).pack(fill="x")

        der = ttk.Frame(self)
        der.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(9, 6.5), dpi=90)
        self.canvas = FigureCanvasTkAgg(self.fig, master=der)
        barra = NavigationToolbar2Tk(self.canvas, der, pack_toolbar=False)  # zoom / mover
        barra.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        barra.pack(fill="x")
        self.texto = tk.Text(der, height=9, wrap="word", font=("Consolas", 9))
        self.texto.pack(fill="x")

    # --- carga de archivos ---
    def _agregar(self, rutas):
        for r in rutas:
            if r not in self.rutas:
                self.rutas.append(r)
                self.lista.insert("end", "★ Prueba sintética" if r == PRUEBA else os.path.basename(r))
        if self.rutas and not self.lista.curselection():
            self.lista.selection_set(0)
            self.procesar_seleccion()

    def cargar_imagenes(self):
        rutas = filedialog.askopenfilenames(
            title="Elegir imágenes", filetypes=[("Imágenes", " ".join("*" + e for e in EXTENSIONES)), ("Todos", "*.*")])
        self._agregar(list(rutas))

    def cargar_carpeta(self):
        carpeta = filedialog.askdirectory(title="Elegir carpeta con imágenes")
        if carpeta:
            rutas = listar_carpeta(carpeta)
            if not rutas:
                messagebox.showinfo("Carpeta vacía", "No se encontraron imágenes en esa carpeta.")
            self._agregar(rutas)

    def usar_prueba(self):
        self._agregar([PRUEBA])
        idx = self.rutas.index(PRUEBA)
        self.lista.selection_clear(0, "end")
        self.lista.selection_set(idx)
        self.procesar_seleccion()

    def limpiar(self):
        self.rutas.clear()
        self.lista.delete(0, "end")
        self.fig.clear(); self.canvas.draw()
        self.texto.delete("1.0", "end")

    # --- procesamiento en segundo plano (la ventana no se congela) ---
    # Tkinter no es thread-safe: el hilo de trabajo NO toca la interfaz, solo deja
    # mensajes en una cola que el hilo principal revisa cada 100 ms.
    def _en_hilo(self, trabajo, al_terminar):
        self.app.ocupado(True)
        self.cola = queue.Queue()

        def correr():
            try:
                self.cola.put(("ok", trabajo()))
            except Exception as e:
                self.cola.put(("error", str(e)))
        threading.Thread(target=correr, daemon=True).start()
        self._revisar_cola(al_terminar)

    def _revisar_cola(self, al_terminar):
        try:
            while True:
                tipo, valor = self.cola.get_nowait()
                if tipo == "estado":
                    self.app.estado(valor)
                    continue
                self.app.ocupado(False)
                if tipo == "ok":
                    al_terminar(valor)
                else:
                    self.app.estado("Error")
                    messagebox.showerror("Error", valor)
                return
        except queue.Empty:
            self.after(100, lambda: self._revisar_cola(al_terminar))

    def procesar_seleccion(self):
        sel = self.lista.curselection()
        if not sel or self.app.esta_ocupado:
            return
        ruta = self.rutas[sel[0]]
        self.app.estado(f"Procesando {os.path.basename(ruta) if ruta != PRUEBA else 'imagen de prueba'}...")
        params = self.parametros()  # leer los controles en el hilo principal
        self._en_hilo(lambda: self.procesar(ruta, params), self.mostrar)

    def mostrar(self, datos):
        self.ultimo = datos
        self.dibujar(self.fig, datos)
        self.canvas.draw()
        self.texto.delete("1.0", "end")
        self.texto.insert("1.0", self.resumen(datos))
        self.app.estado("Listo")

    def guardar_figura(self):
        if self.ultimo is None:
            return
        ruta = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if ruta:
            self.fig.savefig(ruta, dpi=120)
            self.app.estado(f"Figura guardada en {ruta}")

    def procesar_lote(self):
        if not self.rutas or self.app.esta_ocupado:
            return
        carpeta = filedialog.askdirectory(title="Carpeta donde guardar CSV y figuras")
        if not carpeta:
            return
        rutas = list(self.rutas)
        params = self.parametros()

        def trabajo():
            filas = []
            fig = Figure(figsize=(12, 8), dpi=90)
            for i, ruta in enumerate(rutas, 1):
                nombre = "prueba_sintetica" if ruta == PRUEBA else os.path.splitext(os.path.basename(ruta))[0]
                self.cola.put(("estado", f"Lote {i}/{len(rutas)}: {nombre}"))
                try:
                    datos = self.procesar(ruta, params)
                    self.dibujar(fig, datos)
                    fig.savefig(os.path.join(carpeta, f"{nombre}_resultado.png"))
                    filas.append(dict(imagen=nombre, **self.fila_csv(datos)))
                except Exception as e:
                    filas.append(dict(imagen=nombre, error=str(e)))
            columnas = sorted({k for f in filas for k in f}, key=lambda k: (k != "imagen", k))
            ruta_csv = os.path.join(carpeta, self.nombre_csv)
            with open(ruta_csv, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=columnas)
                w.writeheader()
                w.writerows(filas)
            return ruta_csv

        self._en_hilo(trabajo, lambda r: (self.app.estado(f"Lote terminado: {r}"),
                                          messagebox.showinfo("Listo", f"Resultados guardados en:\n{r}")))


class PestanaDocumentos(Pestana):
    nombre_csv = "resultados_ejercicio1_lsd.csv"

    def __init__(self, master, app):
        super().__init__(master, app, "Ejercicio 1 — Documentos + OCR con LSD")
        self.idioma = tk.StringVar(value="eng")
        ttk.Label(self.opciones, text="Idioma OCR:").pack(anchor="w")
        idiomas = ["eng"]
        if ej1.HAY_OCR:
            try:
                idiomas = sorted(ej1.pytesseract.get_languages()) or idiomas
            except Exception:
                pass
        ttk.Combobox(self.opciones, textvariable=self.idioma, values=idiomas, width=10, state="readonly").pack(anchor="w")
        if not ej1.HAY_OCR:
            ttk.Label(self.opciones, text=f"Sin OCR. Motivo:\n{ej1.ERROR_OCR}", wraplength=220,
                      foreground="red").pack(anchor="w", pady=(4, 0))

    def parametros(self):
        return dict(idioma=self.idioma.get())

    @staticmethod
    def procesar(ruta, p):
        return procesar_ej1(ruta, p["idioma"])

    dibujar = staticmethod(dibujar_ej1)
    resumen = staticmethod(texto_ej1)

    @staticmethod
    def fila_csv(d):
        fila = {}
        for nombre, r in d["resultados"].items():
            clave = nombre.split(")")[0].strip()  # A, B, C
            fila[f"{clave}_caracteres_ocr"] = len(r["ocr"].strip())
            if r["conf"] is not None:
                fila[f"{clave}_confianza_ocr"] = round(r["conf"], 2)
            if r["iou"] is not None:
                fila[f"{clave}_iou"] = round(r["iou"], 4)
            if r["acc"] is not None:
                fila[f"{clave}_precision_ocr"] = round(r["acc"], 4)
        fila["C_texto_ocr"] = " | ".join(d["resultados"]["C) LSD + umbral adaptativo"]["ocr"].split("\n")).strip()
        return fila


class PestanaNucleos(Pestana):
    nombre_csv = "resultados_ejercicio2_fase.csv"

    def __init__(self, master, app):
        super().__init__(master, app, "Ejercicio 2 — Núcleos con Congruencia de Fase")
        self.invertir = tk.BooleanVar(value=False)
        self.h = tk.DoubleVar(value=0.002)
        ttk.Checkbutton(self.opciones, text="Invertir (objetos oscuros)", variable=self.invertir).pack(anchor="w")
        ttk.Label(self.opciones, text="Sensibilidad h (menor = más objetos):").pack(anchor="w", pady=(4, 0))
        self.lbl_h = ttk.Label(self.opciones, text="0.0020")
        ttk.Scale(self.opciones, from_=0.0005, to=0.01, variable=self.h,
                  command=lambda v: self.lbl_h.config(text=f"{float(v):.4f}")).pack(fill="x")
        self.lbl_h.pack(anchor="e")
        self.contraste = tk.DoubleVar(value=1.15)
        ttk.Label(self.opciones, text="Contraste mínimo vs. entorno (mayor = menos ruido):").pack(anchor="w", pady=(4, 0))
        self.lbl_c = ttk.Label(self.opciones, text="1.15")
        ttk.Scale(self.opciones, from_=1.02, to=2.0, variable=self.contraste,
                  command=lambda v: self.lbl_c.config(text=f"{float(v):.2f}")).pack(fill="x")
        self.lbl_c.pack(anchor="e")

    def parametros(self):
        return dict(h=self.h.get(), invertir=self.invertir.get(), contraste=self.contraste.get())

    @staticmethod
    def procesar(ruta, p):
        return procesar_ej2(ruta, p["h"], p["invertir"], p["contraste"])

    dibujar = staticmethod(dibujar_ej2)
    resumen = staticmethod(texto_ej2)

    @staticmethod
    def fila_csv(d):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d["met"].items()}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Detección de bordes — LSD y Congruencia de Fase")
        self.geometry("1280x820")
        self.esta_ocupado = False
        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True)
        tabs.add(PestanaDocumentos(tabs, self), text="  1. Documentos (LSD)  ")
        tabs.add(PestanaNucleos(tabs, self), text="  2. Núcleos (Congruencia de fase)  ")
        barra = ttk.Frame(self)
        barra.pack(fill="x")
        self.lbl_estado = ttk.Label(barra, text="Carga una imagen, una carpeta o usa la imagen de prueba")
        self.lbl_estado.pack(side="left", padx=6, pady=2)
        self.progreso = ttk.Progressbar(barra, mode="indeterminate", length=160)
        self.progreso.pack(side="right", padx=6)

    def estado(self, msg):
        self.lbl_estado.config(text=msg)

    def ocupado(self, si):
        self.esta_ocupado = si
        self.config(cursor="watch" if si else "")
        if si:
            self.progreso.start(12)
        else:
            self.progreso.stop()


if __name__ == "__main__":
    App().mainloop()
