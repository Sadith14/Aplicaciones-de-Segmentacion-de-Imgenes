# Aplicaciones de Segmentación de Imágenes — Thresholding y Detección de Bordes

Informe y aplicaciones del curso **Visión Artificial** — Escuela Profesional de Ingeniería de Sistemas, Universidad Nacional del Altiplano.

## Descripción general

Aplicación y comparación de dos técnicas clásicas de segmentación —**thresholding** y **detección de bordes**— en tres dominios reales: OCR, imágenes médicas y control de calidad industrial. Incluye además dos detectores alternativos a Canny: **LSD** y **Congruencia de Fase**.

## Herramientas

OpenCV 4.12.0 · C++ (Visual Studio 2026, vcpkg) · Python 3 · Tesseract OCR / pytesseract · scikit-image · SciPy · Tkinter

## Aplicaciones

1. **OCR con Thresholding** — Binariza un comprobante de pago (Otsu vs. adaptativo) antes del OCR con Tesseract; compara confianza y precisión del texto extraído.

2. **Segmentación médica con Thresholding** — Aísla estructura ósea en radiografías (dataset FracAtlas) probando seis métodos de umbralización; Otsu resultó el más efectivo.

3. **Control de calidad industrial con Thresholding** — Detecta defectos en superficies metálicas (dataset NEU-DET) con Otsu y umbral adaptativo, emitiendo veredicto OK/DEFECTUOSA.

4. **OCR / Documentos con Canny** — Detecta el contorno de una hoja fotografiada (Canny implementado paso a paso), corrige perspectiva y binariza para OCR; incluye pipeline híbrido Canny + thresholding.

5. **Segmentación con Canny (Object Masking)** — Aísla anatomías (cráneo, mano, pie) en imágenes médicas mediante Canny + morfología + contorno de mayor área, sobre 5 casos de prueba.

6. **OCR / Documentos con LSD** — Alternativa a Canny para localizar bordes rectos de una hoja en condiciones difíciles (sombra, bajo contraste); combinado con umbral adaptativo logra ~91% de confianza OCR.

7. **Núcleos celulares con Congruencia de Fase** — Segmenta y cuenta núcleos en microscopía de fluorescencia usando congruencia de fase (invariante al contraste) + watershed; F1@0.5 = 0.974.

## Estructura del repositorio
 
```
.
├── OCR_Thresholding/                      # Ítem 1 — OCR con thresholding (Otsu vs. adaptativo)
├── Aplicación2_Segmentación de imágenes/  # Ítem 2 — Segmentación de radiografías (Otsu)
├── Thresholding-C-control_calidad/        # Ítem 3 — Control de calidad (NEU-DET)
├── CannySegmentacion/                     # Ítems 5 y 6 — Canny (OCR/documentos e imágenes médicas)
├── Aplicación_4-OCR con Canny.py          # Ítem 5 — Script de escaneo de documentos con Canny
├── Bordes-extra-LSD-Congruencia de fase/  # Ítems 5b y 6b — Detectores alternativos (LSD, Congruencia de Fase)
└── README.md
```
 
## Resumen por aplicación
 
| Ítem | Aplicación | Técnica | Dataset / imagen | Carpeta |
|---|---|---|---|---|
| 1 | OCR de un comprobante de pago | Thresholding: Otsu vs. adaptativo | Comprobante Banco de la Nación | `OCR_Thresholding/` |
| 2 | Segmentación ósea en radiografías | Thresholding: Otsu (y 5 métodos más comparados) | FracAtlas | `Aplicación2_Segmentación de imágenes/` |
| 3 | Control de calidad industrial | Thresholding: Otsu vs. adaptativo | NEU-DET | `Thresholding-C-control_calidad/` |
| 5 | Escaneo de documentos | Canny | Recibo fotografiado | `CannySegmentacion/`, `Aplicación_4-OCR con Canny.py` |
| 5b | Escaneo de documentos (alternativo) | LSD (Line Segment Detector) | Imagen sintética con sombra | `Bordes-extra-LSD-Congruencia de fase/` |
| 6 | Segmentación de estructuras anatómicas | Canny + morfología | MRI, radiografías, ilustraciones | `CannySegmentacion/` |
| 6b | Segmentación de núcleos celulares (alternativo) | Congruencia de Fase + watershed | Microscopía sintética, radiografía de mano | `Bordes-extra-LSD-Congruencia de fase/` |
 
Los ítems 4 y 7 corresponden a desarrollos teóricos (explicación matemática del
thresholding y del algoritmo de Canny, respectivamente) incluidos únicamente en el
informe, sin carpeta ni código asociado en este repositorio.
 
## Tecnologías
 
- **OpenCV 4.12.0** — librería principal de procesamiento de imágenes.
- **C++** (Visual Studio 2026, vcpkg) — ítems 1 y 2.
- **Python 3** (OpenCV, scikit-image, SciPy, NumPy, Matplotlib) — ítems 3, 5b y 6b.
- **Tesseract OCR** + **pytesseract** — reconocimiento de texto (ítems 1 y 5).
- **Tkinter** — interfaces gráficas de escritorio interactivas.
## Datasets utilizados
 
- **NEU-DET** (NEU Surface Defect Database) — defectos superficiales en acero laminado.
- **FracAtlas** — radiografías musculoesqueléticas (licencia CC BY 4.0).
