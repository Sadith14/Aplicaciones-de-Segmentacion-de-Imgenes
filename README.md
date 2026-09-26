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

## Estructura sugerida
