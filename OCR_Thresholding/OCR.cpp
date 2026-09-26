#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <iostream>
#include <chrono>
#include <sstream>
#include <algorithm>

// ===================== ESTRUCTURA: resultado de un metodo de threshold =====================
struct ResultadoOCR {
    cv::Mat imagenBinaria;
    std::string texto;
    float confianza = 0.0f;
    double tiempoMs = 0.0;
    std::string nombre;
    std::string extra; // ej: valor del umbral Otsu
};

// ===================== ESTRUCTURA: estado del scroll del panel de texto =====================
struct EstadoScroll {
    int offset = 0;      // desplazamiento vertical actual (px)
    int maxOffset = 0;   // maximo desplazamiento permitido (px)
    int paso = 45;       // px que avanza cada "click" de rueda o tecla
};

// ===================== FUNCION: Ejecutar OCR sobre una imagen binarizada =====================
std::string ejecutarOCR(const cv::Mat& imagenBinaria, tesseract::TessBaseAPI& ocr) {
    ocr.SetImage(imagenBinaria.data, imagenBinaria.cols, imagenBinaria.rows,
        1, imagenBinaria.step);
    char* texto = ocr.GetUTF8Text();
    std::string resultado(texto);
    delete[] texto;
    return resultado;
}

// ===================== FUNCION: Calcular confianza promedio del OCR =====================
float calcularConfianza(tesseract::TessBaseAPI& ocr) {
    int* confidences = ocr.AllWordConfidences();
    if (!confidences || confidences[0] == -1) return 0.0f;

    float suma = 0;
    int cantidad = 0;
    for (int i = 0; confidences[i] != -1; i++) {
        suma += confidences[i];
        cantidad++;
    }
    delete[] confidences;
    return (cantidad > 0) ? (suma / cantidad) : 0.0f;
}

// ===================== FUNCION: Envolver texto respetando saltos de linea reales =====================
std::vector<std::string> envolverTexto(const std::string& texto, size_t maxCharsPorLinea) {
    std::vector<std::string> lineasFinales;
    std::istringstream streamLineas(texto);
    std::string lineaOriginal;

    while (std::getline(streamLineas, lineaOriginal)) {
        if (lineaOriginal.empty()) {
            lineasFinales.push_back("");
            continue;
        }
        std::istringstream streamPalabras(lineaOriginal);
        std::string palabra, lineaActual;
        while (streamPalabras >> palabra) {
            if (lineaActual.size() + palabra.size() + 1 > maxCharsPorLinea) {
                lineasFinales.push_back(lineaActual);
                lineaActual = palabra;
            }
            else {
                lineaActual += (lineaActual.empty() ? "" : " ") + palabra;
            }
        }
        if (!lineaActual.empty()) lineasFinales.push_back(lineaActual);
    }
    return lineasFinales;
}

// ===================== FUNCION: Redimensionar para mostrar en pantalla =====================
cv::Mat miniatura(const cv::Mat& src, int anchoDestino) {
    cv::Mat dst;
    double escala = (double)anchoDestino / src.cols;
    cv::resize(src, dst, cv::Size(), escala, escala, cv::INTER_AREA);
    return dst;
}

// ===================== FUNCION: Construir el bloque de texto de UN metodo (estilo consola) =====================
std::vector<std::pair<std::string, cv::Scalar>> construirBloqueTexto(
    const ResultadoOCR& r, bool activo, size_t charsPorLinea) {

    std::vector<std::pair<std::string, cv::Scalar>> lineas;
    cv::Scalar colorTitulo = activo ? cv::Scalar(0, 255, 0) : cv::Scalar(120, 120, 120);
    cv::Scalar colorTexto = activo ? cv::Scalar(255, 255, 255) : cv::Scalar(110, 110, 110);
    cv::Scalar colorEtiqueta = activo ? cv::Scalar(0, 200, 255) : cv::Scalar(90, 90, 90);

    lineas.push_back({ std::string(40, '='), colorTitulo });
    lineas.push_back({ r.nombre + (activo ? "  [ACTIVO]" : ""), colorTitulo });
    if (!r.extra.empty()) lineas.push_back({ r.extra, colorTexto });
    lineas.push_back({ cv::format("Confianza promedio: %.2f%%", r.confianza), colorTexto });
    lineas.push_back({ cv::format("Tiempo OCR: %.2f ms", r.tiempoMs), colorTexto });
    lineas.push_back({ "Texto reconocido:", colorEtiqueta });

    for (const auto& l : envolverTexto(r.texto, charsPorLinea)) {
        lineas.push_back({ l.empty() ? " " : l, colorTexto });
    }
    lineas.push_back({ "", colorTexto });
    return lineas;
}

// ===================== CALLBACK: rueda del mouse para scrollear el panel =====================
void onMouse(int event, int x, int y, int flags, void* userdata) {
    EstadoScroll* estado = reinterpret_cast<EstadoScroll*>(userdata);
    if (event == cv::EVENT_MOUSEWHEEL) {
        int delta = cv::getMouseWheelDelta(flags);
        estado->offset += (delta > 0) ? -estado->paso : estado->paso;
        estado->offset = std::clamp(estado->offset, 0, estado->maxOffset);
    }
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    // ================= Cargar imagen =================
    cv::Mat img = cv::imread("D:/Decimo_Semestre/Vision_Artificial/PT2/documento.jpg");
    if (img.empty()) {
        std::cout << "Error: no se pudo cargar la imagen" << std::endl;
        return -1;
    }

    int anchoMaximo = 900;
    if (img.cols > anchoMaximo) {
        double escala = (double)anchoMaximo / img.cols;
        cv::resize(img, img, cv::Size(), escala, escala, cv::INTER_AREA);
    }

    cv::Mat gris;
    cv::cvtColor(img, gris, cv::COLOR_BGR2GRAY);

    // ===================== THRESHOLD GLOBAL (OTSU) =====================
    ResultadoOCR otsu;
    otsu.nombre = "OTSU (Global)";
    double umbralOtsu = cv::threshold(gris, otsu.imagenBinaria, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    otsu.extra = cv::format("Umbral calculado: %.1f", umbralOtsu);

    // ===================== THRESHOLD LOCAL (ADAPTATIVO) =====================
    ResultadoOCR adaptativo;
    adaptativo.nombre = "ADAPTATIVO (Local)";
    cv::adaptiveThreshold(gris, adaptativo.imagenBinaria, 255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 25, 10);
    adaptativo.extra = "Tamano de bloque: 25, Constante C: 10";

    // ================= Inicializar Tesseract =================
    tesseract::TessBaseAPI ocr;
    if (ocr.Init("D:/Decimo_Semestre/Vision_Artificial/PT2/tessdata", "spa")) {
        std::cout << "Error: no se pudo inicializar Tesseract" << std::endl;
        return -1;
    }
    ocr.SetPageSegMode(tesseract::PSM_AUTO);

    auto inicio1 = std::chrono::high_resolution_clock::now();
    otsu.texto = ejecutarOCR(otsu.imagenBinaria, ocr);
    otsu.confianza = calcularConfianza(ocr);
    auto fin1 = std::chrono::high_resolution_clock::now();
    otsu.tiempoMs = std::chrono::duration<double, std::milli>(fin1 - inicio1).count();

    auto inicio2 = std::chrono::high_resolution_clock::now();
    adaptativo.texto = ejecutarOCR(adaptativo.imagenBinaria, ocr);
    adaptativo.confianza = calcularConfianza(ocr);
    auto fin2 = std::chrono::high_resolution_clock::now();
    adaptativo.tiempoMs = std::chrono::duration<double, std::milli>(fin2 - inicio2).count();

    // ================= Salida por consola (para el informe) =================
    std::cout << "========================================" << std::endl;
    std::cout << "THRESHOLD GLOBAL (OTSU) - " << otsu.extra << std::endl;
    std::cout << "Confianza promedio: " << otsu.confianza << "%   Tiempo: " << otsu.tiempoMs << " ms" << std::endl;
    std::cout << "Texto reconocido:\n" << otsu.texto << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "THRESHOLD LOCAL (ADAPTATIVO) - " << adaptativo.extra << std::endl;
    std::cout << "Confianza promedio: " << adaptativo.confianza << "%   Tiempo: " << adaptativo.tiempoMs << " ms" << std::endl;
    std::cout << "Texto reconocido:\n" << adaptativo.texto << std::endl;
    std::cout << "========================================" << std::endl;

    // ================= Layout de la interfaz =================
    const int ANCHO_MINIATURA = 380;
    const int ANCHO_PANEL = 620;
    const int ALTO_BARRA_ETIQUETA = 36;
    const int ALTO_BARRA_INSTRUCCIONES = 30;
    const int ALTO_LINEA = 17;
    const size_t CHARS_POR_LINEA = 68;
    const int ANCHO_SCROLLBAR = 6;

    const std::string NOMBRE_VENTANA = "Comparativa OCR - Thresholding";
    cv::namedWindow(NOMBRE_VENTANA, cv::WINDOW_AUTOSIZE);

    EstadoScroll estadoScroll;
    cv::setMouseCallback(NOMBRE_VENTANA, onMouse, &estadoScroll);

    bool mostrandoOtsu = true;
    while (true) {
        const ResultadoOCR& activo = mostrandoOtsu ? otsu : adaptativo;

        // ---- Miniaturas de imagenes ----
        cv::Mat miniOriginal = miniatura(img, ANCHO_MINIATURA).clone();
        cv::Mat miniBinaria;
        cv::cvtColor(miniatura(activo.imagenBinaria, ANCHO_MINIATURA), miniBinaria, cv::COLOR_GRAY2BGR);

        cv::Mat filaImagenes;
        cv::hconcat(miniOriginal, miniBinaria, filaImagenes);

        cv::Mat barraEtiquetas(ALTO_BARRA_ETIQUETA, filaImagenes.cols, CV_8UC3, cv::Scalar(20, 20, 20));
        cv::putText(barraEtiquetas, "ORIGINAL", { 15, 25 }, cv::FONT_HERSHEY_SIMPLEX, 0.65, { 0,255,255 }, 2);
        cv::putText(barraEtiquetas, activo.nombre, { ANCHO_MINIATURA + 15, 25 },
            cv::FONT_HERSHEY_SIMPLEX, 0.65, { 0,255,0 }, 2);

        cv::Mat barraInstrucciones(ALTO_BARRA_INSTRUCCIONES, filaImagenes.cols, CV_8UC3, cv::Scalar(20, 20, 20));
        cv::putText(barraInstrucciones,
            "'O'=Otsu  'A'=Adaptativo  Rueda del mouse o 'W'/'S' = scroll texto  'ESC'=salir",
            { 15, 20 }, cv::FONT_HERSHEY_SIMPLEX, 0.48, { 200,200,0 }, 1);

        cv::Mat bloqueImagenes;
        cv::vconcat(std::vector<cv::Mat>{ barraEtiquetas, filaImagenes, barraInstrucciones }, bloqueImagenes);

        const int ALTO_VISIBLE = bloqueImagenes.rows; // alto fijo de la ventana completa

        // ---- Construir el panel de texto COMPLETO (puede ser mas alto que lo visible) ----
        auto lineasOtsu = construirBloqueTexto(otsu, mostrandoOtsu, CHARS_POR_LINEA);
        auto lineasAdapt = construirBloqueTexto(adaptativo, !mostrandoOtsu, CHARS_POR_LINEA);
        std::vector<std::pair<std::string, cv::Scalar>> todasLasLineas = lineasOtsu;
        todasLasLineas.insert(todasLasLineas.end(), lineasAdapt.begin(), lineasAdapt.end());

        int altoContenido = 25 + (int)todasLasLineas.size() * ALTO_LINEA + 15;

        cv::Mat panelCompleto(std::max(altoContenido, ALTO_VISIBLE), ANCHO_PANEL, CV_8UC3, cv::Scalar(15, 15, 15));
        int y = 25;
        for (const auto& par : todasLasLineas) {
            cv::putText(panelCompleto, par.first, { 12, y }, cv::FONT_HERSHEY_SIMPLEX, 0.42, par.second, 1);
            y += ALTO_LINEA;
        }

        // ---- Actualizar limites de scroll y recortar la parte visible ----
        estadoScroll.maxOffset = std::max(0, altoContenido - ALTO_VISIBLE);
        estadoScroll.offset = std::clamp(estadoScroll.offset, 0, estadoScroll.maxOffset);

        cv::Mat panelVisible = panelCompleto(cv::Rect(0, estadoScroll.offset, ANCHO_PANEL, ALTO_VISIBLE)).clone();

        // ---- Dibujar barra de scroll (solo si hay contenido oculto) ----
        if (estadoScroll.maxOffset > 0) {
            cv::rectangle(panelVisible,
                { ANCHO_PANEL - ANCHO_SCROLLBAR, 0 }, { ANCHO_PANEL, ALTO_VISIBLE },
                cv::Scalar(40, 40, 40), cv::FILLED);

            double proporcionVisible = (double)ALTO_VISIBLE / altoContenido;
            int altoThumb = std::max(20, (int)(ALTO_VISIBLE * proporcionVisible));
            int yThumb = (int)((double)estadoScroll.offset / estadoScroll.maxOffset * (ALTO_VISIBLE - altoThumb));
            cv::rectangle(panelVisible,
                { ANCHO_PANEL - ANCHO_SCROLLBAR, yThumb }, { ANCHO_PANEL, yThumb + altoThumb },
                cv::Scalar(0, 200, 120), cv::FILLED);
        }

        cv::Mat frameFinal;
        cv::hconcat(bloqueImagenes, panelVisible, frameFinal);
        cv::imshow(NOMBRE_VENTANA, frameFinal);

        int tecla = cv::waitKey(30);
        if (tecla == 27) break;                                   // ESC
        else if (tecla == 'o' || tecla == 'O') mostrandoOtsu = true;
        else if (tecla == 'a' || tecla == 'A') mostrandoOtsu = false;
        else if (tecla == 'w' || tecla == 'W')
            estadoScroll.offset = std::clamp(estadoScroll.offset - estadoScroll.paso, 0, estadoScroll.maxOffset);
        else if (tecla == 's' || tecla == 'S')
            estadoScroll.offset = std::clamp(estadoScroll.offset + estadoScroll.paso, 0, estadoScroll.maxOffset);
    }

    cv::destroyAllWindows();
    ocr.End();
    return 0;
}