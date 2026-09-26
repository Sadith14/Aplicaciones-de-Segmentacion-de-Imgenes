#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

// ================= Lista de imagenes a alternar =================
std::vector<std::string> rutasImagenes = {
    "D:/Decimo_Semestre/Vision_Artificial/PT2/mri.jpg",
    "D:/Decimo_Semestre/Vision_Artificial/PT2/xray.jpg",
    "D:/Decimo_Semestre/Vision_Artificial/PT2/xray2.jpg",
    "D:/Decimo_Semestre/Vision_Artificial/PT2/mri3.jpg",
    "D:/Decimo_Semestre/Vision_Artificial/PT2/xray3.jpg"
};
int indiceActual = 0;

cv::Mat imgOriginal, gris;
int umbralBajo = 50;
int umbralAlto = 150;

// ================= Dimensiones del mosaico =================
int anchoCelda = 400;
int altoCelda = 300;

// ================= Boton para cambiar de imagen =================
cv::Rect botonSiguiente(10, 10, 180, 40);

void etiquetar(cv::Mat& img, const std::string& texto) {
    cv::putText(img, texto, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
}

cv::Mat prepararCelda(const cv::Mat& img, const std::string& etiqueta) {
    cv::Mat celda, salida;
    if (img.channels() == 1) {
        cv::cvtColor(img, celda, cv::COLOR_GRAY2BGR);
    }
    else {
        celda = img.clone();
    }
    cv::resize(celda, salida, cv::Size(anchoCelda, altoCelda));
    etiquetar(salida, etiqueta);
    return salida;
}

void cargarImagenActual() {
    imgOriginal = cv::imread(rutasImagenes[indiceActual]);
    if (imgOriginal.empty()) {
        std::cout << "Error: no se pudo cargar " << rutasImagenes[indiceActual] << std::endl;
        return;
    }
    int anchoMaximo = 800;
    if (imgOriginal.cols > anchoMaximo) {
        double escala = (double)anchoMaximo / imgOriginal.cols;
        cv::resize(imgOriginal, imgOriginal, cv::Size(), escala, escala, cv::INTER_AREA);
    }
    cv::cvtColor(imgOriginal, gris, cv::COLOR_BGR2GRAY);
}

void procesar(int, void*) {
    if (imgOriginal.empty()) return;

    // ================= Suavizado =================
    cv::Mat suavizada;
    cv::GaussianBlur(gris, suavizada, cv::Size(5, 5), 0);

    // ===================== CANNY =====================
    cv::Mat bordes;
    cv::Canny(suavizada, bordes, umbralBajo, umbralAlto);

    // ===================== CIERRE MORFOLOGICO =====================
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat bordesCerrados;
    cv::morphologyEx(bordes, bordesCerrados, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);

    // ===================== CONTORNOS Y MASCARA =====================
    std::vector<std::vector<cv::Point>> contornos;
    cv::findContours(bordesCerrados, contornos, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat mascara = cv::Mat::zeros(gris.size(), CV_8UC1);
    cv::drawContours(mascara, contornos, -1, cv::Scalar(255), cv::FILLED);

    // ===================== APLICAR MASCARA =====================
    cv::Mat objetoSegmentado;
    cv::bitwise_and(imgOriginal, imgOriginal, objetoSegmentado, mascara);

    // ===================== ARMAR MOSAICO (2x2 en una sola ventana) =====================
    cv::Mat c1 = prepararCelda(imgOriginal, "Original");
    cv::Mat c2 = prepararCelda(bordes, "Bordes (Canny)");
    cv::Mat c3 = prepararCelda(mascara, "Mascara");
    cv::Mat c4 = prepararCelda(objetoSegmentado, "Objeto Segmentado");

    cv::Mat filaSuperior, filaInferior, mosaico;
    cv::hconcat(c1, c2, filaSuperior);
    cv::hconcat(c3, c4, filaInferior);
    cv::vconcat(filaSuperior, filaInferior, mosaico);

    // ===================== DIBUJAR BOTON SOBRE EL MOSAICO =====================
    cv::rectangle(mosaico, botonSiguiente, cv::Scalar(0, 120, 0), -1);
    cv::rectangle(mosaico, botonSiguiente, cv::Scalar(255, 255, 255), 1);
    cv::putText(mosaico, "Siguiente imagen", cv::Point(botonSiguiente.x + 10, botonSiguiente.y + 27),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    // ==========================================================================

    cv::imshow("Segmentacion - Canny + Morfologia", mosaico);
}

void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        if (botonSiguiente.contains(cv::Point(x, y))) {
            indiceActual = (indiceActual + 1) % rutasImagenes.size();
            cargarImagenActual();
            procesar(0, 0);
        }
    }
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    cargarImagenActual();
    if (imgOriginal.empty()) return -1;

    cv::namedWindow("Segmentacion - Canny + Morfologia");
    cv::setMouseCallback("Segmentacion - Canny + Morfologia", onMouse);

    // Trackbars van en una ventana aparte para no interferir con el mosaico/boton
    cv::namedWindow("Controles");
    cv::createTrackbar("Umbral Bajo", "Controles", &umbralBajo, 255, procesar);
    cv::createTrackbar("Umbral Alto", "Controles", &umbralAlto, 255, procesar);

    procesar(0, 0);

    cv::waitKey(0);
    return 0;
}