#include "render.h"

#include ".\include\implot\implot.h"
#include ".\include\implot\implot_internal.h"

//using namespace Eigen;
using namespace cv;
using namespace std;

const float c = 299792458.0f;  // Скорость света
const float freq = 0.14e12;   // Частота
const float lambda = (c * 1000.0) / freq;  // Длина волны
const float k = 2 * M_PI / lambda;  // Волновой вектор
const int pixels = 32;  // Количество пикселей
const float pixelsize = 0.53;  // Размер пикселя
const float focus = 30.0;  // Фокусное расстояние
const int iteration_num = 100;  // Количество итераций

std::mutex resultMutex;


ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
vector<vector<complex<float>>> far_field(pixels, vector<complex<float>>(pixels, 0.0f));
vector<vector<complex<float>>> far_field2(pixels, vector<complex<float>>(pixels, 0.0f));
vector<float> input_target_vec(pixels * pixels, 0);
vector<vector<float>> input_i;
float power_in;

//algo stuff
vector<float> r = { 0, 0, focus };

vector<vector<complex<float>>> Ed1(pixels, vector<complex<float>>(pixels, 0.0f));

vector<complex<float>> temp1(pixels, 0.0f);
vector<float> temp2(pixels, 0.0f);
vector<complex<float>> temp3(pixels, 0.0f);
float error_f = 0;
const std::complex<float> i(0.0, 1.0);


// Векторное произведение двух vector<float>
vector<float> cross_product(const vector<float>& a, const vector<float>& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

// Скалярное произведение двух vector<float>
float dot_product(const vector<float>& a, const vector<float>& b) {
    return inner_product(a.begin(), a.end(), b.begin(), 0.0f);
}

// Среднее в vector<float>
float arr_mean(const vector<float>& arr) {
    if (arr.empty()) {
        return 0.0; // Чтобы избежать деления на ноль
    }
    float sum = std::accumulate(arr.begin(), arr.end(), 0.0); // Суммируем элементы
    return sum / arr.size(); // Делим сумму на количество элементов
}

// Евклидова норма (длина вектора) vector<float>
float l2_norm(const vector<float>& vec) {
    float sum = 0.0;
    for (float value : vec) {
        sum += std::pow(value, 2);
    }
    return std::sqrt(sum);
}

// Линейное пространство (разбитие на сетку) от start до finish с количеством pixels
void linspace(vector<float>& input, int pixels, float start, float finish) {
    input.clear();
    float diff = (finish - start)/pixels;
    for (int i = 1; i <= pixels; ++i) {
        input.push_back(i*diff);
    }
}

// Сумма элементов матрицы vector<vector<float>>
float matrix_sum(vector<vector<float>> input) {
    float result = 0;
    for (const auto& row : input) {
        for (const auto& value : row) {
            result += value;
        }
    }
    return result;
}

// Сумма вектора комплексных чисел
complex<float> complex_sum(vector<complex<float>> input) {
    complex<float> result;
    for (const auto& i : input) {
        result += i;
    }
    return result;
}

// Развертывание матрицы input в длинный vector output
template<typename T>
void unwrap(const vector<vector<T>>& input, vector<T>& output) {
    output.clear();
    for (const auto& row : input) {
        output.insert(output.end(), row.begin(), row.end()) ;
    }
}

// Свертывание длинного vector input в матрицу output по кол-ву эл-ов size
template<typename T>
void wrap(const vector<T>& input, vector<vector<T>>& output, int size) {
    output.clear();
    for (int j = 0; j < input.size(); j += size) {
        auto start = input.begin() + j;
        auto end = (j + size <= input.size()) ? start + size : input.end();
        output.emplace_back(start, end);
    }
}

// Real-part от вектора input 
template<typename T>
void decomplex(const vector<complex<T>>& input, vector<T>& output) {
    output.clear();
    output.reserve(input.size());
    for (auto& value : input) {
        output.push_back(value.real());
    }
}

// Установка единичной матрицы на вход
void set_input_intensity() {
    float resolution = pixels * pixelsize;

    vector<float> x;
    vector<float> y;

    linspace(x, pixels, -resolution / 2, resolution / 2);
    linspace(y, pixels, -resolution / 2, resolution / 2);

    input_i.assign(y.size(), vector<float>(x.size(), 1));

    power_in = matrix_sum(input_i);
}

// Чтение изображения из указанного файла. TODO: Сделать окошко с выбором файла
void image_read(Mat& target) {

    target = imread("./images/cross2_300x300_small.png", IMREAD_GRAYSCALE);

    target.convertTo(target, CV_32F);

    target = (target - 255.0) * (-1.0) / 255.0;

    target = target * (power_in / cv::sum(target)[0]);

    Size newSize(pixels, pixels);

    Mat resizedTarget;
    cv::resize(target, resizedTarget, newSize, 0, 0, cv::INTER_AREA); // INTER_AREA лучше всего подходит для уменьшения

    target = resizedTarget;
}

// Вернуть размер Mat
int return_sizeof_mat(const Mat& target) {
    // Получение размеров матрицы
    int rows = target.rows;
    int cols = target.cols;

    int size_total = rows * cols;
    return size_total;
}

// Единичное ФФТ преобразование
void single_fft(int size_total, const Mat& resizedTarget, vector<complex<float>>& fft_result) {
    PFFFT_Setup* pffft_setup = pffft_new_setup(pixels * pixels, PFFFT_REAL);

    // Создание массива C++ для хранения данных
    float* input_target = new float[size_total];

    input_target_vec.clear();

    if (resizedTarget.isContinuous()) {
        // Если данные непрерывны, копируем их напрямую
        std::memcpy(input_target, resizedTarget.datastart, sizeof(float) * size_total);
        input_target_vec.assign((float*)resizedTarget.datastart, (float*)resizedTarget.dataend);
    }
    else {
        // Если данные не непрерывны, копируем построчно
        size_t idx = 0;
        for (int i = 0; i < resizedTarget.rows; ++i) {
            std::memcpy(input_target + idx, resizedTarget.ptr<float>(i), resizedTarget.cols * resizedTarget.elemSize());
            input_target_vec.insert(input_target_vec.end(), resizedTarget.ptr<float>(i), resizedTarget.ptr<float>(i) + resizedTarget.cols);
            idx += resizedTarget.cols;
        }
    }

    float* output = new float[size_total * 2];

    pffft_transform_ordered(pffft_setup, input_target, output, NULL, PFFFT_BACKWARD);

    fft_result.resize(size_total + 1, 0);
    for (int i = 0; i < size_total + 1; ++i) {
        fft_result[i] = complex<float>(output[2 * i], output[2 * i + 1]);
    }

    delete[] output;
    delete[] input_target;
    pffft_destroy_setup(pffft_setup);
}

// Часть цикла SPR алгоритма в + направлении (+focus)
void spr_cycle_threading_front(int start, int end, const vector<vector<complex<float>>>& near_field) {
    vector<vector<complex<float>>> Ed(pixels, vector<complex<float>>(pixels, 0.0f));
    vector<vector<float>> omega(pixels, vector<float>(pixels, 0.0f));
    vector<vector<complex<float>>> far_field_temp(pixels, vector<complex<float>>(pixels, 0.0f));
    vector<complex<float>> temp1(pixels, 0.0f);
    vector<float> temp2(pixels, 0.0f);
    vector<complex<float>> temp3(pixels, 0.0f);


    Ed.resize(pixels);
    for (auto& row : Ed)
        row.resize(pixels);

    for (int x2 = start; x2 < end; ++x2) {
        for (int y2 = 0; y2 < pixels; ++y2) {
            for (int x1 = 0; x1 < pixels; ++x1) {
                for (int y1 = 0; y1 < pixels; ++y1) {
                    r = { (x2 * pixelsize) - (x1 * pixelsize), (y2 * pixelsize) - (y1 * pixelsize), focus };
                    Ed[x1][y1] = (i * input_i[x1][y1] / (l2_norm(r) * 2 * lambda) * exp(i * k * l2_norm(r) + i * arg(near_field[x1][y1])));
                    omega[x1][y1] = atan2(l2_norm(cross_product(r, { 0, 0, 1 })), dot_product(r, { 0, 0, 1 }));
                }
            }
            unwrap(Ed, temp1);
            unwrap(omega, temp2);
            temp3.resize(temp1.size(), 0.0f);
            for (int k = 0; k < temp1.size(); ++k)
                temp3[k] = temp1[k] * (1 + cos(temp2[k]));
            far_field_temp[y2][x2-start] = complex_sum(temp3);
            //far_field[y2][x2] = complex_sum(temp3);
        }
    }

    unique_lock<mutex> lock(resultMutex);
    for (int x2 = start; x2 < end; ++x2) {
        for (int y2 = 0; y2 < pixels; ++y2)
            far_field[y2][x2] = far_field[y2][x2 - start];
    }
    lock.unlock();

}

// Часть цикла SPR алгоритма в - направлении (-focus)
void spr_cycle_threading_back(int start, int end, vector<vector<complex<float>>>& near_field) {
    vector<vector<complex<float>>> Ed(pixels, vector<complex<float>>(pixels, 0.0f));
    vector<vector<float>> omega(pixels, vector<float>(pixels, 0.0f));
    vector<vector<complex<float>>> near_field_temp(pixels, vector<complex<float>>(pixels, 0.0f));
    vector<complex<float>> temp1(pixels, 0.0f);
    vector<float> temp2(pixels, 0.0f);
    vector<complex<float>> temp3(pixels, 0.0f);

    Ed.resize(pixels);
    for (auto& row : Ed)
        row.resize(pixels);

    for (int x2 = start; x2 < end; ++x2) {
        for (int y2 = 0; y2 < pixels; ++y2) {
            for (int x1 = 0; x1 < pixels; ++x1) {
                for (int y1 = 0; y1 < pixels; ++y1) {
                    r = { (x2 * pixelsize) - (x1 * pixelsize), (y2 * pixelsize) - (y1 * pixelsize), -focus };
                    Ed[x1][y1] = (i * abs(far_field2[x1][y1]) / (l2_norm(r) * 2 * lambda) * exp(-i * k * l2_norm(r) + i * arg(far_field2[x1][y1])));
                    omega[x1][y1] = atan2(l2_norm(cross_product(r, { 0, 0, -1 })), dot_product(r, { 0, 0, -1 }));
                }
            }
            unwrap(Ed1, temp1);
            unwrap(omega, temp2);
            temp3.resize(temp1.size(), 0.0f);
            for (int k = 0; k < temp1.size(); ++k)
                temp3[k] = temp1[k] * (1 + cos(temp2[k]));
            near_field_temp[y2][x2 - start] = complex_sum(temp3);
            //near_field[y2][x2] = complex_sum(temp3);
        }
    }
    unique_lock<mutex> lock(resultMutex);
    for (int x2 = start; x2 < end; ++x2) {
        for (int y2 = 0; y2 < pixels; ++y2)
            near_field[y2][x2] = near_field[y2][x2 - start];
    }
    lock.unlock();

}

// SPR алгоритм
void spr() {
    const int num_threads = 4; // thread::hardware_concurrency();
    vector<thread> threads;


    set_input_intensity();
    Mat target;
    image_read(target);
    int size_total = return_sizeof_mat(target);

    vector<complex<float>> fft_result;
    single_fft(size_total, target, fft_result);

    vector<vector<complex<float>>> near_field(pixels, vector<complex<float>>(pixels, 0.0f));

    wrap(fft_result, near_field, pixels);
    for (auto& rows : near_field) {
        for (int j = 0; j < rows.size(); ++j) {
            rows[j] = exp(i * arg(rows[j]));
        }
    }

    for (int iter = 0; iter < iteration_num; ++iter) {

        for (int i = 0; i < num_threads; ++i) {
            int temp = pixels / num_threads;
            int start = i * num_threads;
            int end = (i == num_threads - 1) ? pixels : start + temp;
            threads.emplace_back(spr_cycle_threading_front, start, end, std::ref(near_field));
        }

        for (auto& thread : threads) {
            thread.join();
        }

        threads.clear();

        unwrap(far_field, temp1);
        temp3.resize(size_total, 0.0f);
        for (int k = 0; k < size_total; ++k)
            temp3[k] = abs(input_target_vec[k] * exp(i * arg(temp1[k])));
        wrap(temp3, far_field2, pixels); 

        for (int i = 0; i < num_threads; ++i) {
            int temp = pixels / num_threads;
            int start = i * num_threads;
            int end = (i == num_threads - 1) ? pixels : start + temp;
            threads.emplace_back(spr_cycle_threading_front, start, end, std::ref(near_field));
        }

        for (auto& thread : threads) {
            thread.join();
        }   

        threads.clear();

        unwrap(near_field, temp1);
        temp3.resize(temp1.size(), 0.0f);
        for (int k = 0; k < temp1.size(); ++k)
            temp3[k] = exp(i * arg(temp1[k]));
        wrap(temp3, near_field, pixels);
        unwrap(far_field2, temp1);

        for (int k = 0; k < size_total; ++k) {
            error_f += (pow(abs(temp1[k]), 2.0f) / pow(abs(input_target_vec[k]), 2.0f));
        }
    }
}

void main() {

    init_window_class();
    bool done = false;
    while (!done)
    {
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE)) // message dispatching
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT)
                done = true;
        }
        if (done)
            break;

        // Handle window being minimized or screen locked
        if (g_SwapChainOccluded && g_pSwapChain->Present(0, DXGI_PRESENT_TEST) == DXGI_STATUS_OCCLUDED)
        {
            ::Sleep(10);
            continue;
        }
        g_SwapChainOccluded = false;

        // Handle window resize (we don't resize directly in the WM_SIZE handler)
        if (g_ResizeWidth != 0 && g_ResizeHeight != 0)
        {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, g_ResizeWidth, g_ResizeHeight, DXGI_FORMAT_UNKNOWN, 0);
            g_ResizeWidth = g_ResizeHeight = 0;
            CreateRenderTarget();
        }

        // Start the Dear ImGui frame
        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        bool show_demo_window = true;
        // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);


        // Start of main window
        static bool use_work_area = true;
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(use_work_area ? viewport->WorkPos : viewport->Pos);
        ImGui::SetNextWindowSize(use_work_area ? viewport->WorkSize : viewport->Size);

        static ImGuiWindowFlags flags = NULL;// ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBringToFrontOnFocus;
        ImGui::Begin("Main", NULL, flags);

        if (ImGui::Button("Start")) {
            spr();
        }

        static float max_val = *max_element(input_target_vec.begin(), input_target_vec.end());
        static float min_val = *min_element(input_target_vec.begin(), input_target_vec.end());

        static ImPlotAxisFlags axes_flags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks;
        ImPlot::PushColormap(ImPlotColormap_Greys);
        if (ImPlot::BeginPlot("Target Image", ImVec2(350, 350), ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText)) {
            ImPlot::SetupAxes(nullptr, nullptr, axes_flags, axes_flags);
            ImPlot::PlotHeatmap("heat", &input_target_vec[0], pixels, pixels, min_val, max_val, nullptr, ImPlotPoint(0, 0), ImPlotPoint(1, 1), ImPlotHeatmapFlags_ColMajor);
            ImPlot::EndPlot();
        }
        ImGui::SameLine();
        ImPlot::ColormapScale("##target", min_val, max_val, ImVec2(60, 225));

        unwrap(far_field, temp1);
        decomplex(temp1, temp2);

        static float max_val_res = *max_element(temp2.begin(), temp2.end());
        static float min_val_res = *min_element(temp2.begin(), temp2.end());

        static ImPlotAxisFlags res_axes_flags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks;
        ImPlot::PushColormap(ImPlotColormap_Greys);
        if (ImPlot::BeginPlot("Result Image", ImVec2(350, 350), ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText)) {
            ImPlot::SetupAxes(nullptr, nullptr, axes_flags, axes_flags);
            ImPlot::PlotHeatmap("result", &temp2[0], pixels, pixels, min_val_res, max_val_res, nullptr, ImPlotPoint(0, 0), ImPlotPoint(1, 1), ImPlotHeatmapFlags_ColMajor);
            ImPlot::EndPlot();
        }
        ImGui::SameLine();
        ImPlot::ColormapScale("##target", min_val_res, max_val_res, ImVec2(60, 225));

        ImGui::End();
        // Rendering
        ImGui::Render();
        const float clear_color_with_alpha[4] = { clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color_with_alpha);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        // Present
        HRESULT hr = g_pSwapChain->Present(1, 0);   // Present with vsync
        //HRESULT hr = g_pSwapChain->Present(0, 0); // Present without vsync
        g_SwapChainOccluded = (hr == DXGI_STATUS_OCCLUDED);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        // (Your code calls swapchain's Present() function)
    }

    CleanUp();
}

