#pragma once
#include <windows.h>
#include <stdexcept>

#include <tchar.h>
#include <shobjidl.h> 
#include <string>
#include <cmath>
#include <vector>
#include <complex>
#include <numeric>
#include <thread>
#include <pffft/pffft.h>
#include <corecrt_math_defines.h>
//#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <fftw3.h>

#include ".\include\imgui\imgui.h"
#include ".\include\imgui\imconfig.h"
#include ".\include\imgui\imgui_impl_win32.h"
#include ".\include\imgui\imgui_impl_dx11.h"
#include ".\include\implot\implot.h"
#include ".\include\implot\implot_internal.h"
#include <d3d11.h>

// Data
extern ID3D11Device* g_pd3dDevice;
extern ID3D11DeviceContext* g_pd3dDeviceContext;
extern IDXGISwapChain* g_pSwapChain;
extern bool g_SwapChainOccluded;
extern UINT g_ResizeWidth, g_ResizeHeight;
extern ID3D11RenderTargetView* g_mainRenderTargetView;

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
void init_window_class();
LRESULT CALLBACK wnd_proc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void CreateRenderTarget();
void CleanupRenderTarget();
void CleanUp();