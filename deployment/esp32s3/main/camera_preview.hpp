#pragma once

#include "esp_err.h"

namespace dl {
class Model;
}

/** 初始化 OV2640、自建 Wi-Fi 热点和 PC 浏览器预览服务。 */
esp_err_t start_camera_preview(dl::Model *model);
