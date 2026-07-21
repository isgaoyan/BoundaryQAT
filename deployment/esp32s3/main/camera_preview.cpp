#include "camera_preview.hpp"

#include <inttypes.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>

#include "dl_model_base.hpp"
#include "esp_camera.h"
#include "esp_event.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "img_converters.h"
#include "nvs_flash.h"

namespace {

static const char *TAG = "相机预览";
static constexpr char WIFI_SSID[] = "BoundaryQAT-CAM";
static constexpr char WIFI_PASSWORD[] = "boundaryqat";

static constexpr int CAMERA_PIN_SIOD = 4;
static constexpr int CAMERA_PIN_SIOC = 5;
static constexpr int CAMERA_PIN_VSYNC = 6;
static constexpr int CAMERA_PIN_HREF = 7;
static constexpr int CAMERA_PIN_XCLK = 15;
static constexpr int CAMERA_PIN_PCLK = 13;
static constexpr int CAMERA_PIN_Y2 = 11;
static constexpr int CAMERA_PIN_Y3 = 9;
static constexpr int CAMERA_PIN_Y4 = 8;
static constexpr int CAMERA_PIN_Y5 = 10;
static constexpr int CAMERA_PIN_Y6 = 12;
static constexpr int CAMERA_PIN_Y7 = 18;
static constexpr int CAMERA_PIN_Y8 = 17;
static constexpr int CAMERA_PIN_Y9 = 16;
static constexpr int PREVIEW_WIDTH = 320;
static constexpr int PREVIEW_HEIGHT = 240;
static constexpr int MODEL_SIZE = 64;
static constexpr int MODEL_CONTENT_HEIGHT = 48;
static constexpr int MODEL_PAD_TOP = 8;
static constexpr float SEGMENTATION_THRESHOLD = 0.14F;

static httpd_handle_t server = nullptr;
static dl::Model *preview_model = nullptr;
static uint8_t visited_pixels[MODEL_SIZE * MODEL_SIZE] = {};
static uint8_t largest_component_mask[MODEL_SIZE * MODEL_SIZE] = {};
static uint16_t component_queue[MODEL_SIZE * MODEL_SIZE] = {};

struct PupilOverlay {
    bool found;
    float center_x;
    float center_y;
    uint32_t area;
};

static constexpr char INDEX_HTML[] = R"HTML(
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>BoundaryQAT 红外相机预览</title>
  <style>
    body { margin: 0; background: #111827; color: #f9fafb; font-family: sans-serif; text-align: center; }
    main { max-width: 720px; margin: auto; padding: 20px; }
    img { width: 100%; image-rendering: auto; border-radius: 12px; background: #000; }
    p { color: #cbd5e1; line-height: 1.6; }
    button { padding: 10px 18px; border: 0; border-radius: 8px; font-size: 16px; cursor: pointer; }
  </style>
</head>
<body>
  <main>
    <h1>OV2640 夜视红外预览</h1>
    <p>红色为模型分割边缘，绿色十字为瞳孔质心。先对普通物体验证；未经光生物安全评估，不得使用红外光源照射眼睛。</p>
    <img id="preview" alt="相机画面">
    <p id="state">正在获取画面……</p>
    <button onclick="refreshImage()">立即刷新</button>
  </main>
  <script>
    const preview = document.getElementById('preview');
    const state = document.getElementById('state');
    let timer;
    function refreshImage() {
      clearTimeout(timer);
      state.textContent = '正在获取画面……';
      preview.src = '/capture?t=' + Date.now();
    }
    preview.onload = () => {
      state.textContent = '画面正常';
      timer = setTimeout(refreshImage, 500);
    };
    preview.onerror = () => {
      state.textContent = '取图失败，正在重试……';
      timer = setTimeout(refreshImage, 1500);
    };
    refreshImage();
  </script>
</body>
</html>
)HTML";

/** 按已确认的 GOOUUU ESP32-S3-CAM 引脚初始化 OV2640。 */
static esp_err_t initialize_camera()
{
    camera_config_t config = {};
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = CAMERA_PIN_Y2;
    config.pin_d1 = CAMERA_PIN_Y3;
    config.pin_d2 = CAMERA_PIN_Y4;
    config.pin_d3 = CAMERA_PIN_Y5;
    config.pin_d4 = CAMERA_PIN_Y6;
    config.pin_d5 = CAMERA_PIN_Y7;
    config.pin_d6 = CAMERA_PIN_Y8;
    config.pin_d7 = CAMERA_PIN_Y9;
    config.pin_xclk = CAMERA_PIN_XCLK;
    config.pin_pclk = CAMERA_PIN_PCLK;
    config.pin_vsync = CAMERA_PIN_VSYNC;
    config.pin_href = CAMERA_PIN_HREF;
    config.pin_sccb_sda = CAMERA_PIN_SIOD;
    config.pin_sccb_scl = CAMERA_PIN_SIOC;
    config.pin_pwdn = -1;
    config.pin_reset = -1;
    config.xclk_freq_hz = 10000000;
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

    const esp_err_t result = esp_camera_init(&config);
    if (result != ESP_OK) {
        ESP_LOGE(TAG, "OV2640 初始化失败：%s (0x%x)", esp_err_to_name(result), result);
        return result;
    }

    sensor_t *sensor = esp_camera_sensor_get();
    if (sensor == nullptr || sensor->id.PID != OV2640_PID) {
        ESP_LOGE(TAG, "未识别到预期的 OV2640 传感器");
        esp_camera_deinit();
        return ESP_ERR_NOT_FOUND;
    }

    ESP_LOGI(TAG, "OV2640 初始化成功：QVGA 320x240，RGB565，单帧缓冲");
    return ESP_OK;
}

/** 初始化 NVS 和开发板自建的 Wi-Fi 热点。 */
static esp_err_t initialize_wifi_access_point()
{
    esp_err_t result = nvs_flash_init();
    if (result != ESP_OK) {
        ESP_LOGE(TAG, "NVS 初始化失败：%s；为保护已有数据，本程序不会自动擦除 NVS", esp_err_to_name(result));
        return result;
    }

    if ((result = esp_netif_init()) != ESP_OK) {
        return result;
    }
    if ((result = esp_event_loop_create_default()) != ESP_OK) {
        return result;
    }
    if (esp_netif_create_default_wifi_ap() == nullptr) {
        return ESP_FAIL;
    }

    wifi_init_config_t initialization = WIFI_INIT_CONFIG_DEFAULT();
    if ((result = esp_wifi_init(&initialization)) != ESP_OK) {
        return result;
    }

    wifi_config_t access_point = {};
    std::memcpy(access_point.ap.ssid, WIFI_SSID, sizeof(WIFI_SSID));
    std::memcpy(access_point.ap.password, WIFI_PASSWORD, sizeof(WIFI_PASSWORD));
    access_point.ap.ssid_len = std::strlen(WIFI_SSID);
    access_point.ap.channel = 1;
    access_point.ap.max_connection = 2;
    access_point.ap.authmode = WIFI_AUTH_WPA2_PSK;
    access_point.ap.pmf_cfg.required = false;

    if ((result = esp_wifi_set_mode(WIFI_MODE_AP)) != ESP_OK ||
        (result = esp_wifi_set_config(WIFI_IF_AP, &access_point)) != ESP_OK ||
        (result = esp_wifi_start()) != ESP_OK) {
        return result;
    }

    ESP_LOGI(TAG, "Wi-Fi 热点已启动：名称=%s，密码=%s", WIFI_SSID, WIFI_PASSWORD);
    return ESP_OK;
}

/** 返回中文相机预览页面。 */
static esp_err_t handle_index(httpd_req_t *request)
{
    httpd_resp_set_type(request, "text/html; charset=utf-8");
    httpd_resp_set_hdr(request, "Cache-Control", "no-store");
    return httpd_resp_send(request, INDEX_HTML, HTTPD_RESP_USE_STRLEN);
}

/** 将 RGB565 大端像素转换为 8 位灰度值。 */
static uint8_t rgb565_to_grayscale(const uint8_t *pixel_bytes)
{
    const uint16_t pixel = (static_cast<uint16_t>(pixel_bytes[0]) << 8U) | pixel_bytes[1];
    const uint32_t red = ((pixel >> 11U) & 0x1FU) * 255U / 31U;
    const uint32_t green = ((pixel >> 5U) & 0x3FU) * 255U / 63U;
    const uint32_t blue = (pixel & 0x1FU) * 255U / 31U;
    return static_cast<uint8_t>((77U * red + 150U * green + 29U * blue + 128U) >> 8U);
}

/** 将 QVGA RGB565 帧等比例缩放并补边为模型要求的 64×64 INT8 输入。 */
static bool prepare_camera_input(const camera_fb_t *frame, dl::TensorBase *input)
{
    if (frame == nullptr || frame->format != PIXFORMAT_RGB565 ||
        frame->width != PREVIEW_WIDTH || frame->height != PREVIEW_HEIGHT ||
        input == nullptr || input->get_dtype() != dl::DATA_TYPE_INT8 ||
        input->get_size() != MODEL_SIZE * MODEL_SIZE) {
        return false;
    }

    int8_t *input_data = input->get_element_ptr<int8_t>();
    std::fill(input_data, input_data + MODEL_SIZE * MODEL_SIZE, static_cast<int8_t>(0));
    const float input_scale = std::ldexp(1.0F, input->get_exponent());
    constexpr int sample_step = PREVIEW_WIDTH / MODEL_SIZE;

    for (int model_y = 0; model_y < MODEL_CONTENT_HEIGHT; ++model_y) {
        for (int model_x = 0; model_x < MODEL_SIZE; ++model_x) {
            uint32_t grayscale_sum = 0;
            for (int offset_y = 0; offset_y < sample_step; ++offset_y) {
                const int source_y = model_y * sample_step + offset_y;
                for (int offset_x = 0; offset_x < sample_step; ++offset_x) {
                    const int source_x = model_x * sample_step + offset_x;
                    const size_t byte_index =
                        static_cast<size_t>(source_y * PREVIEW_WIDTH + source_x) * 2U;
                    grayscale_sum += rgb565_to_grayscale(frame->buf + byte_index);
                }
            }

            const uint32_t grayscale = grayscale_sum / static_cast<uint32_t>(sample_step * sample_step);
            const float normalized = static_cast<float>(grayscale) / 255.0F;
            const int quantized = std::clamp(static_cast<int>(std::lround(normalized / input_scale)), -128, 127);
            const int model_index = (model_y + MODEL_PAD_TOP) * MODEL_SIZE + model_x;
            input_data[model_index] = static_cast<int8_t>(quantized);
        }
    }
    return true;
}

/** 从阈值掩码中提取八连通最大区域，并计算映射到 QVGA 的质心。 */
static PupilOverlay locate_pupil_overlay(dl::TensorBase *output)
{
    PupilOverlay result = {};
    if (output == nullptr || output->get_dtype() != dl::DATA_TYPE_INT8 ||
        output->get_size() != MODEL_SIZE * MODEL_SIZE) {
        return result;
    }

    const float output_scale = std::ldexp(1.0F, output->get_exponent());
    const int threshold_int8 = std::clamp(
        static_cast<int>(std::ceil(SEGMENTATION_THRESHOLD / output_scale)), -128, 127);
    const int8_t *output_data = output->get_element_ptr<int8_t>();
    std::fill(std::begin(visited_pixels), std::end(visited_pixels), static_cast<uint8_t>(0));
    std::fill(std::begin(largest_component_mask),
              std::end(largest_component_mask),
              static_cast<uint8_t>(0));

    uint32_t largest_area = 0;
    uint32_t largest_sum_x = 0;
    uint32_t largest_sum_y = 0;
    for (int start = 0; start < MODEL_SIZE * MODEL_SIZE; ++start) {
        if (visited_pixels[start] != 0 || output_data[start] < threshold_int8) {
            continue;
        }

        size_t queue_head = 0;
        size_t queue_tail = 0;
        uint32_t area = 0;
        uint32_t sum_x = 0;
        uint32_t sum_y = 0;
        component_queue[queue_tail++] = static_cast<uint16_t>(start);
        visited_pixels[start] = 1;

        while (queue_head < queue_tail) {
            const int index = component_queue[queue_head++];
            const int x = index % MODEL_SIZE;
            const int y = index / MODEL_SIZE;
            ++area;
            sum_x += static_cast<uint32_t>(x);
            sum_y += static_cast<uint32_t>(y);

            for (int delta_y = -1; delta_y <= 1; ++delta_y) {
                for (int delta_x = -1; delta_x <= 1; ++delta_x) {
                    if (delta_x == 0 && delta_y == 0) {
                        continue;
                    }
                    const int next_x = x + delta_x;
                    const int next_y = y + delta_y;
                    if (next_x < 0 || next_x >= MODEL_SIZE || next_y < 0 || next_y >= MODEL_SIZE) {
                        continue;
                    }
                    const int next_index = next_y * MODEL_SIZE + next_x;
                    if (visited_pixels[next_index] == 0 && output_data[next_index] >= threshold_int8) {
                        visited_pixels[next_index] = 1;
                        component_queue[queue_tail++] = static_cast<uint16_t>(next_index);
                    }
                }
            }
        }

        if (area > largest_area) {
            largest_area = area;
            largest_sum_x = sum_x;
            largest_sum_y = sum_y;
            std::fill(std::begin(largest_component_mask),
                      std::end(largest_component_mask),
                      static_cast<uint8_t>(0));
            for (size_t queue_index = 0; queue_index < queue_tail; ++queue_index) {
                largest_component_mask[component_queue[queue_index]] = 1;
            }
        }
    }

    if (largest_area == 0) {
        return result;
    }

    constexpr float model_to_preview_scale = static_cast<float>(PREVIEW_WIDTH) / MODEL_SIZE;
    const float model_center_x = static_cast<float>(largest_sum_x) / largest_area;
    const float model_center_y = static_cast<float>(largest_sum_y) / largest_area;
    result.found = true;
    result.center_x = (model_center_x + 0.5F) * model_to_preview_scale - 0.5F;
    result.center_y = (model_center_y - MODEL_PAD_TOP + 0.5F) * model_to_preview_scale - 0.5F;
    result.area = largest_area;
    return result;
}

/** 向 RGB565 帧写入一个像素，自动忽略画面外坐标。 */
static void set_rgb565_pixel(camera_fb_t *frame, int x, int y, uint16_t color)
{
    if (x < 0 || x >= frame->width || y < 0 || y >= frame->height) {
        return;
    }
    const size_t byte_index = static_cast<size_t>(y * frame->width + x) * 2U;
    frame->buf[byte_index] = static_cast<uint8_t>(color >> 8U);
    frame->buf[byte_index + 1U] = static_cast<uint8_t>(color & 0xFFU);
}

/** 将最大瞳孔掩码的边缘映射回 QVGA，并绘制红色轮廓和绿色质心。 */
static void draw_pupil_contour(camera_fb_t *frame, const PupilOverlay &overlay)
{
    if (!overlay.found) {
        return;
    }

    constexpr uint16_t red = 0xF800;
    constexpr uint16_t green = 0x07E0;
    constexpr int model_to_preview_scale = PREVIEW_WIDTH / MODEL_SIZE;

    for (int model_y = MODEL_PAD_TOP; model_y < MODEL_PAD_TOP + MODEL_CONTENT_HEIGHT; ++model_y) {
        for (int model_x = 0; model_x < MODEL_SIZE; ++model_x) {
            const int index = model_y * MODEL_SIZE + model_x;
            if (largest_component_mask[index] == 0) {
                continue;
            }

            const bool left_edge = model_x == 0 || largest_component_mask[index - 1] == 0;
            const bool right_edge = model_x == MODEL_SIZE - 1 || largest_component_mask[index + 1] == 0;
            const bool top_edge = model_y == 0 || largest_component_mask[index - MODEL_SIZE] == 0;
            const bool bottom_edge =
                model_y == MODEL_SIZE - 1 || largest_component_mask[index + MODEL_SIZE] == 0;
            if (!left_edge && !right_edge && !top_edge && !bottom_edge) {
                continue;
            }

            const int preview_left = model_x * model_to_preview_scale;
            const int preview_top = (model_y - MODEL_PAD_TOP) * model_to_preview_scale;
            const int preview_right = preview_left + model_to_preview_scale - 1;
            const int preview_bottom = preview_top + model_to_preview_scale - 1;
            for (int offset = 0; offset < model_to_preview_scale; ++offset) {
                if (left_edge) {
                    set_rgb565_pixel(frame, preview_left, preview_top + offset, red);
                    set_rgb565_pixel(frame, preview_left + 1, preview_top + offset, red);
                }
                if (right_edge) {
                    set_rgb565_pixel(frame, preview_right, preview_top + offset, red);
                    set_rgb565_pixel(frame, preview_right - 1, preview_top + offset, red);
                }
                if (top_edge) {
                    set_rgb565_pixel(frame, preview_left + offset, preview_top, red);
                    set_rgb565_pixel(frame, preview_left + offset, preview_top + 1, red);
                }
                if (bottom_edge) {
                    set_rgb565_pixel(frame, preview_left + offset, preview_bottom, red);
                    set_rgb565_pixel(frame, preview_left + offset, preview_bottom - 1, red);
                }
            }
        }
    }

    const int center_x = static_cast<int>(std::lround(overlay.center_x));
    const int center_y = static_cast<int>(std::lround(overlay.center_y));
    for (int offset = -5; offset <= 5; ++offset) {
        set_rgb565_pixel(frame, center_x + offset, center_y, green);
        set_rgb565_pixel(frame, center_x, center_y + offset, green);
    }
}

/** 获取一帧、执行瞳孔定位、绘制掩码轮廓并编码为 JPEG 发送给浏览器。 */
static esp_err_t handle_capture(httpd_req_t *request)
{
    camera_fb_t *frame = esp_camera_fb_get();
    if (frame == nullptr) {
        ESP_LOGE(TAG, "OV2640 取帧失败");
        httpd_resp_send_err(request, HTTPD_500_INTERNAL_SERVER_ERROR, "camera capture failed");
        return ESP_FAIL;
    }

    if (frame->format != PIXFORMAT_RGB565 || preview_model == nullptr) {
        esp_camera_fb_return(frame);
        httpd_resp_send_err(request, HTTPD_500_INTERNAL_SERVER_ERROR, "unexpected frame format");
        return ESP_FAIL;
    }

    const int64_t inference_start_us = esp_timer_get_time();
    PupilOverlay overlay = {};
    if (prepare_camera_input(frame, preview_model->get_input())) {
        preview_model->run();
        overlay = locate_pupil_overlay(preview_model->get_output());
        draw_pupil_contour(frame, overlay);
    } else {
        ESP_LOGE(TAG, "相机帧预处理失败，本帧不绘制掩码轮廓");
    }
    const int64_t inference_elapsed_us = esp_timer_get_time() - inference_start_us;

    static uint32_t capture_count = 0;
    ++capture_count;
    if (capture_count == 1 || capture_count % 10 == 0) {
        if (overlay.found) {
            ESP_LOGI(TAG, "预览定位：帧=%" PRIu32 "，中心=(%.1f, %.1f)，区域=%" PRIu32 "，推理=%" PRId64 " us",
                     capture_count,
                     static_cast<double>(overlay.center_x),
                     static_cast<double>(overlay.center_y),
                     overlay.area,
                     inference_elapsed_us);
        } else {
            ESP_LOGW(TAG, "预览定位：帧=%" PRIu32 "，未检测到瞳孔，推理=%" PRId64 " us",
                     capture_count, inference_elapsed_us);
        }
    }

    uint8_t *jpeg_buffer = nullptr;
    size_t jpeg_size = 0;
    const bool encoded = frame2jpg(frame, 80, &jpeg_buffer, &jpeg_size);
    esp_camera_fb_return(frame);
    if (!encoded || jpeg_buffer == nullptr || jpeg_size == 0) {
        std::free(jpeg_buffer);
        httpd_resp_send_err(request, HTTPD_500_INTERNAL_SERVER_ERROR, "jpeg encode failed");
        return ESP_FAIL;
    }

    httpd_resp_set_type(request, "image/jpeg");
    httpd_resp_set_hdr(request, "Cache-Control", "no-store, no-cache, must-revalidate");
    const esp_err_t result = httpd_resp_send(request,
                                             reinterpret_cast<const char *>(jpeg_buffer),
                                             jpeg_size);
    std::free(jpeg_buffer);
    return result;
}

/** 启动网页首页和 JPEG 抓拍接口。 */
static esp_err_t initialize_http_server()
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.stack_size = 8192;
    config.max_uri_handlers = 4;
    config.lru_purge_enable = true;

    esp_err_t result = httpd_start(&server, &config);
    if (result != ESP_OK) {
        return result;
    }

    const httpd_uri_t index_uri = {
        .uri = "/",
        .method = HTTP_GET,
        .handler = handle_index,
        .user_ctx = nullptr,
    };
    const httpd_uri_t capture_uri = {
        .uri = "/capture",
        .method = HTTP_GET,
        .handler = handle_capture,
        .user_ctx = nullptr,
    };

    if ((result = httpd_register_uri_handler(server, &index_uri)) != ESP_OK ||
        (result = httpd_register_uri_handler(server, &capture_uri)) != ESP_OK) {
        httpd_stop(server);
        server = nullptr;
        return result;
    }

    ESP_LOGI(TAG, "PC 预览已就绪：http://192.168.4.1");
    return ESP_OK;
}

}  // namespace

/** 初始化 OV2640、自建 Wi-Fi 热点和 PC 浏览器预览服务。 */
esp_err_t start_camera_preview(dl::Model *model)
{
    if (model == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }
    preview_model = model;
    esp_err_t result = initialize_camera();
    if (result != ESP_OK) {
        return result;
    }
    if ((result = initialize_wifi_access_point()) != ESP_OK) {
        ESP_LOGE(TAG, "Wi-Fi 热点启动失败：%s", esp_err_to_name(result));
        return result;
    }
    if ((result = initialize_http_server()) != ESP_OK) {
        ESP_LOGE(TAG, "网页预览启动失败：%s", esp_err_to_name(result));
        return result;
    }
    return ESP_OK;
}
