#include <inttypes.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

#include "dl_model_base.hpp"
#include "esp_chip_info.h"
#include "esp_err.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_psram.h"
#include "esp_timer.h"

static const char *TAG = "固定模型验收";
static constexpr size_t BOUNDARYQAT_MODEL_SIZE = 497376;
static constexpr size_t PROBE_PIXEL_COUNT = 64U * 64U;
static constexpr float SEGMENTATION_THRESHOLD = 0.14F;

extern const uint8_t probe_data_start[] asm("_binary_lpw__02_04_0001_gray64_bin_start");
extern const uint8_t probe_data_end[] asm("_binary_lpw__02_04_0001_gray64_bin_end");

/** 打印 ESP32-S3、Flash 与 PSRAM 的实测信息。 */
static void print_hardware_info()
{
    esp_chip_info_t chip_info = {};
    uint32_t flash_size = 0;

    esp_chip_info(&chip_info);
    ESP_ERROR_CHECK(esp_flash_get_size(nullptr, &flash_size));
    ESP_LOGI(TAG, "芯片：ESP32-S3，修订版本：v%d.%d，CPU 核心：%d",
             chip_info.revision / 100,
             chip_info.revision % 100,
             chip_info.cores);
    ESP_LOGI(TAG, "Flash：%" PRIu32 " MB", flash_size / (1024U * 1024U));

    if (esp_psram_is_initialized()) {
        ESP_LOGI(TAG, "PSRAM：已初始化，容量：%u MB",
                 static_cast<unsigned int>(esp_psram_get_size() / (1024U * 1024U)));
    } else {
        ESP_LOGE(TAG, "PSRAM：初始化失败，停止模型验收");
    }
}

/** 打印内部 RAM 与 PSRAM 的当前可分配容量。 */
static void print_memory_info()
{
    ESP_LOGI(TAG, "内部 RAM：可用 %u KB / 总计 %u KB",
             static_cast<unsigned int>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024U),
             static_cast<unsigned int>(heap_caps_get_total_size(MALLOC_CAP_INTERNAL) / 1024U));
    ESP_LOGI(TAG, "PSRAM 堆：可用 %u KB / 总计 %u KB",
             static_cast<unsigned int>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024U),
             static_cast<unsigned int>(heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / 1024U));
}

/** 打印模型输入输出张量，并返回模型结构是否满足单输入单输出要求。 */
static bool print_model_io_info(dl::Model &model)
{
    auto &inputs = model.get_inputs();
    auto &outputs = model.get_outputs();
    if (inputs.size() != 1 || outputs.size() != 1) {
        ESP_LOGE(TAG, "模型必须恰好包含一个输入和一个输出，当前输入 %u 个、输出 %u 个",
                 static_cast<unsigned int>(inputs.size()),
                 static_cast<unsigned int>(outputs.size()));
        return false;
    }

    for (const auto &[name, tensor] : inputs) {
        const std::string shape = dl::vector_to_string(tensor->get_shape());
        ESP_LOGI(TAG, "输入张量：%s，形状：%s，类型：%s，量化指数：%d",
                 name.c_str(), shape.c_str(), tensor->get_dtype_string(), tensor->get_exponent());
    }
    for (const auto &[name, tensor] : outputs) {
        const std::string shape = dl::vector_to_string(tensor->get_shape());
        ESP_LOGI(TAG, "输出张量：%s，形状：%s，类型：%s，量化指数：%d",
                 name.c_str(), shape.c_str(), tensor->get_dtype_string(), tensor->get_exponent());
    }
    return true;
}

/** 对字节序列计算稳定的 FNV-1a 摘要，便于核对重复推理结果。 */
static uint32_t calculate_fnv1a(const uint8_t *data, size_t size)
{
    uint32_t hash = 2166136261U;
    for (size_t index = 0; index < size; ++index) {
        hash ^= data[index];
        hash *= 16777619U;
    }
    return hash;
}

/** 将固定灰度探针按训练口径归一化并写入模型 INT8 输入。 */
static bool prepare_probe_input(dl::TensorBase *input)
{
    const size_t probe_size = static_cast<size_t>(probe_data_end - probe_data_start);
    if (input == nullptr || input->get_dtype() != dl::DATA_TYPE_INT8 ||
        input->get_size() != static_cast<int>(PROBE_PIXEL_COUNT) || probe_size != PROBE_PIXEL_COUNT) {
        ESP_LOGE(TAG, "固定输入与模型输入不匹配：探针 %u 字节，张量 %d 个元素",
                 static_cast<unsigned int>(probe_size), input == nullptr ? 0 : input->get_size());
        return false;
    }

    const float input_scale = std::ldexp(1.0F, input->get_exponent());
    int8_t *input_data = input->get_element_ptr<int8_t>();
    int64_t quantized_sum = 0;
    for (size_t index = 0; index < PROBE_PIXEL_COUNT; ++index) {
        const float normalized = static_cast<float>(probe_data_start[index]) / 255.0F;
        const int quantized = std::clamp(static_cast<int>(std::lround(normalized / input_scale)), -128, 127);
        input_data[index] = static_cast<int8_t>(quantized);
        quantized_sum += quantized;
    }

    ESP_LOGI(TAG, "固定输入：lpw__02_04_0001，原始灰度和：435249，原始 FNV：0x%08" PRIX32,
             calculate_fnv1a(probe_data_start, probe_size));
    ESP_LOGI(TAG, "固定输入量化完成：scale=%.8f，INT8 和=%" PRId64 "，INT8 FNV=0x%08" PRIX32,
             input_scale,
             quantized_sum,
             calculate_fnv1a(reinterpret_cast<const uint8_t *>(input_data), PROBE_PIXEL_COUNT));
    return true;
}

/** 汇总单次推理输出，并打印阈值掩码与瞳孔中心。 */
static bool print_inference_summary(dl::TensorBase *output, int64_t elapsed_us)
{
    if (output == nullptr || output->get_dtype() != dl::DATA_TYPE_INT8 ||
        output->get_size() != static_cast<int>(PROBE_PIXEL_COUNT)) {
        ESP_LOGE(TAG, "模型输出不是预期的 64×64 INT8 张量");
        return false;
    }

    const float output_scale = std::ldexp(1.0F, output->get_exponent());
    const int threshold_int8 = std::clamp(
        static_cast<int>(std::ceil(SEGMENTATION_THRESHOLD / output_scale)), -128, 127);
    const int8_t *output_data = output->get_element_ptr<int8_t>();
    int minimum = 127;
    int maximum = -128;
    int64_t output_sum = 0;
    uint32_t foreground_count = 0;
    uint32_t coordinate_sum_x = 0;
    uint32_t coordinate_sum_y = 0;

    for (size_t index = 0; index < PROBE_PIXEL_COUNT; ++index) {
        const int value = output_data[index];
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        output_sum += value;
        if (value >= threshold_int8) {
            ++foreground_count;
            coordinate_sum_x += static_cast<uint32_t>(index % 64U);
            coordinate_sum_y += static_cast<uint32_t>(index / 64U);
        }
    }

    ESP_LOGI(TAG, "固定输入推理完成：耗时=%" PRId64 " us，输出 scale=%.8f，阈值 INT8=%d",
             elapsed_us, output_scale, threshold_int8);
    ESP_LOGI(TAG, "输出摘要：min=%d，max=%d，sum=%" PRId64 "，FNV=0x%08" PRIX32,
             minimum,
             maximum,
             output_sum,
             calculate_fnv1a(reinterpret_cast<const uint8_t *>(output_data), PROBE_PIXEL_COUNT));
    if (foreground_count == 0) {
        ESP_LOGW(TAG, "阈值掩码为空：前景像素=0");
        return true;
    }

    ESP_LOGI(TAG, "阈值掩码：前景像素=%" PRIu32 "，中心=(%.3f, %.3f)",
             foreground_count,
             static_cast<double>(coordinate_sum_x) / foreground_count,
             static_cast<double>(coordinate_sum_y) / foreground_count);
    return true;
}

/** 执行一次固定输入推理并输出可复核摘要。 */
static bool run_fixed_probe(dl::Model &model)
{
    if (!prepare_probe_input(model.get_input())) {
        return false;
    }

    const int64_t start_us = esp_timer_get_time();
    model.run();
    const int64_t elapsed_us = esp_timer_get_time() - start_us;
    return print_inference_summary(model.get_output(), elapsed_us);
}

/** 启动 BoundaryQAT 板端模型加载与固定输入推理验收。 */
extern "C" void app_main()
{
    ESP_LOGI(TAG, "BoundaryQAT 固定模型与固定输入验收启动");
    print_hardware_info();
    print_memory_info();
    ESP_LOGI(TAG, "模型分区有效数据：%u 字节", static_cast<unsigned int>(BOUNDARYQAT_MODEL_SIZE));
    if (!esp_psram_is_initialized()) {
        return;
    }

    dl::Model model("model",
                    fbs::MODEL_LOCATION_IN_FLASH_PARTITION,
                    0,
                    dl::MEMORY_MANAGER_GREEDY,
                    nullptr,
                    true);
    if (!print_model_io_info(model)) {
        return;
    }

    model.profile_memory();
    print_memory_info();
    if (!run_fixed_probe(model)) {
        ESP_LOGE(TAG, "固定输入推理验收失败");
        return;
    }

    print_memory_info();
    ESP_LOGI(TAG, "固定输入推理验收通过");
}
