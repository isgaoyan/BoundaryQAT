# BoundaryQAT ESP32-S3 板端工程

## 当前阶段

本工程已在 ESP32-S3-WROOM-1-N16R8 上通过固定图模型加载与单次推理验收。相机采集、连续推理和实施拍摄尚未完成。

当前最近邻 INT8 候选仍有 `0.6993%` 测试空预测率，仅用于工程验证，尚未定为最终发布模型。

## 发布边界

- `main/models/` 中的候选 `.espdl` 未公开，因为最终模型尚未锁定。
- `main/testdata/` 中的 LPW 固定输入未公开，因为原数据受非商业科研许可约束。
- 构建前需自行放入 `main/models/boundaryqat_pupil.espdl` 和一张合法来源的 `main/testdata/lpw__02_04_0001_gray64.bin`；若更换输入，应同步修改源码中的固定摘要。

## 构建

在 ESP-IDF PowerShell 中执行：

```powershell
cd <仓库路径>\deployment\esp32s3
idf.py set-target esp32s3
idf.py build
```

## 烧录与监视

将 `<端口>` 替换为设备管理器中实际识别到的开发板串口：

```powershell
idf.py -p <端口> flash monitor
```

按 `Ctrl+]` 退出串口监视。COM 口可能在重新插拔或更换 USB 接口后变化，烧录前应重新确认。
