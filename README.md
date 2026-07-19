# BoundaryQAT: INT8 Pupil Segmentation on ESP32-S3

> **Status:** fixed-image inference has been validated on an ESP32-S3; camera integration and the implementation demo shoot are still pending. The deployed nearest-neighbor INT8 candidate remains an engineering candidate because its test empty-prediction rate is `0.6993%`.

[中文说明](README-zh.md) · [Current progress (Chinese)](docs/当前进度.md) · [Data and licensing](docs/数据来源与许可.md)

## Goal

Build a reproducible edge-vision pipeline from public eye images through lightweight segmentation, hardware-compatible INT8 quantization, `.espdl` export, and on-device inference.

## Verified results

| Stage | Result |
|---|---|
| 128×128 FP32 baseline | Dice `0.9021`, Boundary IoU `0.6474`, center MAE `1.467 px` |
| 64×64 bilinear INT8 | Dice `0.8448`, Boundary IoU `0.5028`, center MAE `1.415 px`, empty rate `0%` |
| 64×64 nearest INT8 | Dice `0.8632`, Boundary IoU `0.5058`, center MAE `1.335 px`, empty rate `0.6993%` |
| ESP32-S3 fixed probe | `377260 us` for one inference; board/PC center difference about `0.212 px` |
| Board resources | model about `485.70 KB`; about `619.08 KB` PSRAM after loading |

The board result is a fixed-probe consistency check, not a full test-set benchmark or a real-time claim.

## Repository scope

The repository includes experiment configurations, data preparation, training, evaluation, quantization search, automated tests, technical reports, and the ESP-IDF deployment source used for board validation.

Raw/processed datasets, run directories, device flash backups, failed early exports, personal learning material, the licensed probe image, and non-final model binaries are intentionally excluded.

## Verify

```bash
pytest -q
```

Current result: `45 passed`.

## Known limitations

- The nearest-neighbor candidate fixes the ESP-DL Resize compatibility issue but misses the original zero-empty-prediction requirement.
- Continuous inference, OV2640 capture/preprocessing, and end-to-end frame-rate measurements are pending.
- The available OV2640 is visible-light; infrared-domain validation still requires an independently collected and annotated set.
- Implementation photos and video have not yet been recorded.

## License

Source code is released under the MIT License. Datasets and third-party components retain their own terms; see the [data and licensing notes](docs/数据来源与许可.md).
