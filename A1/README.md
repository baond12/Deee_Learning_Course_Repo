# Assignment 1 - CNN vs Transformer

Mục tiêu: so sánh CNN và Transformer trên **CIFAR-100 subset** với các chế độ
from-scratch, pretrained freeze backbone/head-only, pretrained partial unfreeze và full
fine-tune.

## Cấu trúc

- `data.py`: CIFAR-100 subset, split train/validation/test, augmentation.
- `models.py`: ResNet18 và ViT-B/16 với các chế độ freeze/fine-tune.
- `train.py`: training loop A1, lưu metric và hình phân tích.
- `analyze_results.py`: tổng hợp `summary.csv`, tạo bar charts và `analysis.md`.
- `smoke_test.py`: kiểm tra model shape/trainable params, không tải weights/dataset.
- `report.md`: bản nháp A1.1/A1.2.
- `COLAB.md`: hướng dẫn chạy trên Google Colab.
- `checkpoints/`: nơi lưu checkpoint tùy chọn; model files bị Git ignore.
- `results/`: thư mục sinh kết quả sau khi train.

## Dataset

Mặc định dùng 20 lớp đầu tiên của CIFAR-100:

- Tối đa 500 ảnh train/lớp.
- Tối đa 100 ảnh test/lớp.
- 10% training subset làm validation.
- Resize ảnh về `224 x 224`.

Subset là một phần dữ liệu được lấy theo quy tắc cố định để phù hợp GPU/Colab, nhưng
vẫn phải ghi rõ protocol trong báo cáo.

## Smoke test

```bash
python A1/smoke_test.py
```

## Train

```bash
python A1/train.py --experiments all --epochs 5 --batch-size 32 --device auto
```

Trên Colab GPU:

```bash
python A1/train.py --experiments all --epochs 5 --batch-size 32 --device cuda --progress
```

Nếu muốn lưu checkpoint trên Colab/local nhưng không đẩy model lên GitHub:

```bash
python A1/train.py --experiments all --epochs 5 --batch-size 32 --device cuda --progress --save-checkpoints
```

Các file `.pt`, `.pth`, `.ckpt`, `.safetensors` trong `A1/checkpoints/` đã được ignore.

## Phân tích A1.2

Sau khi train xong:

```bash
python A1/analyze_results.py --results-dir A1/results
```

Script sinh:

- `A1/results/analysis.md`
- `A1/results/bar_test_accuracy.png`
- `A1/results/bar_macro_f1.png`
- `A1/results/bar_trainable_parameters.png`
- `A1/results/bar_training_time_seconds.png`

## Experiments

- `resnet18_scratch`
- `resnet18_pretrained_head`
- `resnet18_pretrained_partial`
- `resnet18_pretrained_full`
- `vit_b16_scratch`
- `vit_b16_pretrained_head`
- `vit_b16_pretrained_partial`
- `vit_b16_pretrained_full`

## Kết quả đầu ra

- `A1/results/summary.csv`
- `A1/results/metrics.json`
- `A1/results/analysis.md`
- `A1/results/learning_curves.png`
- `A1/results/bar_*.png`
- `A1/results/confusion_matrix_*.png`
- `A1/results/misclassified_*.png`
