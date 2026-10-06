# E2 - Multi-Head Self-Attention and Tokenization

Mục tiêu: hiện thực MSA từ đầu, đối chiếu với `nn.MultiheadAttention` của PyTorch và
thử nhiều cách tokenize ảnh trên MNIST.

## Cấu trúc

- `models.py`: patch tokenizer, row tokenizer, custom MSA, PyTorch MSA wrapper và
  image transformer classifier.
- `train.py`: training loop, đánh giá, lưu metric và hình phân tích.
- `report.md`: bản nháp báo cáo tiếng Việt cho phần E2.
- `COLAB.md`: hướng dẫn chạy E2 trên Google Colab.
- `results/`: thư mục sinh ra sau khi chạy training.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy local hoặc Colab

```bash
python E2/train.py --experiments all --epochs 5 --batch-size 128 --device auto
```

Nếu chạy Google Colab có GPU:

```bash
python E2/train.py --experiments all --epochs 5 --batch-size 128 --device cuda --progress
```

## Các thí nghiệm mặc định

- `custom_patch`: MSA tự viết + patch tokens `7 x 7`.
- `pytorch_patch`: `nn.MultiheadAttention` + patch tokens `7 x 7`.
- `custom_row`: MSA tự viết + row tokens.

## Kết quả đầu ra

Script lưu vào `E2/results/`:

- `summary.csv`
- `metrics.json`
- `learning_curves.png`
- `history_custom_patch.csv`, `history_pytorch_patch.csv`, `history_custom_row.csv`
- `confusion_matrix_*.png`
- `misclassified_*.png`

Sau khi có kết quả, cập nhật bảng và nhận xét vào `E2/report.md`, rồi gom vào
`reports/exercises.pdf` khi làm báo cáo chung E1-E3.

