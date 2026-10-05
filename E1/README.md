# E1 - Softmax, MLP, CNN

Mục tiêu: xây dựng và so sánh ba bộ phân loại ảnh nhỏ gồm softmax classifier, MLP và
CNN trên MNIST.

## Cấu trúc

- `models.py`: định nghĩa SoftmaxRegression, MLPClassifier và SmallCNN.
- `train.py`: training loop, đánh giá, lưu bảng metric và hình phân tích.
- `report.md`: bản nháp báo cáo tiếng Việt cho phần E1.
- `results/`: thư mục sinh ra sau khi chạy training.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy local hoặc Colab

```bash
python E1/train.py --models all --epochs 5 --batch-size 128 --device auto
```

Nếu chạy Google Colab có GPU:

```bash
python E1/train.py --models all --epochs 5 --batch-size 128 --device cuda
```

## Kết quả đầu ra

Script lưu vào `E1/results/`:

- `summary.csv`
- `metrics.json`
- `learning_curves.png`
- `history_softmax.csv`, `history_mlp.csv`, `history_cnn.csv`
- `confusion_matrix_*.png`
- `misclassified_*.png`

Sau khi có kết quả, cập nhật bảng và nhận xét vào `E1/report.md`, rồi gom vào
`reports/exercises.pdf` khi làm báo cáo chung E1-E3.

