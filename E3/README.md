# E3 - LSTM / GRU on Image Sequences

Mục tiêu: biểu diễn ảnh thành chuỗi, huấn luyện LSTM và GRU, so sánh với baseline
MLP/CNN từ E1. Repo cũng có thêm pipeline sentiment analysis dạng văn bản dùng
packed/unpacked sequences.

## Cấu trúc

- `models.py`: LSTM/GRU classifier cho ảnh MNIST dạng row sequence và patch sequence.
- `text_data.py`: toy sentiment dataset, vocabulary, padding collate function.
- `text_models.py`: LSTM/GRU sentiment classifier dùng `pack_padded_sequence` và
  `pad_packed_sequence`.
- `train.py`: train E3 image experiments và baseline MLP/CNN.
- `train_sentiment.py`: train sentiment LSTM/GRU.
- `smoke_test.py`: kiểm tra shape/logic, không train dài.
- `report.md`: bản nháp báo cáo tiếng Việt.
- `COLAB.md`: hướng dẫn chạy Colab.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Smoke test

```bash
python E3/smoke_test.py
```

## Train ảnh MNIST

```bash
python E3/train.py --experiments all --epochs 5 --batch-size 128 --device auto
```

Các experiment mặc định:

- `lstm_row`
- `gru_row`
- `lstm_patch`
- `gru_patch`
- `mlp_baseline`
- `cnn_baseline`

## Train sentiment

```bash
python E3/train_sentiment.py --rnn-type lstm --epochs 5 --device auto
python E3/train_sentiment.py --rnn-type gru --epochs 5 --device auto
```

## Kết quả đầu ra

- `E3/results/summary.csv`
- `E3/results/comparison_e1_e2_e3.csv`
- `E3/results/metrics.json`
- `E3/results/learning_curves.png`
- `E3/results/confusion_matrix_*.png`
- `E3/results/misclassified_*.png`
- `E3/results/sentiment/*.json`

