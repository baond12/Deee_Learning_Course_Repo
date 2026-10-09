# Chạy E3 trên Google Colab

## 1. Clone repo

```bash
!git clone https://github.com/baond12/Deep_Learning_Course_HK261.git
%cd Deep_Learning_Course_HK261
```

## 2. Cài dependencies

```bash
!pip install -r requirements.txt
```

## 3. Smoke test shape/logic

```bash
!python E3/smoke_test.py
```

## 4. Train ảnh MNIST dạng chuỗi

```bash
!python E3/train.py --experiments all --epochs 5 --batch-size 128 --device cuda --progress
```

Nếu không có GPU:

```bash
!python E3/train.py --experiments all --epochs 5 --batch-size 128 --device cpu --progress
```

## 5. Train sentiment analysis toy pipeline

```bash
!python E3/train_sentiment.py --rnn-type lstm --epochs 5 --device cuda
!python E3/train_sentiment.py --rnn-type gru --epochs 5 --device cuda
```

## 6. Kết quả

Kết quả ảnh nằm ở:

```text
E3/results/
```

Các file chính:

- `summary.csv`
- `comparison_e1_e2_e3.csv`
- `metrics.json`
- `learning_curves.png`
- `confusion_matrix_*.png`
- `misclassified_*.png`

Kết quả sentiment nằm ở:

```text
E3/results/sentiment/
```

