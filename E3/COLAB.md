# Cháº¡y E3 trÃªn Google Colab

## 1. Clone repo

```bash
!git clone https://github.com/baond12/Deep_Learning_Course_HK261.git
%cd Deep_Learning_Course_HK261
```

## 2. CÃ i dependencies

```bash
!pip install -r requirements.txt
```

## 3. Smoke test shape/logic

```bash
!python E3/smoke_test.py
```

## 4. Train áº£nh MNIST dáº¡ng chuá»—i

```bash
!python E3/train.py --experiments all --epochs 5 --batch-size 128 --device cuda --progress
```

Náº¿u khÃ´ng cÃ³ GPU:

```bash
!python E3/train.py --experiments all --epochs 5 --batch-size 128 --device cpu --progress
```

## 5. Train sentiment analysis toy pipeline

```bash
!python E3/train_sentiment.py --rnn-type lstm --epochs 5 --device cuda
!python E3/train_sentiment.py --rnn-type gru --epochs 5 --device cuda
```

## 6. Káº¿t quáº£

Káº¿t quáº£ áº£nh náº±m á»Ÿ:

```text
E3/results/
```

CÃ¡c file chÃ­nh:

- `summary.csv`
- `comparison_e1_e2_e3.csv`
- `metrics.json`
- `learning_curves.png`
- `confusion_matrix_*.png`
- `misclassified_*.png`

Káº¿t quáº£ sentiment náº±m á»Ÿ:

```text
E3/results/sentiment/
```

