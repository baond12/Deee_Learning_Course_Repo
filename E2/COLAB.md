# Cháº¡y E2 trÃªn Google Colab

## 1. Clone repo

```bash
!git clone https://github.com/baond12/Deep_Learning_Course_HK261.git
%cd Deep_Learning_Course_HK261
```

## 2. CÃ i dependencies

```bash
!pip install -r requirements.txt
```

## 3. Train cÃ¡c cáº¥u hÃ¬nh E2

```bash
!python E2/train.py --experiments all --epochs 5 --batch-size 128 --device cuda --progress
```

Náº¿u runtime khÃ´ng cÃ³ GPU:

```bash
!python E2/train.py --experiments all --epochs 5 --batch-size 128 --device cpu --progress
```

## 4. Test nhanh code vá»›i 1 epoch

```bash
!python E2/train.py --experiments all --epochs 1 --batch-size 128 --device cuda --progress
```

## 5. Káº¿t quáº£

Káº¿t quáº£ náº±m á»Ÿ:

```text
E2/results/
```

CÃ¡c file quan trá»ng:

- `summary.csv`
- `metrics.json`
- `learning_curves.png`
- `confusion_matrix_*.png`
- `misclassified_*.png`

## 6. ÄÆ°a káº¿t quáº£ vá» repo

CÃ¡ch Ä‘Æ¡n giáº£n: táº£i thÆ° má»¥c `E2/results/` tá»« Colab vá» mÃ¡y, copy vÃ o repo local rá»“i cháº¡y:

```powershell
git add E2/results E2/report.md
git commit -m "Add E2 experiment results"
git push
```

