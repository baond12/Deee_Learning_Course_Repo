# Cháº¡y E1 trÃªn Google Colab

## 1. Clone repo

```bash
!git clone https://github.com/baond12/Deep_Learning_Course_HK261.git
%cd Deep_Learning_Course_HK261
```

## 2. CÃ i dependencies

Colab thÆ°á»ng Ä‘Ã£ cÃ³ PyTorch. Náº¿u thiáº¿u package, cháº¡y:

```bash
!pip install -r requirements.txt
```

## 3. Train 3 mÃ´ hÃ¬nh

```bash
!python E1/train.py --models all --epochs 5 --batch-size 128 --device cuda --progress
```

Náº¿u runtime khÃ´ng cÃ³ GPU:

```bash
!python E1/train.py --models all --epochs 5 --batch-size 128 --device cpu --progress
```

## 4. Xem káº¿t quáº£

Káº¿t quáº£ náº±m á»Ÿ:

```text
E1/results/
```

CÃ¡c file quan trá»ng:

- `summary.csv`
- `metrics.json`
- `learning_curves.png`
- `confusion_matrix_*.png`
- `misclassified_*.png`

## 5. Commit káº¿t quáº£ tá»« Colab

```bash
!git config user.email "ngobao.bk.tp@gmail.com"
!git config user.name "baond12"
!git add E1/results E1/report.md
!git commit -m "Add E1 experiment results"
!git push
```

Náº¿u Colab yÃªu cáº§u Ä‘Äƒng nháº­p khi push, cÃ¡ch Ä‘Æ¡n giáº£n hÆ¡n lÃ  táº£i thÆ° má»¥c `E1/results/`
vá» mÃ¡y rá»“i copy vÃ o repo local, sau Ä‘Ã³ commit/push tá»« PowerShell.

