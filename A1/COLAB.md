# Cháº¡y A1 trÃªn Google Colab

## 1. Clone repo

```bash
!git clone https://github.com/baond12/Deep_Learning_Course_HK261.git
%cd Deep_Learning_Course_HK261
```

## 2. CÃ i dependencies

```bash
!pip install -r requirements.txt
```

## 3. Smoke test model/shape

```bash
!python A1/smoke_test.py
```

## 4. Train cÃ¡c thÃ­ nghiá»‡m A1

Máº·c Ä‘á»‹nh dÃ¹ng CIFAR-100 subset gá»“m 20 lá»›p Ä‘áº§u tiÃªn, tá»‘i Ä‘a 500 áº£nh train/lá»›p vÃ  100 áº£nh
test/lá»›p. áº¢nh Ä‘Æ°á»£c resize vá» `224 x 224`.

```bash
!python A1/train.py --experiments all --epochs 5 --batch-size 32 --device cuda --progress
```

Náº¿u muá»‘n lÆ°u checkpoint trong phiÃªn Colab nhÆ°ng khÃ´ng push model lÃªn repo:

```bash
!python A1/train.py --experiments all --epochs 5 --batch-size 32 --device cuda --progress --save-checkpoints
```

Checkpoint sáº½ náº±m trong `A1/checkpoints/`. CÃ¡c file model `.pt/.pth/.ckpt/.safetensors`
Ä‘Ã£ Ä‘Æ°á»£c `.gitignore` cháº·n.

Náº¿u muá»‘n cháº¡y thá»­ nhanh má»™t vÃ i thÃ­ nghiá»‡m:

```bash
!python A1/train.py --experiments resnet18_scratch resnet18_pretrained_head vit_b16_pretrained_head --epochs 1 --batch-size 32 --device cuda --progress
```

## 5. Táº¡o phÃ¢n tÃ­ch A1.2

Sau khi train xong:

```bash
!python A1/analyze_results.py --results-dir A1/results
```

Script nÃ y táº¡o `analysis.md` vÃ  cÃ¡c bar chart Ä‘á»ƒ Ä‘Æ°a vÃ o bÃ¡o cÃ¡o A1.2.

## 6. Káº¿t quáº£

Káº¿t quáº£ náº±m á»Ÿ:

```text
A1/results/
```

CÃ¡c file chÃ­nh:

- `summary.csv`
- `metrics.json`
- `analysis.md`
- `learning_curves.png`
- `bar_test_accuracy.png`
- `bar_macro_f1.png`
- `bar_trainable_parameters.png`
- `bar_training_time_seconds.png`
- `confusion_matrix_*.png`
- `misclassified_*.png`
