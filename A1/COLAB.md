# Chạy A1 trên Google Colab

## 1. Clone repo

```bash
!git clone https://github.com/baond12/Deee_Learning_Course_Repo.git
%cd Deee_Learning_Course_Repo
```

## 2. Cài dependencies

```bash
!pip install -r requirements.txt
```

## 3. Smoke test model/shape

```bash
!python A1/smoke_test.py
```

## 4. Train các thí nghiệm A1

Mặc định dùng CIFAR-100 subset gồm 20 lớp đầu tiên, tối đa 500 ảnh train/lớp và 100 ảnh
test/lớp. Ảnh được resize về `224 x 224`.

```bash
!python A1/train.py --experiments all --epochs 5 --batch-size 32 --device cuda --progress
```

Nếu muốn chạy thử nhanh một vài thí nghiệm:

```bash
!python A1/train.py --experiments resnet18_scratch resnet18_pretrained_head vit_b16_pretrained_head --epochs 1 --batch-size 32 --device cuda --progress
```

## 5. Kết quả

Kết quả nằm ở:

```text
A1/results/
```

Các file chính:

- `summary.csv`
- `metrics.json`
- `learning_curves.png`
- `confusion_matrix_*.png`
- `misclassified_*.png`
