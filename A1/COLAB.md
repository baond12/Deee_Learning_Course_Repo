# Chạy A1 trên Google Colab

## 1. Clone repo

```bash
!git clone https://github.com/baond12/Deep_Learning_Course_HK261.git
%cd Deep_Learning_Course_HK261
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

Nếu muốn lưu checkpoint trong phiên Colab nhưng không push model lên repo:

```bash
!python A1/train.py --experiments all --epochs 5 --batch-size 32 --device cuda --progress --save-checkpoints
```

Checkpoint sẽ nằm trong `A1/checkpoints/`. Các file model `.pt/.pth/.ckpt/.safetensors`
đã được `.gitignore` chặn.

Nếu muốn chạy thử nhanh một vài thí nghiệm:

```bash
!python A1/train.py --experiments resnet18_scratch resnet18_pretrained_head vit_b16_pretrained_head --epochs 1 --batch-size 32 --device cuda --progress
```

## 5. Tạo phân tích A1.2

Sau khi train xong:

```bash
!python A1/analyze_results.py --results-dir A1/results
```

Script này tạo `analysis.md` và các bar chart để đưa vào báo cáo A1.2.

## 6. Kết quả

Kết quả nằm ở:

```text
A1/results/
```

Các file chính:

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
