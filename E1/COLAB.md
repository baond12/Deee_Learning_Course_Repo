# Chạy E1 trên Google Colab

## 1. Clone repo

```bash
!git clone https://github.com/baond12/Deep_Learning_Course_HK261.git
%cd Deep_Learning_Course_HK261
```

## 2. Cài dependencies

Colab thường đã có PyTorch. Nếu thiếu package, chạy:

```bash
!pip install -r requirements.txt
```

## 3. Train 3 mô hình

```bash
!python E1/train.py --models all --epochs 5 --batch-size 128 --device cuda --progress
```

Nếu runtime không có GPU:

```bash
!python E1/train.py --models all --epochs 5 --batch-size 128 --device cpu --progress
```

## 4. Xem kết quả

Kết quả nằm ở:

```text
E1/results/
```

Các file quan trọng:

- `summary.csv`
- `metrics.json`
- `learning_curves.png`
- `confusion_matrix_*.png`
- `misclassified_*.png`

## 5. Commit kết quả từ Colab

```bash
!git config user.email "ngobao.bk.tp@gmail.com"
!git config user.name "baond12"
!git add E1/results E1/report.md
!git commit -m "Add E1 experiment results"
!git push
```

Nếu Colab yêu cầu đăng nhập khi push, cách đơn giản hơn là tải thư mục `E1/results/`
về máy rồi copy vào repo local, sau đó commit/push từ PowerShell.

