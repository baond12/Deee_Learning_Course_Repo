# Chạy E2 trên Google Colab

## 1. Clone repo

```bash
!git clone https://github.com/baond12/Deee_Learning_Course_Repo.git
%cd Deee_Learning_Course_Repo
```

## 2. Cài dependencies

```bash
!pip install -r requirements.txt
```

## 3. Train các cấu hình E2

```bash
!python E2/train.py --experiments all --epochs 5 --batch-size 128 --device cuda --progress
```

Nếu runtime không có GPU:

```bash
!python E2/train.py --experiments all --epochs 5 --batch-size 128 --device cpu --progress
```

## 4. Test nhanh code với 1 epoch

```bash
!python E2/train.py --experiments all --epochs 1 --batch-size 128 --device cuda --progress
```

## 5. Kết quả

Kết quả nằm ở:

```text
E2/results/
```

Các file quan trọng:

- `summary.csv`
- `metrics.json`
- `learning_curves.png`
- `confusion_matrix_*.png`
- `misclassified_*.png`

## 6. Đưa kết quả về repo

Cách đơn giản: tải thư mục `E2/results/` từ Colab về máy, copy vào repo local rồi chạy:

```powershell
git add E2/results E2/report.md
git commit -m "Add E2 experiment results"
git push
```

