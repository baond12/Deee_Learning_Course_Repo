# A1 - CNN vs Transformer trên CIFAR-100 subset

## 1. Chủ đề

A1 so sánh CNN và Transformer trên một tập ảnh lớn hơn E1. Dataset chọn là
**CIFAR-100 subset**, không trùng MNIST/Fashion-MNIST/CIFAR-10 đã dùng cho E1.

## 2. Dataset và license

Dataset gốc: CIFAR-100, gồm 60,000 ảnh màu `32 x 32`, 100 lớp, mỗi lớp có 500 ảnh
train và 100 ảnh test. Dataset được cung cấp công khai bởi Alex Krizhevsky cho mục đích
nghiên cứu/học tập.

Để phù hợp Colab T4 16GB, A1 dùng subset có kiểm soát:

- Chọn 20 lớp đầu tiên trong CIFAR-100.
- Train: tối đa 500 ảnh/lớp từ training set gốc.
- Test: tối đa 100 ảnh/lớp từ test set gốc.
- Tách 10% training subset làm validation bằng seed `42`.

Subset nghĩa là chỉ lấy một phần dữ liệu gốc theo quy tắc cố định, có ghi rõ số lớp, số
ảnh mỗi lớp và seed để tái lập. Cách này giúp giảm thời gian train nhưng vẫn giữ bài toán
khó hơn E1 vì có nhiều lớp hơn, ảnh RGB và nội dung đa dạng hơn.

## 3. Data pipeline và augmentation

Theo slide data pipeline, test set chỉ dùng cho đánh giá cuối; validation dùng để chọn
checkpoint/cấu hình.

Transform train:

- Resize ảnh về `224 x 224`.
- Random horizontal flip.
- ToTensor.
- Normalize theo ImageNet mean/std để phù hợp pretrained models.

Transform validation/test:

- Resize `224 x 224`.
- ToTensor.
- Normalize ImageNet mean/std.

## 4. Kiến trúc

### 4.1 CNN

CNN chọn **ResNet18** từ `torchvision`. ResNet là CNN backbone nhẹ, phổ biến, có
pretrained ImageNet weights và phù hợp để so sánh các chế độ freeze/fine-tune.

### 4.2 Transformer

Transformer chọn **ViT-B/16** từ `torchvision`. ViT chuyển ảnh thành patch tokens, thêm
classification head và dùng self-attention toàn cục trên patch sequence.

## 5. Protocol thí nghiệm

Các thí nghiệm mặc định:

| Experiment | Family | Architecture | Training mode |
| --- | --- | --- | --- |
| `resnet18_scratch` | CNN | ResNet18 | From scratch |
| `resnet18_pretrained_head` | CNN | ResNet18 | Freeze backbone, train head |
| `resnet18_pretrained_partial` | CNN | ResNet18 | Freeze một phần, train layer4 + head |
| `resnet18_pretrained_full` | CNN | ResNet18 | Full fine-tune |
| `vit_b16_scratch` | Transformer | ViT-B/16 | From scratch |
| `vit_b16_pretrained_head` | Transformer | ViT-B/16 | Freeze backbone, train head |
| `vit_b16_pretrained_partial` | Transformer | ViT-B/16 | Freeze một phần, train 2 encoder blocks cuối + head |
| `vit_b16_pretrained_full` | Transformer | ViT-B/16 | Full fine-tune |

## 6. Cấu hình train mặc định

- Epochs: 5.
- Batch size: 32.
- Optimizer: Adam.
- Learning rate: `1e-4`.
- Weight decay: `1e-4`.
- Loss: CrossEntropyLoss.
- Metric: accuracy, macro F1, số tham số total/trainable, training time.

## 7. Lệnh chạy

```bash
python A1/train.py --experiments all --epochs 5 --batch-size 32 --device auto
```

Trên Google Colab GPU:

```bash
python A1/train.py --experiments all --epochs 5 --batch-size 32 --device cuda --progress
```

Smoke test không tải dataset/weights:

```bash
python A1/smoke_test.py
```

## 8. Kết quả cần điền sau khi train

Sau khi train, kết quả nằm trong `A1/results/`:

- `summary.csv`: bảng so sánh chính.
- `metrics.json`: cấu hình và metric chi tiết.
- `learning_curves.png`: loss/accuracy curves.
- `confusion_matrix_*.png`: confusion matrix từng experiment.
- `misclassified_*.png`: mẫu dự đoán sai.

## 9. Nhận xét dự kiến

ResNet18 có inductive bias không gian mạnh nên thường học tốt hơn khi dữ liệu không quá
lớn. ViT-B/16 có attention toàn cục và thường cần pretrained weights hoặc dữ liệu lớn để
phát huy tốt. Freeze-head thường nhanh và ít tốn bộ nhớ nhất nhưng khả năng thích nghi
hạn chế; partial unfreeze cân bằng giữa chi phí và độ thích nghi; full fine-tune có khả
năng đạt kết quả tốt hơn nhưng tốn compute và dễ overfit hơn trên subset nhỏ.
