# E1 - Phân loại ảnh: Softmax, MLP, CNN

## 1. Mục tiêu

Bài E1 xây dựng và so sánh ba bộ phân loại trên cùng một tập ảnh nhỏ:

- Softmax classifier trên vector ảnh flatten.
- MLP classifier trên vector ảnh flatten.
- CNN classifier dùng convolution và pooling.

Các mô hình đều dùng PyTorch, tự viết training loop theo chuỗi
`forward -> loss -> backward() -> optimizer.step()`.

## 2. Dữ liệu và split

Dataset chọn cho E1-E3 là **MNIST**, gồm ảnh xám kích thước `1 x 28 x 28` và 10 lớp
chữ số từ 0 đến 9.

Theo nguyên tắc trong slide data pipeline:

- Train set dùng để học tham số.
- Validation set dùng để chọn epoch tốt nhất và tune cấu hình.
- Test set chỉ dùng cho đánh giá cuối cùng.

Split mặc định trong code:

- Train: 54,000 ảnh.
- Validation: 6,000 ảnh, tách từ training set với seed cố định `42`.
- Test: 10,000 ảnh gốc của MNIST.

Tiền xử lý gồm `ToTensor()` và normalize bằng mean/std chuẩn của MNIST:
`mean = 0.1307`, `std = 0.3081`.

## 3. Mô hình

### 3.1 Softmax classifier

Ảnh được flatten thành vector 784 chiều rồi đưa qua một lớp tuyến tính `784 -> 10`.
Model trả về logits; `nn.CrossEntropyLoss` tự áp dụng log-softmax bên trong để ổn định
số học.

### 3.2 MLP classifier

MLP dùng hai tầng ẩn:

- `784 -> 256 -> 128 -> 10`
- Activation: ReLU.
- Dropout: 0.2 ở các tầng ẩn.

### 3.3 CNN classifier

CNN dùng hai block convolution đơn giản:

- `Conv2d(1, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)`
- `Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool2d(2)`
- Head phân loại: `64*7*7 -> 128 -> 10`, ReLU và Dropout 0.3.

CNN tận dụng locality và weight sharing, phù hợp hơn với ảnh so với flatten toàn bộ đầu
vào.

## 4. Huấn luyện

Cấu hình mặc định:

- Epochs: 5.
- Batch size: 128.
- Optimizer: Adam.
- Learning rate: `1e-3`.
- Weight decay: `1e-4`.
- Loss: CrossEntropyLoss.
- Device: tự động chọn CUDA nếu có, nếu không dùng CPU.

Lệnh chạy:

```bash
python E1/train.py --models all --epochs 5 --batch-size 128 --device auto
```

Trên Google Colab:

```bash
git clone https://github.com/baond12/Deee_Learning_Course_Repo.git
cd Deee_Learning_Course_Repo
pip install -r requirements.txt
python E1/train.py --models all --epochs 5 --batch-size 128 --device cuda
```

## 5. Kết quả cần điền sau khi chạy

Sau khi chạy script, kết quả nằm trong `E1/results/`:

- `summary.csv`: bảng so sánh số tham số, best validation accuracy, test accuracy,
  macro F1 và thời gian train.
- `metrics.json`: metadata và metric chi tiết.
- `learning_curves.png`: đường cong loss/accuracy.
- `confusion_matrix_softmax.png`, `confusion_matrix_mlp.png`,
  `confusion_matrix_cnn.png`.
- `misclassified_softmax.png`, `misclassified_mlp.png`, `misclassified_cnn.png`.

## 6. Nhận xét dự kiến

Softmax classifier có ít tham số nhất và tạo baseline tuyến tính. MLP tăng capacity nhờ
các tầng phi tuyến, nên thường tốt hơn softmax trên cùng vector flatten. CNN thường đạt
kết quả tốt nhất vì giữ cấu trúc không gian của ảnh, dùng local receptive field và weight
sharing để học đặc trưng ảnh hiệu quả hơn.

Khi có số liệu thực nghiệm, cần bổ sung:

- Bảng kết quả cuối từ `summary.csv`.
- So sánh learning curves để nhận xét underfitting/overfitting.
- Phân tích confusion matrix: các cặp chữ số dễ nhầm.
- Một số mẫu dự đoán sai tiêu biểu.

