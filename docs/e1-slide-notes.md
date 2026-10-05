# E1 slide notes

Các điểm được dùng để thiết kế E1, trích từ slide môn học:

- `03-cnn/09-Augmentation.pdf`, trang 8: train set dùng để học tham số, validation
  set dùng để tune hyperparameters/chọn checkpoint, test set chỉ dùng cho đánh giá cuối.
- `03-cnn/09-Augmentation.pdf`, trang 9: tránh data leakage; không tạo biến thể dữ
  liệu rồi mới split.
- `02-foundation/05-Losses.pdf`, trang 21-23: multiclass classification dùng
  cross-entropy; softmax + CE tương đương log loss và tối đa hóa xác suất lớp đúng.
- `02-foundation/06-Metrics.pdf`, trang 12-22: classification nên báo cáo accuracy,
  F1 và confusion matrix; metric thể hiện chi phí của lỗi.
- `02-foundation/08-MLP.pdf`, trang 7 và 11-16: softmax regression là mô hình tuyến
  tính cho multiclass; MLP học nonlinear feature transformer trước classification head.
- `03-cnn/01-Conv2d-Overview.pdf`, trang 4-5 và 39: Conv2D khai thác locality,
  weight sharing và tạo hierarchical representations.
- `03-cnn/06-Pool-Dropout.pdf`, trang 4-5: pooling giảm kích thước spatial; dropout
  regularize và giảm overfitting.

Thiết kế E1 theo các điểm trên:

- Dataset MNIST, giữ test set riêng, tách 10% training set làm validation với seed cố định.
- Ba model: linear softmax baseline, MLP flatten, CNN dùng Conv/ReLU/MaxPool.
- Dùng `nn.CrossEntropyLoss` trên logits, tự viết training loop.
- Lưu loss/accuracy curves, confusion matrix và mẫu sai cho từng model.

