# E2 - Transformer: MSA và tokenization

## 1. Mục tiêu

Bài E2 hiện thực Multi-Head Self-Attention (MSA) từ đầu và đối chiếu với
`nn.MultiheadAttention` của PyTorch. Các mô hình được đưa vào một classifier ảnh nhỏ
trên cùng dataset MNIST đã dùng ở E1.

Các mục tiêu chính:

- Tự viết MSA bằng Linear, phép nhân ma trận, softmax, split/concat heads.
- Có bản dùng PyTorch MSA để đối chiếu.
- Thử ít nhất hai cách lấy token từ ảnh.
- So sánh accuracy, macro F1, số tham số, thời gian train và learning curves.

## 2. Dữ liệu và split

Dataset: **MNIST**, ảnh xám kích thước `1 x 28 x 28`, 10 lớp chữ số.

Split giữ giống E1:

- Train: 54,000 ảnh.
- Validation: 6,000 ảnh, tách từ training set bằng seed `42`.
- Test: 10,000 ảnh gốc của MNIST, chỉ dùng để đánh giá cuối.

Tiền xử lý:

- `ToTensor()`.
- Normalize với `mean = 0.1307`, `std = 0.3081`.

## 3. Tokenization

### 3.1 Patch tokens

Ảnh MNIST `28 x 28` được chia thành các patch `7 x 7`, tạo ra `4 x 4 = 16` tokens.
Patch embedding được cài bằng `Conv2d(1, embed_dim, kernel_size=7, stride=7)`, sau đó
flatten thành tensor dạng `(B, 16, d_model)`.

### 3.2 Row tokens

Mỗi hàng ảnh là một token. Ảnh `28 x 28` tạo ra 28 tokens, mỗi token ban đầu có 28
giá trị pixel, sau đó được chiếu bằng `Linear(28, d_model)`.

## 4. Kiến trúc

Cấu hình mặc định:

- `d_model = 64`.
- `num_heads = 4`, nên mỗi head có `d_k = 16`.
- `d_ff = 128`.
- `depth = 1` transformer encoder block.
- Có learnable `[CLS]` token.
- Có learnable absolute positional embedding.
- Classification head lấy representation của `[CLS]`.

Encoder block dùng pre-norm:

```text
x -> LayerNorm -> MSA -> residual
  -> LayerNorm -> FFN -> residual
```

## 5. MSA tự viết

Với input `X` có shape `(B, n, d_model)`, MSA tự viết thực hiện:

```text
Q, K, V = Linear(X)
Q, K, V -> split thành h heads
Attention = softmax(QK^T / sqrt(d_k))
Y = Attention V
Output = Concat(heads) W_o
```

Bản này không dùng `nn.MultiheadAttention` hoặc `nn.TransformerEncoder`.

## 6. Thí nghiệm

Ba cấu hình mặc định:

| Experiment | MSA | Tokenization | Mục đích |
| --- | --- | --- | --- |
| `custom_patch` | Tự viết | Patch `7 x 7` | Mô hình chính |
| `pytorch_patch` | `nn.MultiheadAttention` | Patch `7 x 7` | Đối chiếu MSA tự viết vs PyTorch |
| `custom_row` | Tự viết | Row tokens | So sánh tokenization patch vs row |

Lệnh chạy:

```bash
python E2/train.py --experiments all --epochs 5 --batch-size 128 --device auto
```

Trên Google Colab có GPU:

```bash
python E2/train.py --experiments all --epochs 5 --batch-size 128 --device cuda
```

## 7. Kết quả cần điền sau khi chạy

Kết quả được lưu trong `E2/results/`:

- `summary.csv`: bảng accuracy, macro F1, số tham số, thời gian train.
- `metrics.json`: cấu hình và metric chi tiết.
- `learning_curves.png`: loss/accuracy curves.
- `confusion_matrix_custom_patch.png`, `confusion_matrix_pytorch_patch.png`,
  `confusion_matrix_custom_row.png`.
- `misclassified_custom_patch.png`, `misclassified_pytorch_patch.png`,
  `misclassified_custom_row.png`.

## 8. Nhận xét dự kiến

`custom_patch` và `pytorch_patch` dùng cùng cách tokenization nên dùng để kiểm tra bản
MSA tự viết có hành vi hợp lý so với module PyTorch. Nếu kết quả gần nhau, bản tự viết
có thể xem là đúng về mặt thực nghiệm.

Patch tokens có chuỗi ngắn hơn row tokens (`16 + CLS` so với `28 + CLS`), nên thường
nhẹ hơn về chi phí attention. Row tokens giữ cấu trúc theo chiều dọc của ảnh và có thể
cho kết quả khác tùy cách mô hình khai thác quan hệ giữa các hàng.

Khi có số liệu thật, cần bổ sung:

- Bảng kết quả cuối từ `summary.csv`.
- Nhận xét custom MSA vs PyTorch MSA trên patch tokens.
- Nhận xét patch tokenization vs row tokenization.
- Phân tích confusion matrix và mẫu dự đoán sai.

