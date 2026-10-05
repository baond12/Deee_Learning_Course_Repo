# E2 - Multi-Head Self-Attention and Tokenization

Mục tiêu: hiện thực MSA từ đầu, đối chiếu với module PyTorch và thử nhiều cách
tokenize ảnh.

## Checklist

- MSA tự viết bằng Linear, matrix multiplication/einsum, softmax và ghép head.
- MSA bằng `nn.MultiheadAttention` hoặc `TransformerEncoderLayer`.
- Ít nhất hai cách tokenization.
- So sánh accuracy, thời gian và nhận xét trong `reports/exercises.pdf`.

