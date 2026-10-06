# E2 slide notes

Các điểm từ slide được dùng để thiết kế E2:

- `05-transformers/01-Positional-Mask-Training-Basics.pdf`, trang 5-8: token là đơn vị
  mà attention đọc trực tiếp; với ảnh có thể dùng patch tokens hoặc tokenizer thị giác
  khác.
- `05-transformers/01-Positional-Mask-Training-Basics.pdf`, trang 11: input block đầu
  thường là token embedding cộng positional embedding.
- `05-transformers/02-Attention-Transformer-Core.pdf`, trang 7-9: transformer giữ
  shape chính `(B, n, d_model)`; Q/K/V và head chỉ là tensor trung gian.
- `05-transformers/02-Attention-Transformer-Core.pdf`, trang 11-12: scaled dot-product
  attention là `softmax(QK^T / sqrt(d_k))V`, mỗi hàng attention là một phân phối trên
  các token.
- `05-transformers/02-Attention-Transformer-Core.pdf`, trang 17-19: MHSA gồm project
  Q/K/V, split thành nhiều heads, attention độc lập từng head, concat và output projection.
- `05-transformers/02-Attention-Transformer-Core.pdf`, trang 21-23: transformer block
  gồm MHA, LayerNorm, FFN và residual; pre-norm thường dễ tối ưu hơn.
- `05-transformers/03-Model-Families-BERT-GPT-T5-ViT.pdf`, trang 29-32: ViT chuyển ảnh
  thành chuỗi patch tokens, thêm `[CLS]` token và positional embedding; patch embedding
  có thể cài bằng `Conv2d(kernel_size=P, stride=P)`.

Thiết kế E2 theo các điểm trên:

- Dataset MNIST giữ giống E1.
- Tokenization: patch `7 x 7` tạo 16 tokens và row tokenization tạo 28 tokens.
- Classifier dùng `[CLS]` token, learnable positional embedding, một encoder block
  pre-norm và classification head.
- So sánh `custom_patch`, `pytorch_patch`, `custom_row`.

