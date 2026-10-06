# E3 slide notes

Các điểm từ slide được dùng để thiết kế E3:

- `04-rnn/RNN-Summary.pdf`, trang 6: batch sequence thường có shape `(B, T, D)` với
  `B` là batch, `T` là số bước thời gian và `D` là số đặc trưng mỗi bước.
- `04-rnn/RNN-Summary.pdf`, trang 10-11: bài phân loại một nhãn cho cả chuỗi là
  many-to-one; có thể gắn classification head vào hidden state cuối hoặc pooling qua
  các hidden states.
- `04-rnn/RNN-Summary.pdf`, trang 20-22: training RNN dùng backpropagation through
  time; chuỗi dài có thể gặp vanishing/exploding gradients, nên gradient clipping là
  thực hành tốt.
- `04-rnn/RNN-Summary.pdf`, trang 24-29: LSTM dùng cell state và các gate để kiểm
  soát quên/ghi/đọc thông tin, giúp giữ phụ thuộc dài tốt hơn vanilla RNN.
- `04-rnn/RNN-Summary.pdf`, trang 30-34: GRU dùng update/reset gates, ít tham số hơn
  LSTM trong cấu hình tương đương và thường là baseline mạnh khi cần train nhanh.
- `04-rnn/RNN-Summary.pdf`, trang 37-39: PyTorch `nn.LSTM` và `nn.GRU` nhận input
  `(B, T, D)` khi `batch_first=True`; ví dụ many-to-one classifier có thể lấy
  `out[:, -1, :]`.

Thiết kế E3 theo các điểm trên:

- Ảnh MNIST row sequence: `T=28`, `D=28`.
- Ảnh MNIST patch sequence `7 x 7`: `T=16`, `D=49`.
- Các mô hình ảnh: `lstm_row`, `gru_row`, `lstm_patch`, `gru_patch`.
- Baseline không hồi quy: MLP và CNN từ E1.
- Sentiment analysis dùng text tokens độ dài thay đổi, padding theo batch, rồi dùng
  `pack_padded_sequence` và `pad_packed_sequence` để RNN bỏ qua padding.

