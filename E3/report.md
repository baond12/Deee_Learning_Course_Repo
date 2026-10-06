# E3 - LSTM / GRU trên chuỗi ảnh và sentiment analysis

## 1. Mục tiêu

Bài E3 mô hình hóa ảnh MNIST như dữ liệu chuỗi bằng LSTM và GRU, sau đó so sánh với
baseline không hồi quy từ E1. Ngoài phần ảnh, repo có thêm pipeline sentiment analysis
dạng văn bản để minh họa RNN trên chuỗi token có độ dài thay đổi.

## 2. Dữ liệu ảnh và split

Dataset ảnh giữ giống E1/E2:

- MNIST, ảnh xám `1 x 28 x 28`, 10 lớp.
- Train: 54,000 ảnh.
- Validation: 6,000 ảnh.
- Test: 10,000 ảnh.
- Seed split: `42`.

## 3. Biểu diễn ảnh thành chuỗi

### 3.1 Row sequence

Mỗi hàng ảnh là một timestep:

```text
image: 1 x 28 x 28 -> sequence: T = 28, D = 28
```

### 3.2 Patch sequence

Ảnh được chia thành patch `7 x 7`, đi theo thứ tự raster scan:

```text
image: 1 x 28 x 28 -> 16 patches -> sequence: T = 16, D = 49
```

## 4. Mô hình ảnh

Các mô hình recurrent:

- `lstm_row`
- `gru_row`
- `lstm_patch`
- `gru_patch`

Cấu hình mặc định:

- `hidden_dim = 128`.
- `num_layers = 1`.
- `bidirectional = False` cho ảnh.
- Readout: lấy hidden output ở timestep cuối `out[:, -1, :]`.
- Head: Linear từ hidden state sang 10 logits.

Theo slide RNN, đây là bài many-to-one classification: một chuỗi đầu vào, một nhãn cho
toàn bộ chuỗi.

## 5. Baseline không hồi quy

Script E3 có thể train lại hai baseline từ E1 trên cùng split:

- `mlp_baseline`.
- `cnn_baseline`.

Nếu đã có `E1/results/summary.csv` và `E2/results/summary.csv`, script sẽ tạo thêm
`E3/results/comparison_e1_e2_e3.csv` để gom bảng so sánh.

## 6. Sentiment analysis dạng văn bản

Phần sentiment dùng một toy dataset nhỏ để minh họa pipeline chuỗi văn bản:

- Tokenize câu thành word tokens.
- Build vocabulary có `<pad>` và `<unk>`.
- Pad batch theo độ dài lớn nhất trong batch.
- Dùng `pack_padded_sequence` trước LSTM/GRU.
- Dùng `pad_packed_sequence` để unpack output.
- Mean pooling trên các timestep hợp lệ, bỏ qua padding.
- Classification head dự đoán 2 lớp: negative/positive.

Phần văn bản mặc định dùng bidirectional LSTM/GRU vì toàn bộ câu đã có sẵn khi phân loại.

## 7. Lệnh chạy

Smoke test shape/logic:

```bash
python E3/smoke_test.py
```

Train ảnh:

```bash
python E3/train.py --experiments all --epochs 5 --batch-size 128 --device auto
```

Train sentiment:

```bash
python E3/train_sentiment.py --rnn-type lstm --epochs 5 --device auto
python E3/train_sentiment.py --rnn-type gru --epochs 5 --device auto
```

## 8. Kết quả cần điền sau khi chạy

Kết quả ảnh:

- `E3/results/summary.csv`.
- `E3/results/comparison_e1_e2_e3.csv`.
- `E3/results/learning_curves.png`.
- `E3/results/confusion_matrix_*.png`.
- `E3/results/misclassified_*.png`.

Kết quả sentiment:

- `E3/results/sentiment/sentiment_lstm.json`.
- `E3/results/sentiment/sentiment_gru.json`.

## 9. Nhận xét dự kiến

LSTM và GRU đều có gating để giảm vấn đề vanishing gradients so với vanilla RNN. GRU có
ít trạng thái hơn và thường nhẹ hơn LSTM, còn LSTM tách cell state và hidden state nên
có thể linh hoạt hơn trên phụ thuộc dài. Row sequence có nhiều bước thời gian hơn patch
sequence, trong khi patch sequence có timestep ít hơn nhưng mỗi timestep giàu thông tin
cục bộ hơn. CNN baseline thường là baseline mạnh cho ảnh vì có inductive bias không gian,
trong khi LSTM/GRU cho thấy cách xử lý ảnh như chuỗi.

