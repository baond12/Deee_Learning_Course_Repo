# Tóm tắt yêu cầu môn CO5085

Nguồn: `Exercise/exercise-vne.pdf`, `Exercise/assignment1-vne.pdf`,
`Exercise/assignment2-vne.pdf`.

## Landing Page

Hạn khung trang: **19/10/2026 23:59 GMT+7**.

Trang GitHub Pages cần có:

- Tên nhóm, thành viên, giảng viên.
- Link đến Exercise E1-E3, Assignment A1 và Assignment A2.
- Link code, báo cáo PDF và đề bài liên quan.

## Exercise E1-E3

Nộp qua GitHub, không nộp file lên LMS. Dùng một tập ảnh nhỏ cho cả E1-E3, ví dụ
MNIST, Fashion-MNIST hoặc CIFAR-10. Yêu cầu chung là tự viết training loop PyTorch
và có một báo cáo PDF chung cho cả ba bài.

### E1 - Softmax, MLP, CNN

Hạn: **02/11/2026 23:59 GMT+7**.

- Softmax classifier trên vector flatten.
- MLP classifier trên vector flatten.
- CNN classifier phù hợp ảnh nhỏ.
- So sánh accuracy, số tham số, learning curves và mẫu lỗi/confusion matrix.

### E2 - Transformer: MSA và tokenization

Hạn: **02/11/2026 23:59 GMT+7**.

- MSA tự viết từ các phép toán cơ bản.
- MSA bằng module PyTorch để đối chiếu.
- Ít nhất hai cách tokenization.
- So sánh MSA tự viết vs PyTorch và các cách tokenization.

### E3 - LSTM / GRU trên chuỗi ảnh

Hạn: **09/11/2026 23:59 GMT+7**.

- Có cả LSTM và GRU.
- Mô tả cách đưa ảnh thành chuỗi.
- So sánh với baseline MLP/CNN từ E1.
- Báo cáo accuracy, số tham số và nhận xét.

## Assignment A1

Chủ đề: **CNN vs Transformer trên tập ảnh lớn**.

- A1.1 hạn **26/10/2026 23:59 GMT+7**: khóa dataset, split, augmentation, kiến trúc
  và protocol.
- A1.2 hạn **09/11/2026 23:59 GMT+7**: chạy đủ thí nghiệm và báo cáo kết quả.

Yêu cầu chính:

- Dataset công khai, đủ lớn và khó hơn MNIST/Fashion-MNIST/CIFAR-10.
- Ít nhất một CNN và một Transformer.
- So sánh from-scratch vs pretrained.
- Với pretrained, so sánh freeze toàn bộ backbone, freeze một phần và full fine-tune.
- Báo cáo accuracy/F1, thời gian, số tham số trainable và phân tích lỗi.

## Assignment A2

Chủ đề: **tự chọn trong computer vision + phân tích paper**.

- A2.1 hạn **02/11/2026 23:59 GMT+7**: khóa chủ đề, paper, dataset, metric và kế hoạch
  pipeline.
- A2.2 hạn **16/11/2026 23:59 GMT+7**: pipeline chạy được, so sánh ít nhất hai biến
  thể, phân tích paper đầy đủ.

Yêu cầu chính:

- Chủ đề không trùng phân loại ảnh thuần của A1.
- Có pipeline huấn luyện hoặc fine-tune và đánh giá.
- So sánh ít nhất hai phương pháp hoặc hai biến thể.
- Phân tích paper: phương pháp, thí nghiệm, hạn chế và liên hệ với pipeline của nhóm.
- Có thể chọn segmentation, detection, ReID, generative hoặc chủ đề CV khác nếu được
  giảng viên đồng ý.

