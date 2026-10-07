# E1 - Phan loai anh: Softmax, MLP, CNN

## 1. Muc tieu

Bai E1 xay dung va so sanh ba bo phan loai tren cung mot tap anh nho:

- Softmax classifier tren vector anh flatten.
- MLP classifier tren vector anh flatten.
- CNN classifier dung convolution va pooling.

Tat ca mo hinh deu dung PyTorch va tu viet training loop theo chuoi:

```text
forward -> loss -> backward() -> optimizer.step()
```

Muc tieu chinh la so sanh su khac nhau ve kha nang bieu dien cua mo hinh tuyen tinh,
MLP phi tuyen va CNN co inductive bias phu hop voi anh.

## 2. Du lieu va split

Dataset chon cho E1-E3 la **MNIST**, gom anh xam kich thuoc `1 x 28 x 28` va 10 lop chu so
tu 0 den 9.

Theo nguyen tac data pipeline:

- Train set dung de hoc tham so cua mo hinh.
- Validation set dung de theo doi qua trinh train, chon epoch tot nhat va tuning hyperparameters.
- Test set chi dung de danh gia cuoi cung.

Split trong thuc nghiem:

| Split | So mau |
|---|---:|
| Train | 54,000 |
| Validation | 6,000 |
| Test | 10,000 |

Validation set duoc tach tu training set voi `val_ratio = 0.1` va seed co dinh `42`.
Tien xu ly gom `ToTensor()` va normalize bang thong ke chuan cua MNIST:
`mean = 0.1307`, `std = 0.3081`.

## 3. Mo hinh

### 3.1 Softmax classifier

Anh duoc flatten thanh vector 784 chieu roi dua qua mot lop tuyen tinh `784 -> 10`.
Model tra ve logits. Trong PyTorch, `nn.CrossEntropyLoss` nhan logits truc tiep va tu tinh
`log_softmax` ben trong, nen khong can dat softmax o cuoi model.

So tham so:

```text
784 * 10 + 10 = 7,850
```

### 3.2 MLP classifier

MLP dung hai tang an:

```text
784 -> 256 -> 128 -> 10
```

Cac tang an dung ReLU va Dropout `0.2`. MLP manh hon Softmax vi cac hidden layers va
activation phi tuyen giup hoc feature transformation truoc khi phan loai. Tuy nhien, MLP van
flatten anh nen lam mat cau truc khong gian giua cac pixel lan can.

So tham so:

```text
784 * 256 + 256 = 200,960
256 * 128 + 128 = 32,896
128 * 10 + 10 = 1,290
Total = 235,146
```

### 3.3 CNN classifier

CNN dung hai block convolution:

```text
Conv2d(1, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)
Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool2d(2)
Flatten -> Linear(64*7*7, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 10)
```

CNN phu hop voi anh hon vi giu input o dang `B x C x H x W`, khai thac locality, weight
sharing va feature maps. Convolution hoc cac dac trung cuc bo nhu canh, net cong va hinh dang
chu so.

So tham so:

```text
Conv1: 32 * 1 * 3 * 3 + 32 = 320
Conv2: 64 * 32 * 3 * 3 + 64 = 18,496
FC1: 64 * 7 * 7 * 128 + 128 = 401,536
FC2: 128 * 10 + 10 = 1,290
Total = 421,642
```

## 4. Cau hinh huan luyen

Thuc nghiem chinh duoc chay tren Google Colab voi GPU CUDA:

| Hyperparameter | Gia tri |
|---|---:|
| Epochs | 20 |
| Batch size | 16 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Loss | CrossEntropyLoss |
| Device | CUDA |

Lenh chay:

```bash
python E1/train.py --models all --epochs 20 --batch-size 16 --device cuda
```

Ket qua duoc luu trong `E1/results/`, gom `summary.csv`, `metrics.json`,
`learning_curves.png`, cac confusion matrix va cac mau du doan sai.

## 5. Ket qua thuc nghiem

| Model | Parameters | Best epoch | Best val accuracy | Test loss | Test accuracy | Macro F1 | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Softmax | 7,850 | 5 | 0.9155 | 0.295770 | 0.9209 | 0.919632 | 6m 58s |
| MLP | 235,146 | 14 | 0.9770 | 0.077527 | 0.9782 | 0.978016 | 7m 51s |
| CNN | 421,642 | 20 | 0.9915 | 0.027936 | 0.9917 | 0.991646 | 8m 24s |

Thu tu ket qua tren test set la:

```text
CNN (99.17%) > MLP (97.82%) > Softmax (92.09%)
```

Softmax co it tham so nhat va tao baseline tuyen tinh. MLP co nhieu tham so hon va co ReLU,
nen hoc duoc quan he phi tuyen tot hon. CNN dat ket qua cao nhat vi kien truc convolution phu
hop hon voi du lieu anh.

## 6. Phan tich learning curves

Softmax hoi tu rat som. Validation accuracy tot nhat o epoch 5, sau do train accuracy tang
nhe nhung validation accuracy khong cai thien on dinh. Dieu nay cho thay mo hinh tuyen tinh
da gan cham gioi han bieu dien cua no tren input flatten.

MLP cai thien ro so voi Softmax va dat best validation accuracy o epoch 14. Training accuracy
tiep tuc tang trong khi validation accuracy dao dong nhe, nhung test accuracy van cao va gan
voi validation accuracy. Dieu nay cho thay MLP tong quat hoa tot trong thuc nghiem nay.

CNN dat validation accuracy cao nhat o epoch 20 va test accuracy 99.17%. Training va validation
accuracy deu cao, khong co dau hieu overfitting nghiem trong. Ket qua nay phu hop voi ly thuyet:
CNN co inductive bias tot cho anh nho nhu MNIST.

## 7. Phan tich confusion matrix va mau sai

Confusion matrix cho thay Softmax co so loi cao hon ro ret. Cac loi lon gom:

- Chu so `5` bi du doan thanh `3` 46 lan.
- Chu so `2` bi du doan thanh `8` 42 lan va thanh `3` 30 lan.
- Chu so `8` bi du doan thanh `3` 33 lan, thanh `5` 24 lan va thanh `6` 20 lan.
- Chu so `9` bi du doan thanh `7` 26 lan va thanh `4` 25 lan.

MLP giam manh cac loi nay. Vi du:

- `5 -> 3` giam tu 46 xuong 6.
- `2 -> 8` giam tu 42 xuong 3.
- `8 -> 3` giam tu 33 xuong 0.

CNN tiep tuc giam loi va confusion matrix tap trung gan duong cheo chinh. Cac loi con lai chu
yeu nam o cac chu so viet khong ro hoac co hinh dang gan nhau:

- `5 -> 3`: 9 lan.
- `9 -> 4`: 12 lan.
- `7 -> 1`: 4 lan.
- `6 -> 4`: 3 lan.

Quan sat cac anh misclassified cho thay nhieu mau bi viet nghieng, dut net, dinh net hoac co hinh
dang gan voi lop khac. Vi du, mot so anh `5` co phan tren/duoi gan voi `3`, mot so anh `9` co
than duoi mo lam model nham voi `4` hoac `5`, va mot so anh `7` viet thang nen gan voi `1`.
Day la cac loi hop ly voi bai toan nhan dang chu so viet tay.

## 8. Ket luan

Ket qua thuc nghiem phu hop voi ky vong ly thuyet:

- Softmax la baseline tuyen tinh, don gian va it tham so nhat, nhung bi gioi han ve kha nang
  hoc dac trung phi tuyen.
- MLP cai thien dang ke nho hidden layers va ReLU, nhung van lam mat cau truc khong gian do
  flatten anh.
- CNN dat ket qua tot nhat vi convolution khai thac locality, weight sharing va feature maps,
  phu hop voi du lieu anh hon Softmax va MLP.

Do do, tren MNIST, CNN la mo hinh phu hop nhat trong ba kien truc duoc so sanh.
