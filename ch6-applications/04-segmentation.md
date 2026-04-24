# 04. 의미론적 분할과 인스턴스 분할 (Segmentation) — FCN, U-Net, DeepLab

## 🎯 핵심 질문

- FCN (Long et al., 2015)이 처음 end-to-end dense prediction을 어떻게 가능하게 했는가?
- U-Net (Ronneberger et al., 2015)의 skip connection이 왜 의료 이미지 분할의 표준이 되었는가?
- DeepLab의 atrous(dilated) convolution과 ASPP (Atrous Spatial Pyramid Pooling)는 어떻게 receptive field를 확대하는가?
- Pixel-level annotation에서 class imbalance를 다루기 위해 IoU loss와 Dice loss를 사용하는 이유는?

---

## 🔍 왜 이 개념이 CNN에 중요한가

분류와 탐지를 넘어, **pixel-level 예측**은 의료 영상(종양 분할), 자율주행(scene understanding), 위성 이미지(토지 이용) 등 중요한 응용입니다. FCN은 **fully convolutional architecture**를 제시하여 임의 크기 이미지를 처리 가능하게 했고, U-Net은 **skip connection** 아이디어로 fine-grained detail을 유지하면서 semantic context를 포착했습니다. DeepLab 시리즈는 **atrous convolution**으로 receptive field를 지수적으로 확대하면서도 계산량을 제어합니다. 이들은 현대 segmentation의 foundation입니다.

---

## 📐 수학적 선행 조건

- [Ch5-03 Convolutional Arithmetic](../ch5-modern-cnn/03-conv-arithmetic.md): Padding, stride, receptive field, transposed convolution
- [Ch6-01 Classification](./01-classification.md): Cross-entropy, softmax
- [Ch6-02 Two-Stage Detection](./02-two-stage-detection.md): IoU metric
- 선형대수: 2D convolution operation

---

## 📖 직관적 이해

### FCN — Fully Convolutional Networks

**문제**: 분류 네트워크(AlexNet, VGG)는 마지막에 FC layer로 축약 → 공간 정보 손실.

**혁신**: FC layer를 $1 \times 1$ convolution으로 대체:
```
FC(1000) → Conv(1, 1, 1000)
```

이제 activation map 모양이 유지되고, 임의 크기 입력 처리 가능.

**아키텍처** (FCN-8):
1. VGG-16 backbone (conv layer만 사용)
2. 마지막 pool 후 $1 \times 1$ conv로 클래스 점수 생성
3. Bilinear upsampling으로 원본 크기 복원
4. Skip connection으로 중간 feature 추가 (detail 보존)

**한계**: upsampling 과정에서 정보 손실, fine detail 복원 어려움.

### U-Net — Encoder-Decoder with Skip Connections

**핵심 아이디어** (Ronneberger et al., 2015):
- **Encoder**: 일반적인 CNN (maxpool로 down-sampling)
- **Decoder**: Transposed convolution으로 up-sampling
- **Skip connection**: Encoder의 각 level에서 feature를 decoder의 대응 level로 직접 연결

```
Encoder:  64ch → 128ch → 256ch → 512ch
          ↓maxpool ↓maxpool ↓maxpool
                        ↓maxpool → 1024ch (bottleneck)
                        ↑↑↑ transpose conv
Decoder:  512ch ← 256ch ← 128ch ← 64ch → 1ch (segmentation)
           ↑concat ↑concat ↑concat
          (skip from encoder)
```

**효과**:
1. **Detail 보존**: Skip connection은 low-level feature(edge, texture)를 제공
2. **정보 손실 감소**: Up-sampling 전 down-sampling에서 버려진 정보 복원
3. **적은 데이터로도 학습**: 의료 영상(몇백 개 샘플)에서도 overfit 없이 학습

**응용**: 의료 이미지 분할의 사실상 표준 (2015년 이후).

### Atrous (Dilated) Convolution

**문제**: CNN의 receptive field는 depth에 선형적으로만 증가.
- Conv 3×3 5개: receptive field ~11×11 만 (깊지 않음)
- 깊은 네트워크는 계산량 증가

**해결**: Atrous convolution (dilation).

표준 convolution (stride 1):
```
output[i, j] = sum(kernel[a, b] * input[i+a, j+b] for a, b)
```

Atrous convolution (dilation rate $d$):
```
output[i, j] = sum(kernel[a, b] * input[i+d*a, j+d*b] for a, b)
```

**효과**:
- $d=1$: 표준 convolution
- $d=2$: kernel 사이에 1칸씩 gap → $3 \times 3$ kernel의 receptive field는 $5 \times 5$로 확대
- $d=4$: $3 \times 3$ kernel의 receptive field는 $7 \times 7$로 확대
- 계산량은 동일 (parameter 수 변화 없음)

**DeepLab** (Chen et al., 2016-2018):
- ResNet backbone + atrous convolution으로 stride 감소
- Atrous Spatial Pyramid Pooling (ASPP): 다양한 dilation rate 조합

```
Input → 1×1 conv + image pooling (global context)
      → 3×3 atrous conv (d=1)
      → 3×3 atrous conv (d=6)
      → 3×3 atrous conv (d=12)
      → Concat → 1×1 conv
      → Output
```

**효과**: 다양한 스케일의 context 동시 포착.

### Loss Functions for Imbalanced Segmentation

**문제**: Pixel-level annotation에서 class imbalance.
- 의료: 배경 99%, 종양 1%
- 자율주행: 도로 70%, 보행자 5%, 기타 25%

Standard cross-entropy:
$$L_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^N \sum_c y_{ic} \log p_{ic}$$

문제: 대다수의 배경 픽셀이 loss를 압도.

**IoU Loss** (Jaccard):
$$L_{\text{IoU}} = 1 - \frac{|X \cap Y|}{|X \cup Y|}$$

픽셀이 아닌 영역 단위로 평가 → 작은 객체도 중요.

**Dice Loss**:
$$L_{\text{Dice}} = 1 - \frac{2|X \cap Y|}{|X| + |Y|}$$

IoU와 유사하지만, 확률 기반:
$$L_{\text{Dice}} = 1 - \frac{2 \sum_i p_i y_i}{\sum_i p_i + \sum_i y_i}$$

**효과**: 클래스 크기에 관계없이 균형잡힌 학습.

---

## ✏️ 엄밀한 정의·정리

### 정의 6.19 — Fully Convolutional Network

모든 layer가 convolution 또는 pooling (FC layer 없음). 입력 크기 $H \times W$에 대해:
- Pooling stride $s = 2^d$일 때 출력 feature map: $\frac{H}{s} \times \frac{W}{s}$
- Up-sampling으로 원본 크기 복원: $\frac{H/s \cdot s}{1} = H$ (ideally)

### 정의 6.20 — Transposed Convolution (Deconvolution)

Input $x \in \mathbb{R}^{H \times W \times C_{\text{in}}}$, kernel $w \in \mathbb{R}^{K_h \times K_w \times C_{\text{in}} \times C_{\text{out}}}$:

출력 크기: $H' = (H-1) \cdot \text{stride} + K_h$

역 convolution이 아니라, 정방향 convolution의 행렬 표현을 전치한 형태.

### 정의 6.21 — Atrous (Dilated) Convolution

Dilation rate $d$에 대해:
$$y[i, j, c] = \sum_{a=0}^{K_h-1} \sum_{b=0}^{K_w-1} \sum_{c_{\text{in}}} w[a, b, c_{\text{in}}, c] \cdot x[i + d \cdot a, j + d \cdot b, c_{\text{in}}]$$

Effective kernel size (receptive field): $K_{\text{eff}} = K + (K-1)(d-1) = K + K(d-1) - (d-1)$

예: $K=3, d=2$ → $K_{\text{eff}} = 3 + 3 \cdot 1 = 5$

### 정의 6.22 — Atrous Spatial Pyramid Pooling (ASPP)

다양한 dilation rate에서의 atrous convolution 병렬:
$$\text{ASPP}(x) = \text{Concat}[\text{pool}(x), \text{conv}(x, d=1), \text{conv}(x, d=6), \text{conv}(x, d=12)]$$

각 branch를 처리 후 concatenate + $1 \times 1$ conv로 통합.

### 정의 6.23 — Intersection over Union (IoU) Loss

$$L_{\text{IoU}} = 1 - \text{IoU}(Y, \hat{Y}) = 1 - \frac{|Y \cap \hat{Y}|}{|Y \cup \hat{Y}|}$$

여기서 $Y$는 정답 마스크, $\hat{Y}$는 예측 마스크.

### 정의 6.24 — Dice Loss

$$L_{\text{Dice}} = 1 - \text{Dice}(Y, \hat{Y}) = 1 - \frac{2|Y \cap \hat{Y}|}{|Y| + |\hat{Y}|}$$

또는 확률 기반:
$$L_{\text{Dice}} = 1 - \frac{2 \sum_i p_i y_i}{\sum_i p_i + \sum_i y_i + \epsilon}$$

---

## 🔬 증명 또는 수학적 유도

### Receptive Field 계산

Convolution layer의 receptive field 증가:

표준 CNN (stride 1, padding same):
- Layer 1: RF = 3
- Layer 2: RF = 5
- Layer 3: RF = 7
- Layer n: RF = 2n + 1 (linear growth)

Atrous convolution (stride 1, dilation $d_i$):
$$\text{RF} = 1 + 2 \sum_{i=1}^n \prod_{j=1}^{i-1} d_j$$

예: $d_1=1, d_2=2, d_3=4$:
$$\text{RF} = 1 + 2(1 + 1 \cdot 2 + 1 \cdot 2 \cdot 4) = 1 + 2(1 + 2 + 8) = 1 + 22 = 23$$

표준 CNN으로는 11 layer 필요한 RF를 3 layer로 달성!

### Skip Connection의 정보 보존

Encoder에서 feature map $f_k$, decoder에서 up-sampling한 $f'_k$가 있을 때:

Skip connection 없음:
$$\text{loss} = \|y - \text{decode}(f'_k)\|^2$$

Skip connection:
$$\text{loss} = \|y - \text{decode}(\text{concat}(f_k, f'_k))\|^2$$

채널이 두 배 증가하지만, 추가 정보가 gradient flow를 향상시킴.

### Dice Loss의 수학적 성질

Pixel-level cross-entropy에서 발생하는 불균형:

$$L_{\text{CE}} = -\frac{1}{N} \sum_i [y_i \log p_i + (1-y_i) \log(1-p_i)]$$

$y_i=0$ (배경)인 픽셀이 99%면:
$$L_{\text{CE}} \approx -0.99 \log(1 - p_i) - 0.01 \log p_i$$

배경 항이 loss를 압도.

Dice loss는 평면(region) 단위:
$$L_{\text{Dice}} = 1 - \frac{2 \sum_i p_i y_i}{\sum_i p_i + \sum_i y_i}$$

이는 개별 픽셀이 아닌 집합의 overlap으로 평가 → 크기 불균형에 robust.

---

## 💻 실험 재현 / PyTorch 구현

### 실험 1 — U-Net 구현

```python
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, init_features=64):
        super().__init__()
        
        self.enc1 = DoubleConv(in_ch, init_features)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(init_features, init_features * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(init_features * 2, init_features * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(init_features * 4, init_features * 8)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(init_features * 8, init_features * 4, 2, stride=2)
        self.dec3 = DoubleConv(init_features * 4 * 2, init_features * 4)  # concat
        
        self.upconv2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, 2, stride=2)
        self.dec2 = DoubleConv(init_features * 2 * 2, init_features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(init_features * 2, init_features, 2, stride=2)
        self.dec1 = DoubleConv(init_features * 2, init_features)
        
        self.final_conv = nn.Conv2d(init_features, out_ch, 1)
    
    def forward(self, x):
        # Encoder with skip connections
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        x = self.final_conv(x)
        return x

# 테스트
model = UNet(in_ch=3, out_ch=1)
x = torch.randn(1, 3, 572, 572)
y = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
```

### 실험 2 — Atrous Convolution

```python
import torch.nn.functional as F

def atrous_conv(x, weight, bias, dilation=1, padding=None):
    """
    Simple atrous convolution implementation
    """
    if padding is None:
        padding = dilation * (weight.shape[2] - 1) // 2
    
    return F.conv2d(x, weight, bias, padding=padding, dilation=dilation)

# 테스트
x = torch.randn(1, 3, 32, 32)
kernel_size = 3
in_ch, out_ch = 3, 64

# 표준 convolution
weight = torch.randn(out_ch, in_ch, kernel_size, kernel_size)
bias = torch.randn(out_ch)

# d=1 (표준)
y1 = atrous_conv(x, weight, bias, dilation=1)
# d=2
y2 = atrous_conv(x, weight, bias, dilation=2)
# d=4
y4 = atrous_conv(x, weight, bias, dilation=4)

# Receptive field 비교
# RF = 1 + 2 * dilation * (kernel_size - 1)
rf_d1 = 1 + 2 * 1 * (3 - 1)  # = 5
rf_d2 = 1 + 2 * 2 * (3 - 1)  # = 9
rf_d4 = 1 + 2 * 4 * (3 - 1)  # = 17

print(f"d=1: RF={rf_d1}, output shape={y1.shape}")
print(f"d=2: RF={rf_d2}, output shape={y2.shape}")
print(f"d=4: RF={rf_d4}, output shape={y4.shape}")
```

### 실험 3 — Dice Loss vs Cross-Entropy

```python
def dice_loss(predictions, targets, smooth=1e-6):
    """
    Dice loss for binary segmentation
    predictions: [B, H, W] or [B, C, H, W]
    targets: [B, H, W] or [B, C, H, W]
    """
    predictions = torch.sigmoid(predictions)
    intersection = (predictions * targets).sum()
    dice = 2.0 * intersection / (predictions.sum() + targets.sum() + smooth)
    return 1.0 - dice

def iou_loss(predictions, targets, smooth=1e-6):
    """
    IoU (Jaccard) loss
    """
    predictions = torch.sigmoid(predictions)
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    iou = intersection / (union + smooth)
    return 1.0 - iou

# 테스트
torch.manual_seed(42)

# 불균형 마스크: 배경 99%, 객체 1%
B, H, W = 4, 64, 64
targets = torch.zeros(B, H, W)
targets[:, :10, :10] = 1  # 작은 객체

# 예측: 일반적으로 배경을 많이 예측하는 모델
predictions = torch.randn(B, H, W) * 0.1 - 0.5  # 대부분 음수

ce_loss = torch.nn.BCEWithLogitsLoss()(predictions, targets)
dice_l = dice_loss(predictions, targets)
iou_l = iou_loss(predictions, targets)

print(f"Cross-Entropy Loss: {ce_loss.item():.4f}")
print(f"Dice Loss: {dice_l.item():.4f}")
print(f"IoU Loss: {iou_l.item():.4f}")

# Dice/IoU는 작은 객체에도 상당한 penalty 부과
# CE는 배경 픽셀에 dominated
```

---

## 🔗 이론과 실전의 간극

### FCN의 Skip Connection 효과

이론: 중간 feature 추가로 detail 보존.

실전: 단순 addition이 concatenation보다 나음 (parameter 적음). 또한, 해상도 불일치 시 bilinear upsampling 후 addition 필요.

### U-Net vs FCN

U-Net이 의료 이미지에서 선호되는 이유:
- Symmetric encoder-decoder (정보 손실 최소)
- Dense skip connection (모든 level 연결)
- 작은 데이터셋에서 overfit 잘 안 함 (data augmentation + skip connection)

### DeepLab의 Multi-Scale Context

이론: ASPP로 다양한 receptive field 조합.

실전:
- d=1, 6, 12, 18 combination이 많은 dataset에서 robust
- 더 큰 d는 작은 feature map에서 zero padding issue 발생 가능
- ImageNet pre-training이 매우 중요

### Loss Function 선택

이론: Dice/IoU loss는 class imbalance robust.

실전:
- CE loss: 단순하고 빠름, balanced dataset에서 충분
- Dice: 의료 이미지(심한 imbalance) 기본 선택
- Hybrid (CE + Dice): 성능 개선, 학습 안정성

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 픽셀 레벨 정확한 annotation | 실제로는 경계 모호, 주석자 간 차이 있음 |
| 고정 atrous rate (6, 12, 18) | 데이터셋과 task에 따라 다른 값이 최적 |
| Up-sampling은 bilinear interpolation | 학습 가능한 transposed conv가 더 나을 수 있음 |
| Skip connection 항상 유용 | 일부 네트워크에선 불필요 또는 해로움 |
| IoU/Dice metric이 ground truth | 실제 의료 응용은 false positive/negative 비용 다름 |

---

## 📌 핵심 정리

$$\boxed{L_{\text{Dice}} = 1 - \frac{2|Y \cap \hat{Y}|}{|Y| + |\hat{Y}|} \text{ — class imbalance robust}}$$

| 개념 | 정의 |
|------|------|
| **FCN** | Fully convolutional, pixel-level 예측 첫 제시 |
| **Transposed convolution** | Up-sampling learnable layer |
| **Skip connection** | Encoder feature를 decoder로 직접 연결 |
| **U-Net** | Symmetric encoder-decoder + skip, 의료 표준 |
| **Atrous convolution** | $y[i+d \cdot a, j + d \cdot b]$, RF 지수 확대 |
| **ASPP** | 다양한 dilation rate 병렬 처리 |
| **Dice Loss** | $1 - \frac{2 \|Y \cap \hat{Y}\|}{\|Y\| + \|\hat{Y}\|}$ |
| **IoU Loss** | $1 - \text{IoU}$ |
| **DeepLab** | Atrous conv + ASPP, Cityscapes SOTA |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $32 \times 32$ 입력을 VGG-style 네트워크(maxpool stride 2, 3개 pool)로 처리하면 feature map 크기는?

<details>
<summary>힌트 및 해설</summary>

입력: $32 \times 32$
Pool 1: $32 / 2 = 16 \times 16$
Pool 2: $16 / 2 = 8 \times 8$
Pool 3: $8 / 2 = 4 \times 4$

Bilinear upsampling으로 원본으로 복원하려면 $4 \times 2 \times 2 \times 2 = 32$ (모두 2배씩).

</details>

**문제 2** (심화): Atrous convolution $d=2, 3 \times 3$ kernel에서 출력 크기가 입력 크기와 같으려면 (stride 1), padding은 얼마여야 하는가?

<details>
<summary>힌트 및 해설</summary>

Atrous convolution의 effective kernel size:
$$K_{\text{eff}} = K + (K-1)(d-1) = 3 + 2 \cdot 1 = 5$$

크기 보존을 위한 padding:
$$P = \lfloor K_{\text{eff}} / 2 \rfloor = 2$$

또는 일반식: $P = \lfloor d(K-1) / 2 \rfloor = \lfloor 2 \cdot 2 / 2 \rfloor = 2$.

</details>

**문제 3** (이론-실전): 의료 이미지 분할에서 종양이 전체 이미지의 1% 차지할 때, cross-entropy loss와 Dice loss 중 어느 것이 더 나을까? 그 이유는?

<details>
<summary>힌트 및 해설</summary>

**Cross-entropy**:
- 배경(99%)의 손실이 전체를 압도
- 모델이 항상 배경이라 예측해도 99% 정확도
- 종양 학습에 충분한 gradient 부여 안 함

**Dice loss**:
- 전체 겹침 면적으로 평가
- 배경 1% 향상보다 종양 1% 향상이 같은 가중치
- 종양에 충분한 학습 신호

따라서 **Dice loss가 훨씬 적합**.

</details>

---

<div align="center">

[◀ 이전](./03-one-stage-detection.md) | [📚 README](../README.md) | [다음 ▶](./05-self-supervised.md)

</div>
