# 04. Semantic Segmentation에서 RF의 중요성

## 🎯 핵심 질문

- Semantic segmentation은 왜 충분한 receptive field를 필요로 하는가?
- FCN (Long et al. 2015)의 fully convolutional 접근이 혁신적이었던 이유는?
- U-Net (Ronneberger et al. 2015)의 encoder-decoder + skip connection 구조가 작은 RF를 보완하는 방식은?
- DeepLab v3+의 ASPP와 어떻게 연결되는가?
- 전역 문맥(global context) 없이 "소파 위의 고양이" 같은 장면을 이해할 수 있는가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

Semantic segmentation은 **픽셀 수준의 분류**입니다. 단순히 이미지를 분류하는 것과 달리:

- ImageNet 분류: "이 사진에 개가 있다" → 이미지 전체 context 활용
- Semantic segmentation: "어느 픽셀이 개인가?" → **각 픽셀의 local + global context 필요**

receptive field와의 관계:
1. **RF가 작으면**: 각 픽셀이 주변만 봄 → 맥락 이해 불가
   - "개의 다리 부분만 픽셀별로 보면 배경과 구분 어려움"
2. **RF가 크면**: 전체 맥락 이해 가능
   - "개 전체 형태, 주변 배경 동시 이해 → 정확한 경계 예측"

따라서 semantic segmentation은 receptive field 설계의 **대표적인 응용**입니다.

---

## 📐 수학적 선행 조건

- 이전 3개 문서: Theoretical RF, Effective RF, Dilated Convolution
- Convolution, pooling, transpose convolution (upsampling)
- Cross-entropy loss, skip connection

---

## 📖 직관적 이해

### Semantic Segmentation의 본질

입력: $H \times W \times 3$ 이미지

출력: $H \times W \times C$ segmentation map (C = 클래스 수)

각 위치 $(i, j)$의 클래스 $c$를 예측:
$$\hat{y}_{i,j} = \arg\max_c P(c | x_{i,j}, \text{context})$$

**핵심**: "context"의 크기가 중요합니다.

### RF와 Segmentation Quality의 관계

**예시**: PASCAL VOC 데이터셋, 평균 객체 크기 ~100×100 픽셀

- RF=50: 불충분 → 객체의 일부만 봄
- RF=100-200: 적절 → 객체 전체 + 배경 동시 보기
- RF=500: 과도 → 너무 큰 context (불필요한 정보)

따라서 **RF ≈ 객체 크기 × 1.5~2배** 권장됩니다.

### Dense Prediction의 구조적 도전

일반적인 CNN (AlexNet 등):
- 입력: 224×224
- 최종 feature map: 7×7 (stride 32)
- RF: 충분하지만 **해상도 손실**

Segmentation에는:
- 입력: 512×512 이상
- 최종 feature map: 원본 해상도 (dense prediction)
- RF: 충분하면서도 해상도 유지 필요 → **모순**

이를 해결하는 3가지 방식이 있습니다.

---

## ✏️ 엄밀한 정의·정리

### 정의 3.12 — Semantic Segmentation

입력 이미지 $I \in \mathbb{R}^{H \times W \times 3}$에 대해, 각 픽셀의 클래스 레이블을 예측:

$$\hat{Y} = f_\theta(I) \in \mathbb{R}^{H \times W \times C}$$

여기서 $f_\theta$는 전체 해상도를 유지하는 네트워크, $C$는 클래스 수입니다.

### 정리 3.13 — Dense Prediction과 RF의 Trade-off

Dense prediction을 위해서는:
1. **해상도 유지**: stride 작아야 함 → RF 느려서 증가
2. **충분한 RF**: depth 증가 필요 → 연산 증가

따라서 다음 3가지 전략으로 해결:
- **FCN**: Upsampling (해상도 복구)
- **U-Net**: Skip connection (low-level 정보 복구)
- **DeepLab**: Dilated convolution (RF 확장, 해상도 유지)

### 정의 3.14 — Receptive Field 요구사항

객체 크기 $S$에 대해, segmentation 성능이 우수하려면:

$$RF \gtrsim 1.5 \cdot S$$

이유: 객체 경계 정확도를 위해 배경도 충분히 봐야 함.

---

## 🔬 증명 또는 수학적 유도

### FCN의 아키텍처: Upsampling으로 해상도 복구

VGG-16 기반:
1. **Encoder** (convolution + pooling): 224→112→56→28→14→7 (stride 32)
   - RF는 크게 증가
2. **Skip connection**: 중간 해상도 feature 보존
3. **Decoder** (transpose convolution): 7→14→28→56→112→224 (stride 복구)
   - 간단한 upsampling (bilinear)

**장점**: 계산 효율적

**단점**: 
- 저해상도에서 "부정확한 판단" → 고해상도 복구해도 오류 유지
- Boundary blur: 경계가 흐릿함

### U-Net: Skip Connection으로 Low-level 정보 복구

Encoder-decoder에 **concatenation skip connection** 추가:

```
Encoder: 572×572 → conv(3×3) → 570×570 → maxpool → 285×285
         ...
         28×28 (bottleneck)

Decoder: 28×28 → upsample → 56×56 → concat [skip from encoder]
         → conv → 56×56 → upsample → 112×112
         ...
         572×572 (output)
```

**핵심**: Skip connection이 high-resolution 정보(low-level features) 제공
- 고해상도 encoder features (edge, texture)가 decoder에 직접 연결
- Upsampling 오류를 low-level 정보로 보정

**수학적 해석**:
$$h_{\text{decoder}}^l = \text{Upsample}(h_{\text{decoder}}^{l-1}) \oplus h_{\text{encoder}}^{L-l}$$

여기서 $\oplus$는 concatenation (또는 addition).

**효과**: 
- RF는 U-Net 크기로 제한되지만
- Skip connection으로 high-resolution edge 정보 보상
- 특히 의료영상 분할(의료기관 응용)에서 SOTA

### DeepLab: Dilated Convolution + ASPP

**핵심 아이디어**: stride 줄이되, dilated convolution으로 RF 확보

1. **Backbone** (ResNet-101 modified):
   - 표준 ResNet-50: stride 32
   - DeepLab: stride 8 (대신 dilation 증가)
   
   정리 3.9의 적용:
   - Layer 원래: stride 1 → dilation 1 (RF ≈ 51)
   - Layer 수정: stride 1 (해상도 유지) → dilation 16 (RF ≈ 1600, 예시)

2. **ASPP** (Atrous Spatial Pyramid Pooling):
   - 다양한 dilation rate (6, 12, 18, ...)로 multi-scale context 추출
   - 이론 배경: 문제 3.2 참고 (gridding artifact 해결)

3. **Decoder**:
   - 간단한 bilinear upsampling (stride 8→32)
   - FCN보다 정확함 (high-level semantic context 충분)

**효과**:
$$\text{Quality} = \text{High-level context (ASPP)} + \text{Spatial accuracy (stride 8)}$$

---

## 💻 실험 재현 / PyTorch 구현

### 구현 1: FCN 아키텍처

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FCN(nn.Module):
    """Fully Convolutional Networks for Semantic Segmentation"""
    
    def __init__(self, num_classes=21):
        super().__init__()
        
        # Load pre-trained VGG-16
        vgg = models.vgg16(pretrained=True)
        features = list(vgg.features.children())
        classifier = list(vgg.classifier.children())
        
        # Encoder: features
        self.features = nn.Sequential(*features)
        
        # Modify classifier to conv layers
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        
        # Score (logits)
        self.score_pool5 = nn.Conv2d(4096, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        
        # Upsampling layers
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, 
                                           4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes,
                                               4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes,
                                          16, stride=8, bias=False)
    
    def forward(self, x):
        input_size = x.size()
        
        # Encoder
        x = self.features(x)  # Stride 32
        
        # FC layers (applied as conv)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.drop6(x)
        
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.drop7(x)
        
        # Score
        x = self.score_pool5(x)  # (B, C, H/32, W/32)
        
        # 2x upsampling
        x = self.upscore2(x)  # (B, C, H/16, W/16)
        
        # Add skip from pool4
        # ... (simplified, skip connection would go here)
        
        # 8x upsampling to input size
        x = self.upscore8(x)  # (B, C, H, W)
        
        # Ensure output matches input size
        if x.size() != input_size:
            x = F.interpolate(x, size=input_size[2:], 
                            mode='bilinear', align_corners=False)
        
        return x

# Usage
model = FCN(num_classes=21)
x = torch.randn(2, 3, 512, 512)
output = model(x)
print(f"Input shape: {x.shape}, Output shape: {output.shape}")
```

### 구현 2: U-Net 아키텍처

```python
class UNet(nn.Module):
    """U-Net for medical image segmentation"""
    
    def __init__(self, num_classes=2, in_channels=3):
        super().__init__()
        
        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)  # Concatenated (512+512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)  # Concatenated (256+256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)  # Concatenated (128+128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)  # Concatenated (64+64)
        
        # Final output
        self.final = nn.Conv2d(64, num_classes, 1)
    
    @staticmethod
    def conv_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)
        x = self.pool1(e1)
        
        e2 = self.enc2(x)
        x = self.pool2(e2)
        
        e3 = self.enc3(x)
        x = self.pool3(e3)
        
        e4 = self.enc4(x)
        x = self.pool4(e4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.upconv4(x)
        x = torch.cat([x, e4], dim=1)  # Skip connection (concatenate)
        x = self.dec4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)
        
        # Output
        x = self.final(x)
        return x

# Usage
model = UNet(num_classes=1)
x = torch.randn(2, 3, 256, 256)
output = model(x)
print(f"U-Net Input: {x.shape}, Output: {output.shape}")
```

### 구현 3: DeepLab v3+ 간소화 버전

```python
class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super().__init__()
        
        # 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Atrous convolutions with different rates
        self.branch_rates = nn.ModuleList()
        for rate in rates:
            self.branch_rates.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 
                         padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Image pooling (ASPP의 global context 부분)
        self.branch_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, 
                     1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        shape = x.shape[2:]  # Spatial dimensions
        
        # Collect all branches
        branches = [self.branch1(x)]
        
        for branch in self.branch_rates:
            branches.append(branch(x))
        
        # Global context branch
        pool = self.branch_pool(x)
        pool = F.interpolate(pool, size=shape, 
                            mode='bilinear', align_corners=False)
        branches.append(pool)
        
        # Concatenate all branches
        concat = torch.cat(branches, dim=1)
        
        # Project
        output = self.project(concat)
        return output

class DeepLabV3Plus(nn.Module):
    """Simplified DeepLab v3+"""
    
    def __init__(self, num_classes=21, backbone='resnet50'):
        super().__init__()
        
        # Backbone (ResNet-50 with stride=8)
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3  # stride=8
                # layer4 skipped to maintain stride=8
            )
            backbone_channels = 1024
        
        # ASPP
        self.aspp = ASPPModule(backbone_channels, 256)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )
    
    def forward(self, x):
        input_shape = x.shape[2:]
        
        # Backbone: stride=8
        backbone_out = self.backbone(x)  # (B, 1024, H/8, W/8)
        
        # ASPP: multi-scale context
        aspp_out = self.aspp(backbone_out)  # (B, 256, H/8, W/8)
        
        # Decoder
        decoder_out = self.decoder(aspp_out)  # (B, num_classes, H/8, W/8)
        
        # Upsample to input size
        output = F.interpolate(decoder_out, size=input_shape,
                             mode='bilinear', align_corners=False)
        
        return output

# Usage
model = DeepLabV3Plus(num_classes=21)
x = torch.randn(2, 3, 512, 512)
output = model(x)
print(f"DeepLab v3+ Input: {x.shape}, Output: {output.shape}")
```

### 구현 4: RF와 성능의 관계 시각화

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_segmentation_methods():
    """Compare different segmentation approaches"""
    
    methods = ['FCN', 'U-Net', 'DeepLab v3+']
    rf_values = [51, 28, 128]  # Approximate RF values
    speed = [100, 85, 90]  # Relative speed (FCN=baseline)
    accuracy = [72, 88, 92]  # mIoU on PASCAL VOC (hypothetical)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RF comparison
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    axes[0].barh(methods, rf_values, color=colors)
    axes[0].set_xlabel('Receptive Field (approx.)')
    axes[0].set_title('Receptive Field Size')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Speed vs Accuracy trade-off
    axes[1].scatter(speed, accuracy, s=200, c=colors, alpha=0.7, edgecolors='black')
    for i, method in enumerate(methods):
        axes[1].annotate(method, (speed[i], accuracy[i]), 
                        xytext=(5, 5), textcoords='offset points')
    axes[1].set_xlabel('Speed (relative)')
    axes[1].set_ylabel('Accuracy (mIoU %)')
    axes[1].set_title('Speed vs Accuracy Trade-off')
    axes[1].grid(alpha=0.3)
    
    # Key characteristics
    characteristics = {
        'FCN': ('Simple\nUpsampling', 'Low\nAccuracy', 'Fast'),
        'U-Net': ('Skip\nConnections', 'Good\nAccuracy', 'Moderate'),
        'DeepLab': ('Dilated\nConv+ASPP', 'Excellent\nAccuracy', 'Slightly\nSlower')
    }
    
    y_pos = np.arange(len(methods))
    axes[2].axis('off')
    
    table_text = "Method | Key Feature | Accuracy | Speed\n"
    table_text += "-" * 50 + "\n"
    for method in methods:
        table_text += f"{method:12} | {characteristics[method][0]:15} | {characteristics[method][1]:15} | {characteristics[method][2]}\n"
    
    axes[2].text(0.1, 0.5, table_text, fontfamily='monospace', 
                fontsize=10, verticalalignment='center')
    axes[2].set_title('Summary Comparison')
    
    plt.tight_layout()
    plt.savefig('segmentation_methods_comparison.png', dpi=150)
    plt.show()

visualize_segmentation_methods()
```

**예상 출력**:
- FCN: RF 작음, 빠르지만 정확도 낮음
- U-Net: RF 중간, skip connection으로 보상, 의료 분야에 최적
- DeepLab: RF 크고 다양, ASPP로 multi-scale context, 가장 정확함

---

## 🔗 이론과 실전의 간극

### 실제 Segmentation 성능

Receptive field는 **필요조건이지만 충분조건이 아닙니다**.

중요한 요소들:
1. **Pre-training**: ImageNet pre-training이 huge boost
2. **Data augmentation**: Random crop, flip, color jitter
3. **Training strategy**: Batch normalization, learning rate scheduling
4. **Multi-scale training**: 여러 input size로 학습

### 특수 도메인

- **의료영상**: U-Net 기반 (작은 RF, skip connection 중요)
- **자율주행**: DeepLab 계열 (큰 RF, 다양한 scale 필요)
- **3D 의료**: V-Net (3D convolution + skip)

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| RF가 크면 항상 좋다 | 연산 cost, overfitting risk 증가 |
| Skip connection이 만능 | U-Net은 깊이에 제약 (의료영상 크기 제약) |
| 고정 크기 입력 | 실제로는 다양한 해상도 필요 (test time augmentation) |
| Stride 8이 최적 | Task에 따라 stride 4 또는 16 더 나을 수도 |

---

## 📌 핵심 정리

$$\boxed{\text{Segmentation quality} = f(\text{RF}, \text{Skip info}, \text{Multi-scale context})}$$

| 방법 | RF | 전략 | 주요 용도 |
|------|-----|------|----------|
| **FCN** | 중간 | Upsampling | 초기 dense prediction |
| **U-Net** | 작음 | Skip connection | 의료 분할 |
| **DeepLab** | 크고 다양 | Dilated+ASPP | 일반 semantic seg |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 256×256 PASCAL VOC 이미지에서 평균 객체 크기가 80×80이라 하자. 추천 RF 최소값은?

<details>
<summary>힌트 및 해설</summary>

정리 3.14: $RF \gtrsim 1.5 \times S$

$RF \gtrsim 1.5 \times 80 = 120$

따라서 최소 RF ≈ 120 권장.

- FCN (stride 32): RF ≈ 51 → 부족
- U-Net (4개 pool): RF ≈ 28 → 부족하지만 skip으로 보상
- DeepLab (stride 8, dilation): RF ≈ 128 → 충분

</details>

**문제 2** (심화): U-Net에서 skip connection이 없으면 정확도가 큰 폭으로 떨어진다. 왜 그럴까? 이론적으로 설명하라.

<details>
<summary>힌트 및 해설</summary>

U-Net의 RF는 약 28정도로 작습니다. 그럼에도 높은 정확도를 얻는 이유:

1. **Skip connection**: Encoder의 high-resolution feature map을 decoder로 직접 전달
2. **Low-level features 보존**: 초반 conv layer의 edge, texture 정보 보존
3. **Upsampling 오류 보정**: Bilinear upsample의 blur를 sharp한 edge로 보정

Skip 없이는:
- Bottleneck에서 모든 정보가 압축 (28×28 feature map)
- Decoder에서 upsampling만으로 복구 불가능
- 결과: boundary 부정확, organ 형태 왜곡

따라서 skip connection은 "RF 부족을 보상하는 구조적 혁신"입니다.

</details>

**문제 3** (논문 비평): DeepLab v3+는 encoder-decoder 구조에도 ASPP를 포함합니다. Encoder에서의 ASPP와 decoder에서의 역할이 다른가?

<details>
<summary>힌트 및 해설</summary>

**Encoder의 ASPP** (후반부, stride=8 feature):
- 다양한 dilation rate로 multi-scale semantic context 추출
- 전역 평균 pooling 포함 → 이미지 전체 context

**Decoder의 역할**:
- 단순 bilinear upsampling (stride 8→32)
- Encoder의 low-level feature와 fusion 가능하나, 논문에서 대부분 생략

따라서:
- **ASPP = semantic richness 확보**
- **Decoder = spatial accuracy 확보** (upsampling)

이것이 DeepLab의 설계 철학: "고품질 high-level semantic이 더 중요하다" (고해상도 encoder feature보다)

</details>

---

<div align="center">

[◀ 이전](./03-dilated-rf.md) | [📚 README](../README.md) | [다음 ▶](../ch4-resnet/01-residual-block.md)

</div>
