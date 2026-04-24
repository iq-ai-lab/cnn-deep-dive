# 05. Neural Architecture Search (NAS): 자동 신경망 설계

## 🎯 핵심 질문

- **신경망 아키텍처를 자동으로 설계**할 수 있는가?
- Reinforcement Learning (RL) 기반 NAS (NASNet)는 어떻게 작동하는가? (컨트롤러 + 자식 네트워크 평가)
- **Differentiable Architecture Search (DARTS)**는 미분 불가능한 선택 문제를 어떻게 푸는가?
- RegNet과 같은 **설계 공간(design space) 접근법**은 무엇인가?

---

## 🔍 왜 이 개념이 CNN에 중요한가

Neural Architecture Search (NAS)는 **메타 학습(meta-learning)** 또는 **AutoML(자동 기계학습)**의 대표 사례입니다:

1. **자동화의 가능성**: 인간이 직관적으로 설계 → 알고리즘이 자동 탐색
2. **최적성의 증명**: NASNet (2017), AmoebaNet (2019) 모두 SOTA (State-of-the-Art) 달성
3. **효율화**: DARTS (2019) — 검색 시간 1200 GPU-day → 4 GPU-day로 **300배 단축**
4. **일반화**: RegNet (2020) — 설계 공간을 이론적으로 분석, 고성능 정칙화

이는 **신경망 설계의 민주화**를 의미합니다: 더 이상 전문가만이 아니라, 자동 도구가 고성능 아키텍처를 찾을 수 있습니다.

---

## 📐 수학적 선행 조건

- **Reinforcement Learning**: Policy gradient, 보상 신호
- **Continuous relaxation**: 이산 선택을 연속 최적화로 변환
- **설계 공간**: 하이퍼파라미터의 범위와 제약
- **정칙화 이론(Regularization)**: 과적합 방지

참고: [04-convnext.md](./04-convnext.md)

---

## 📖 직관적 이해

### NAS의 기본 아이디어

**기존 방식** (수작업):
- 전문가가 아키텍처 설계 → 논문 발표 (한 번에 하나)
- 비효율적, 시간 소모적

**NAS 방식**:
- 제한된 검색 공간(search space) 정의
- 자동 탐색 알고리즘 (RL, 진화 알고리즘, 미분)
- 성능 기반 평가 및 선택

### NASNet: Reinforcement Learning 기반 (Zoph et al., 2017)

**구조**:
```
┌─────────────────────────┐
│   RNN Controller        │
│   (NAS policy)          │
│   Token: [Conv type,    │
│           Filter size,  │
│           Stride, ...]  │
└──────────────┬──────────┘
               │ 생성 (샘플)
               ↓
┌─────────────────────────┐
│  Child Network          │
│  (생성된 아키텍처)       │
│  ImageNet CIFAR-10      │
└──────────────┬──────────┘
               │ 평가 (정확도)
               ↓
┌─────────────────────────┐
│  보상 신호              │
│  R = Accuracy           │
└──────────────┬──────────┘
               │ 업데이트
               ↓
         (다시 반복)
```

**문제**: 각 샘플 네트워크를 다시 학습해야 하므로, 500 GPU-day 소요

### DARTS: 미분 가능한 탐색 (Liu et al., 2019)

**핵심 아이디어**: 이산 선택을 연속 완화(continuous relaxation)

**이전 방식** (NASNet):
- 선택: Conv 3×3 vs 5×5 vs MaxPool — 이산
- 각 선택마다 네트워크 재학습

**DARTS**:
- 모든 연산을 동시에 유지 (가중치 $\alpha$ 사용)
- Softmax로 soft selection
- 단일 네트워크로 미분 가능하게 만듦
- 검색 시간: **4 GPU-day** (300배 단축!)

**구현**:
```
일반 연산 o: y = o(x)

DARTS (soft):
y = Σ_i (α_i / Σ_j α_j) · o_i(x)
  = Σ_i softmax(α)_i · o_i(x)
```

여기서 $\alpha_i$는 학습 가능한 파라미터.

### RegNet: 설계 공간 분석 (Radosavovic et al., 2020)

**문제**: NAS의 결과가 일반화되는가?

**접근**: 설계 공간 체계화
```
설계 공간 W = {가능한 모든 아키텍처}
  ├─ Width (채널 수): w_0, w_1, ..., w_L
  ├─ Depth (깊이): d
  ├─ Block type: Conv, ResBlock, MBConv
  └─ Stride pattern: [1, 2, 1, 2, ...]
```

**통찰**: 무작위 샘플링으로도 고성능 모델을 찾을 수 있음
→ **정칙화된 설계 공간(regularized design space)**

---

## ✏️ 엄밀한 정의·정리

### 정의 5.1 — Neural Architecture Search (NAS) 문제

아키텍처 공간 $\mathcal{A}$, 데이터셋 $\mathcal{D}$, 성능 메트릭 $\text{Perf}(\cdot)$에 대해:

$$\mathcal{A}^* = \arg\max_{a \in \mathcal{A}} \text{Perf}(a, \mathcal{D})$$

목표: 제한된 계산 예산 $B$ (GPU-days) 하에서 최적 또는 near-optimal $a$를 찾기.

### 정의 5.2 — Reinforcement Learning 기반 NAS

**상태(State)**: 현재까지 생성한 아키텍처 표현 (token sequence)
$$s_t = [o_1, o_2, \ldots, o_t]$$

**행동(Action)**: 다음 연산 선택
$$a_t \in \{\text{Conv-3×3}, \text{Conv-5×5}, \text{MaxPool}, \ldots\}$$

**보상(Reward)**: 검증 정확도
$$R = \text{Accuracy}(a, \mathcal{D}_{\text{val}})$$

**정책**: RNN 컨트롤러
$$P(a_t | s_t; \theta) = \text{softmax}(\text{RNN}(s_t; \theta))$$

**목표**: 기댓값 최대화
$$\max_\theta \mathbb{E}_{a \sim P(\cdot; \theta)} [R(a)]$$

### 정리 5.3 — DARTS의 연속 완화

아키텍처 $a = (o_1, o_2, \ldots, o_L)$ (각 $o_i$는 연산 선택)을 다음과 같이 완화:

$$\bar{o}(x) = \sum_{i=1}^{|\mathcal{O}|} \frac{\exp(\alpha_i/T)}{\sum_j \exp(\alpha_j/T)} o_i(x)$$

여기서:
- $\alpha \in \mathbb{R}^{|\mathcal{O}|}$: 학습 가능한 아키텍처 파라미터
- $T$: 온도(temperature) (보통 1)
- $\mathcal{O}$: 가능한 연산들 (Conv-3×3, Conv-5×5, MaxPool, ...)

이제 전체 네트워크 손실:
$$\mathcal{L}(w, \alpha) = \mathcal{L}_{\text{train}}(w^*, \alpha) + \lambda \mathcal{L}_{\text{val}}(w^*, \alpha)$$

여기서 $w^* = \arg\min_w \mathcal{L}_{\text{train}}(w, \alpha)$는 가중치 최적화 후.

**최적화**: 교대 최적화
1. $\alpha$ 고정, $w$ 최소화 (일반 훈련)
2. $w$ 고정, $\alpha$ 최소화 (구조 탐색)

### 정의 5.4 — RegNet 설계 공간

정칙화된 설계 공간:
```
Stage i (i=0,1,2,3):
  d_i: 블록 반복 수
  w_i: 채널 수
  b_i: 블록 타입 (bottleneck ratio)
  s_i: Stride (pooling 포함)

선형 규칙:
  w_i = w_0 + w_m * i  (선형 증가)
  
제약:
  - d_i ≥ 1
  - w_i > 0
  - Bottleneck ratio ≥ 1
```

이 제약 하에서 모든 가능한 조합을 탐색.

---

## 🔬 수학적 유도

### DARTS의 이단계 최적화

**Step 1: 가중치 최적화**

고정된 $\alpha$에 대해 훈련 손실 최소화:
$$w^*(α) = \arg\min_w \mathcal{L}_{\text{train}}(w, \alpha)$$

**Step 2: 구조 최적화**

$$\alpha^* = \arg\min_\alpha \mathcal{L}_{\text{val}}(w^*(\alpha), \alpha)$$

근사 (1단계만 하면):
$$\nabla_\alpha \mathcal{L}_{\text{val}} \approx \nabla_\alpha \mathcal{L}_{\text{val}}(w^*(\alpha), \alpha)$$

실제로는:
$$\nabla_\alpha \mathcal{L}_{\text{val}} = \nabla_\alpha \mathcal{L}_{\text{val}} + \frac{d w^*}{d \alpha} \nabla_w \mathcal{L}_{\text{val}}$$

두 번째 항이 Hessian 계산을 요구 → 근사 기법 필요.

### 계산 복잡도 비교

| 방법 | 검색 비용 | 특징 |
|------|---------|------|
| NASNet (RL) | 500 GPU-days | 정확하지만 느림 |
| AmoebaNet (진화) | 3000+ GPU-days | 더 느림 |
| DARTS | 4 GPU-days | 빠르지만 메모리 많이 사용 |
| RegNet (그리드) | 1 GPU-day | 매우 빠름 (사전 계산) |

---

## 💻 실험 재현: NAS 알고리즘

### 간단한 RL 기반 NAS (의사코드)

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ArchitectureController(nn.Module):
    """RNN-based controller for NAS"""
    def __init__(self, vocab_size=10, embedding_dim=100, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_size=1)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, tokens):
        """
        tokens: [seq_len] - 이전까지의 선택
        return: logits [vocab_size] - 다음 선택의 로짓
        """
        x = self.embedding(tokens.unsqueeze(1))  # [seq_len, 1, emb_dim]
        out, _ = self.lstm(x)  # [seq_len, 1, hidden]
        logits = self.fc(out[-1, 0])  # [vocab_size]
        return logits

# 의사 코드: NAS 루프
def nas_search(controller, num_iterations=100):
    """
    아키텍처 탐색
    """
    optimizer = optim.Adam(controller.parameters(), lr=0.01)
    
    best_architecture = None
    best_accuracy = 0
    
    for iteration in range(num_iterations):
        # Step 1: 아키텍처 샘플링
        controller.eval()
        tokens = []
        for step in range(10):  # 길이 10의 시퀀스
            logits = controller(torch.tensor(tokens, dtype=torch.long))
            prob = torch.softmax(logits, dim=-1)
            action = torch.multinomial(prob, 1).item()
            tokens.append(action)
        
        # Step 2: 자식 네트워크 학습 및 평가
        # (생략: 실제로는 여기서 네트워크를 구성하고 학습)
        # accuracy = train_and_evaluate(tokens)
        accuracy = 0.7 + 0.3 * (iteration / num_iterations)  # 모의
        
        # Step 3: 정책 업데이트 (REINFORCE)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_architecture = tokens
        
        controller.train()
        optimizer.zero_grad()
        
        # 보상을 이용한 정책 기울기
        logits = controller(torch.tensor(tokens, dtype=torch.long))
        log_probs = torch.log_softmax(logits, dim=-1)
        # 선택된 행동의 log prob 합
        log_prob = sum(log_probs[t] for t in tokens)
        reward = accuracy
        loss = -log_prob * reward  # REINFORCE 손실
        loss.backward()
        optimizer.step()
        
        if iteration % 10 == 0:
            print(f"Iter {iteration}: Best acc = {best_accuracy:.4f}")
    
    return best_architecture

# 실행 (모의)
controller = ArchitectureController(vocab_size=6)
best_arch = nas_search(controller, num_iterations=50)
```

### DARTS의 간단한 구현

```python
class DARTSCell(nn.Module):
    """DARTS 셀: 모든 연산을 동시에 유지"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # 가능한 모든 연산들
        self.operations = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),  # 3×3
            nn.Conv2d(in_channels, out_channels, 5, stride, 2, bias=False),  # 5×5
            nn.MaxPool2d(3, stride, 1),  # MaxPool
            nn.Identity() if stride == 1 else nn.Identity()  # Skip
        ])
        
        # 아키텍처 파라미터: 각 연산의 중요도
        self.alpha = nn.Parameter(torch.ones(len(self.operations)))
    
    def forward(self, x):
        """Soft mixture of all operations"""
        # Softmax로 가중치 계산
        weights = torch.softmax(self.alpha, dim=-1)
        
        # 모든 연산의 가중합
        out = torch.zeros_like(x)
        for i, op in enumerate(self.operations):
            out = out + weights[i] * op(x)
        
        return out

class DARTSNetwork(nn.Module):
    """DARTS 네트워크"""
    def __init__(self, num_cells=8, in_channels=3, num_classes=10):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        
        self.cells = nn.ModuleList([
            DARTSCell(64, 64, stride=1 if i < 4 else 2)
            for i in range(num_cells)
        ])
        
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        for cell in self.cells:
            x = cell(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# DARTS 훈련
model = DARTSNetwork(num_cells=8)
x = torch.randn(2, 3, 32, 32)
y = model(x)

print(f"Output shape: {y.shape}")
print(f"Alpha (architecture weights): {model.cells[0].alpha.data}")
```

### RegNet: 그리드 탐색

```python
import numpy as np

class RegNetDesignSpace:
    """RegNet design space"""
    def __init__(self):
        # 하이퍼파라미터 범위
        self.w0_range = [8, 16, 32]        # 초기 채널
        self.wa_range = [5, 10, 20]        # 채널 증가율
        self.wm_range = [1.5, 2.0, 2.5]    # 채널 승수
        self.d_range = [2, 4, 6, 8]        # 깊이
        self.b_range = [1, 2, 4]           # Bottleneck ratio
    
    def generate_configs(self):
        """모든 가능한 설정 생성"""
        configs = []
        for w0 in self.w0_range:
            for wa in self.wa_range:
                for wm in self.wm_range:
                    for d in self.d_range:
                        for b in self.b_range:
                            configs.append({
                                'w0': w0, 'wa': wa, 'wm': wm,
                                'd': d, 'b': b
                            })
        return configs
    
    def config_to_network(self, config):
        """설정을 네트워크 구조로 변환"""
        w0, wa, wm, d, b = config['w0'], config['wa'], config['wm'], config['d'], config['b']
        
        # 각 스테이지의 채널 계산
        widths = []
        for i in range(4):  # 4 stages
            w = w0 * (wm ** i)  # 기하급수 증가
            widths.append(int(w))
        
        # 각 스테이지의 깊이 (균등 분배)
        depths = [d // 4] * 4
        
        return {
            'widths': widths,
            'depths': depths,
            'bottleneck_ratio': b
        }

# 탐색
design_space = RegNetDesignSpace()
configs = design_space.generate_configs()

print(f"Total configurations: {len(configs)}")
print(f"\nSample configuration:")
sample = configs[0]
print(f"  w0={sample['w0']}, wa={sample['wa']}, wm={sample['wm']}, d={sample['d']}, b={sample['b']}")

network_spec = design_space.config_to_network(sample)
print(f"  Network spec: {network_spec}")
```

**출력**:
```
Total configurations: 3 × 3 × 3 × 4 × 3 = 324
Sample configuration:
  w0=8, wa=5, wm=1.5, d=2, b=1
  Network spec: {'widths': [8, 12, 17, 25], 'depths': [0, 0, 0, 0], 'bottleneck_ratio': 1}
```

### Timm에서 NAS 기반 모델 사용

```python
import timm

# NASNet
nasnet = timm.create_model('nasnet_large', pretrained=True)
print(f"NASNet-Large: {sum(p.numel() for p in nasnet.parameters()) / 1e6:.2f}M params")

# EfficientNet (AutoML scaling)
efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
print(f"EfficientNet-B0: {sum(p.numel() for p in efficientnet.parameters()) / 1e6:.2f}M params")

# RegNet
regnet = timm.create_model('regnetx_002', pretrained=True)
print(f"RegNetX-200M: {sum(p.numel() for p in regnet.parameters()) / 1e6:.2f}M params")
```

---

## 🔗 이론과 실전의 간극

### 1. 검색과 평가의 비용

**이론**: NAS는 최적 아키텍처를 찾음

**현실**:
- 검색 비용 (500 GPU-day) → 결과를 여러 번 사용해야 정당화
- 각 후보도 수렴할 때까지 훈련 필요
- 하지만 검색 후, 결과는 전이 학습에 사용 가능

**교훈**: NAS는 **원회성(one-time) 투자**; 이후 재사용 가치 높음

### 2. 탐색 공간의 편향

**문제**: 설정한 탐색 공간 내에서만 탐색

예: NASNet의 "cell" 기반 설계
- 장점: 탐색 공간 축소 (컴퓨팅 가능)
- 단점: Cell 구조 외 혁신은 찾을 수 없음

**해결**: RegNet처럼 설계 공간을 일반화

### 3. DARTS의 메모리 문제

**이론**: 모든 연산을 동시에 유지하므로 미분 가능

**실제**: 메모리 사용량 ≈ 모든 연산의 합
- 대규모 모델에서는 메모리 부족 가능
- Trade-off: 속도 vs 메모리

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 탐색 공간이 대표적 | 공간 외 혁신은 못 찾음 |
| 검증 정확도 = 최종 성능 | 검증 셋에 과적합 가능 |
| 전이성(Transferability) | CIFAR-10에서 최적 ≠ ImageNet에서 최적 |
| 공정한 비교 | 검색에 쓰인 훈련 기법이 부당한 이점 줄 수 있음 |
| 단일 메트릭 | 정확도만 고려, FLOPs/메모리 무시 |

---

## 📌 핵심 정리

$$\boxed{\text{NAS: } \arg\max_{a \in \mathcal{A}} \text{Perf}(a, \mathcal{D}) \quad \text{s.t.} \quad \text{Budget} < B}$$

| 방법 | 기법 | 비용 | 장점 |
|------|------|------|------|
| **NASNet** | RL + child training | 500 GPU-days | 높은 정확도 |
| **DARTS** | Continuous relaxation | 4 GPU-days | 극도로 빠름 |
| **RegNet** | 설계 공간 분석 | 1 GPU-day | 매우 효율적, 이해 가능 |
| **AmoebaNet** | 진화 알고리즘 | 3000+ GPU-days | 극도로 높은 성능 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): NASNet의 검색 과정에서, 500 GPU-day 비용은 어디서 발생하는가? (컨트롤러 훈련 vs 자식 네트워크 평가)

<details>
<summary>힌트 및 해설</summary>

분해:
- 컨트롤러 (RNN) 훈련: ~10 GPU-days
- 자식 네트워크 평가: ~490 GPU-days (대부분!)
  - 300개 후보 × 40회 epoch × (ImageNet 훈련 시간) / GPU 수

**최적화 기회**: 
- Early stopping: 약한 후보는 중간에 중단
- Weight sharing: 이전 학습 가중치 재사용 (→ ENAS)

결론: DARTS가 300배 빠른 이유는 "모든 연산 동시 유지" → 단 1개 네트워크만 훈련 ✓

</details>

**문제 2** (심화): DARTS의 아키텍처 파라미터 $\alpha_i$가 어떻게 "선택"이 되는가? Softmax이면 모든 연산이 여전히 사용되지 않는가?

<details>
<summary>힌트 및 해설</summary>

**훈련 중**: Softmax 가중합 → 모든 연산 사용 (soft selection)

**후 처리 (discretization)**:
```python
# 훈련 후 아키텍처 확정
weights = softmax(alpha)
selected_op = argmax(weights)  # 가장 확률 높은 연산만 선택
```

예: $\alpha = [0.1, 2.5, 0.3, -1.2]$ → softmax → $[0.01, 0.98, 0.01, 0.00]$ → argmax → index 1 (선택)

**문제**: Argmax 연산은 미분 불가능
→ 훈련 중에는 softmax (연속), 최종에는 argmax (이산)

이것이 DARTS의 영리한 점 ✓

</details>

**문제 3** (이론-실전): RegNet이 "무작위 탐색"으로도 좋은 모델을 찾는다면, NAS의 의미는 무엇인가?

<details>
<summary>힌트 및 해설</summary>

**RegNet의 통찰**:
- 무작위 탐색: 324개 설정 × 적절한 훈련 = 고성능 모델 찾을 수 있음
- NAS의 의미가 사라진 것 아닌가?

**실제로는**:
1. **NAS가 설계 공간 발견**: NASNet → EfficientNet → RegNet으로 진화
   - 초기 NAS 결과의 패턴 분석 → RegNet의 선형 규칙 발견
   
2. **RegNet은 NAS의 결론**: "이 구조가 좋다"를 이론화한 것
   - 자동 탐색 (NAS) → 지식 추출 → 설계 공간 (RegNet)

3. **여전한 NAS의 역할**:
   - 새로운 영역 (예: 3D CNN, 비전 언어 모델) 탐색
   - Hardware-aware NAS (Mobile, Edge)

결론: **NAS는 초기 탐색, RegNet은 후기 정칙화** ✓

</details>

---

<div align="center">

[◀ 이전](./04-convnext.md) | [📚 README](../README.md) | [다음 ▶](../ch6-applications/01-classification.md)

</div>
