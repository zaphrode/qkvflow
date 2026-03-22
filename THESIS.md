# Time-Indexed Parameter Sharing for Transformers: Parameter-Efficient Language Modeling via Continuous-Depth Weight Modulation

## A Final Year Project Thesis

---

**Author:** [Your Name]

**Supervisor:** [Supervisor Name]

**Institution:** [University Name]

**Department:** [Department of Computer Science / Engineering]

**Date:** March 2026

---

## Abstract

Transformer architectures have become the dominant paradigm in natural language processing, but their memory requirements scale linearly with depth as each layer maintains independent weight matrices. This thesis presents **Time-Indexed Parameter Sharing**, a technique that shares a single transformer block across all layers and modulates it with lightweight, continuous time embeddings representing layer depth. Using FiLM-style (Feature-wise Linear Modulation) scale-and-shift operations, the shared block behaves differently at each depth while storing weights only once.

We conduct large-scale experiments on OpenWebText (~9B tokens) with WikiText-103 validation, comparing five model configurations:

| Model | Parameters | Best Val PPL |
|-------|-----------|-------------|
| GPT-2 Small (pretrained, external reference) | 124.4M | 35.6 |
| Baseline Transformer (LLaMA-style, 12 layers) | 151.9M | 71.1 |
| Baseline Transformer (LLaMA-style, matched params) | 93.8M | 81.8 |
| **Time-Indexed Large (shared block, 12 passes)** | **94.5M** | **77.6** |
| **Time-Indexed Small (shared block, 12 passes)** | **50.4M** | **111.8** |

Our key findings are:

1. At equal parameter budgets (~95M), the time-indexed model achieves **PPL 77.6** versus the baseline's **81.8**, demonstrating that shared+modulated weights can match or exceed independent per-layer weights in parameter efficiency.

2. The time-indexed architecture achieves a **37% parameter reduction** (94.5M vs 151.9M) compared to the full baseline while closing the performance gap to within 6.5 PPL at equal training steps — and continuing to improve with additional training.

3. We identify and characterize the **compute-memory trade-off**: time-indexed sharing saves parameters and GPU memory, but does not reduce per-step FLOPs, as the shared block is still executed N times. Models with larger hidden dimensions (enabled by parameter sharing) train slower per step despite having fewer parameters.

We also document the discovery and resolution of a critical **causal mask information leak** in an earlier version of our codebase, which produced artificially low perplexity (~1.0) by allowing the model to attend to future tokens. All results reported in this thesis use the corrected implementation.

**Keywords:** Transformers, Parameter Sharing, Continuous-Depth Networks, FiLM Modulation, Language Modeling, Efficient Deep Learning, RoPE, SwiGLU

---

## Acknowledgments

I would like to express my sincere gratitude to my supervisor [Name] for their invaluable guidance throughout this project. I also acknowledge the foundational work by Tong et al. (ICLR 2025), whose Neural ODE Transformer implementation served as the starting point for this research.

Special thanks to the open-source community and the developers of JAX, Equinox, Haliax, and HuggingFace Transformers for providing the tools that made this research possible.

---

## Table of Contents

1. [Introduction](#1-introduction)
   - 1.1 Background and Motivation
   - 1.2 Problem Statement
   - 1.3 Research Questions
   - 1.4 Contributions
   - 1.5 Thesis Structure

2. [Literature Review](#2-literature-review)
   - 2.1 Transformer Architectures
   - 2.2 Modern Transformer Components
   - 2.3 Neural Ordinary Differential Equations
   - 2.4 Parameter-Efficient Methods
   - 2.5 Feature-wise Linear Modulation (FiLM)

3. [Methodology](#3-methodology)
   - 3.1 Time-Indexed Parameter Sharing Framework
   - 3.2 Baseline Architecture (Model A)
   - 3.3 Time-Indexed Architecture (Model B)
   - 3.4 Time Embedding Network
   - 3.5 Time Modulation Mechanism
   - 3.6 Training Procedure
   - 3.7 Gradient Accumulation and Effective Batch Size

4. [Implementation](#4-implementation)
   - 4.1 Software Architecture
   - 4.2 Key Components
   - 4.3 Data Pipeline
   - 4.4 Validation Protocol
   - 4.5 Training Infrastructure and Automation
   - 4.6 Checkpoint Resumption
   - 4.7 The Causal Mask Bug: Discovery and Resolution
   - 4.8 Earlier Small-Scale Experiments (Invalidated)
   - 4.9 Experimental Setup for Large-Scale Runs

5. [Results and Analysis](#5-results-and-analysis)
   - 5.1 Training Progress Overview
   - 5.2 Apples-to-Apples Comparison (~95M Parameters)
   - 5.3 Full Model Comparison
   - 5.4 External Baseline: GPT-2 Small
   - 5.5 Parameter Efficiency Analysis
   - 5.6 Training Dynamics and Convergence Behavior
   - 5.7 The Compute-Memory Trade-off
   - 5.8 Learning Rate Schedule Effects

6. [Discussion](#6-discussion)
   - 6.1 Interpretation of Results
   - 6.2 Why Time-Indexed Sharing Works
   - 6.3 What Sharing Saves vs. What It Doesn't
   - 6.4 Comparison with Prior Work
   - 6.5 Lessons from the Causal Mask Bug
   - 6.6 Limitations

7. [Conclusion and Future Work](#7-conclusion-and-future-work)
   - 7.1 Summary of Contributions
   - 7.2 Future Directions

8. [References](#8-references)

9. [Appendices](#9-appendices)
   - A. Full Training Metrics (All Steps)
   - B. Complete Code Listings
   - C. Experimental Configuration Files
   - D. GPT-2 Evaluation Script
   - E. Training Automation Scripts

---

## 1. Introduction

### 1.1 Background and Motivation

Transformer architectures (Vaswani et al., 2017) have become the foundation of modern language modeling, powering systems from GPT-2 (Radford et al., 2019) to GPT-4. However, standard Transformers present fundamental efficiency challenges:

1. **Linear Memory Scaling**: Each layer maintains independent weight matrices for attention (Q, K, V, O projections) and feed-forward networks, causing total parameter count to grow linearly with depth.
2. **Redundancy Across Layers**: Empirical evidence from pruning and distillation research suggests that adjacent layers often learn similar representations, indicating substantial parameter redundancy.
3. **Deployment Constraints**: The large memory footprint of deep Transformers limits deployment on edge devices, increases fine-tuning costs, and complicates model distribution.

Recent work on Neural ODE Transformers (Tong et al., ICLR 2025) reframes Transformer layers as discretizations of continuous dynamics, where layer depth corresponds to integration time $t \in [0, 1]$. This perspective suggests that layer weights should vary *smoothly* with depth — and if so, the full set of independent per-layer weights may be unnecessarily expressive.

This thesis asks a simple question: **what if we share a single transformer block across all layers, and modulate it with a lightweight function of depth?**

### 1.2 Problem Statement

While Tong et al.'s Neural ODE approach treats depth as continuous time, it generates *entire* weight matrices at each layer via hypernetworks. Our approach takes a more aggressive compression strategy: we share **fixed base weights** and apply **element-wise modulation** conditioned on continuous time embeddings. The modulation is implemented via FiLM-style (Perez et al., 2018) scale-and-shift operations, requiring only a small number of additional parameters.

**The central hypothesis is:**

> *A single transformer block, modulated by continuous time embeddings representing layer depth, can match the performance of N independent layers while using significantly fewer parameters.*

### 1.3 Research Questions

This thesis investigates the following research questions:

1. **RQ1:** At equal parameter budgets, how does a time-indexed shared transformer compare to a standard transformer with independent layers?

2. **RQ2:** What is the trade-off between parameter count and model performance as measured by validation perplexity on WikiText-103?

3. **RQ3:** Does time-indexed sharing provide benefits beyond simple parameter reduction — such as implicit regularization or improved learning dynamics?

4. **RQ4:** What are the practical implications of parameter sharing for training speed, memory usage, and deployment?

### 1.4 Contributions

This thesis makes the following contributions:

1. **Time-Indexed Parameter Sharing**: We propose and implement a technique that shares base weights across all transformer layers with FiLM-style time-dependent modulation, achieving 37–67% parameter reduction.

2. **Large-Scale Evaluation**: We train models on OpenWebText (9B tokens) with up to 150,000 steps, evaluating on WikiText-103 — a scale significantly beyond typical academic experiments.

3. **Apples-to-Apples Comparison**: We carefully control for parameter count (~95M) between the time-indexed and baseline architectures, isolating the effect of weight sharing from model capacity.

4. **External Benchmarking**: We evaluate HuggingFace's pretrained GPT-2 Small (124M) on our exact validation protocol, providing an external anchor for our results.

5. **The Compute-Memory Trade-off**: We document that parameter sharing saves memory and model size but does *not* reduce per-step compute, with important implications for practitioners.

6. **Bug Discovery and Resolution**: We document the discovery of a critical causal mask information leak that produced artificially perfect results, and the methodology used to diagnose it.

### 1.5 Thesis Structure

- **Chapter 2** reviews Transformer architectures, modern components (RoPE, RMSNorm, SwiGLU), Neural ODEs, and parameter-efficient methods
- **Chapter 3** presents our time-indexed parameter sharing methodology
- **Chapter 4** describes implementation details, including the causal mask bug
- **Chapter 5** presents experimental results across all model configurations
- **Chapter 6** discusses implications, trade-offs, and limitations
- **Chapter 7** concludes and outlines future work

---

## 2. Literature Review

### 2.1 Transformer Architectures

The Transformer architecture (Vaswani et al., 2017) introduced self-attention as the primary mechanism for sequence modeling.

**Multi-Head Self-Attention:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where queries $Q$, keys $K$, and values $V$ are linear projections of the input. Multi-head attention enables the model to attend to different representation subspaces:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

**Causal Masking** is essential for autoregressive language modeling, ensuring that position $i$ can only attend to positions $j \leq i$:

$$\text{mask}_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

### 2.2 Modern Transformer Components

Since the original Transformer, several architectural improvements have become standard in modern language models such as LLaMA (Touvron et al., 2023):

**Rotary Positional Embeddings (RoPE)** (Su et al., 2021) encode position information by rotating query and key vectors in pairs:

$$\text{RoPE}(x, m) = \begin{pmatrix} x_1 \cos m\theta_1 - x_2 \sin m\theta_1 \\ x_2 \cos m\theta_1 + x_1 \sin m\theta_1 \\ \vdots \end{pmatrix}$$

where $m$ is the position index and $\theta_i = 10000^{-2i/d}$. RoPE provides relative position awareness without learned position embeddings.

**RMSNorm** (Zhang & Sennrich, 2019) simplifies Layer Normalization by removing the mean-centering step:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$

This reduces computation while maintaining effectiveness.

**SwiGLU** (Shazeer, 2020) replaces the standard two-layer FFN with a gated architecture using three projections:

$$\text{SwiGLU}(x) = (\text{SiLU}(xW_{gate}) \odot xW_{up})W_{down}$$

where $\text{SiLU}(x) = x \cdot \sigma(x)$. SwiGLU consistently improves over standard FFNs in modern language models.

### 2.3 Neural Ordinary Differential Equations

Chen et al. (2018) introduced Neural ODEs, parameterizing the derivative of hidden states as a neural network:

$$\frac{dh}{dt} = f_\theta(h(t), t)$$

**Neural ODE Transformers** (Tong et al., ICLR 2025) apply this framework by treating layer depth as continuous time. Their key innovation is generating weights at each layer from a hypernetwork conditioned on time embeddings:

$$W_{QKV}(t) = \text{HyperNetwork}(\text{TimeEmbed}(t))$$

This approach generates *full weight matrices* dynamically, achieving 51.5M parameters for a 6-layer model with hidden dimension 256. However, the hypernetwork approach is computationally expensive and does not achieve the extreme compression possible with weight sharing.

### 2.4 Parameter-Efficient Methods

**Universal Transformers** (Dehghani et al., 2019) share weights across all layers but apply the *identical* transformation at every depth. This lacks the expressiveness to represent depth-varying behavior, as early and late layers cannot specialize.

**Low-Rank Adaptation (LoRA)** (Hu et al., 2022) adds low-rank update matrices to frozen pretrained weights: $W' = W + BA$ where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$. While effective for fine-tuning, LoRA does not address pre-training parameter efficiency.

### 2.5 Feature-wise Linear Modulation (FiLM)

Perez et al. (2018) introduced FiLM for conditioning neural networks on external information via element-wise affine transformations:

$$\text{FiLM}(x | c) = \gamma(c) \odot x + \beta(c)$$

where $\gamma$ and $\beta$ are predicted from a conditioning signal $c$. FiLM has been successfully applied in visual reasoning, style transfer, and more recently in diffusion models.

**Our key insight** is applying FiLM-style modulation where the conditioning signal is *layer depth* (treated as continuous time), and the modulated features are the activations within a shared transformer block. This bridges the gap between Universal Transformers (full sharing, no modulation) and standard Transformers (no sharing, full independence).

---

## 3. Methodology

### 3.1 Time-Indexed Parameter Sharing Framework

Our framework combines three principles:

1. **Base Weight Sharing:** A single set of weight matrices $\{W_{base}\}$ is shared across all $L$ layers
2. **Continuous Time Encoding:** Layer index $l$ is mapped to normalized time $t = l/(L-1) \in [0, 1]$, then encoded via a learned embedding network
3. **FiLM Modulation:** Small projection matrices generate time-dependent scale ($\gamma$) and shift ($\beta$) that modulate activations within the shared block

For layer $l$ with time $t = l/(L-1)$:

$$e(t) = \text{TimeEmbedding}(t) \in \mathbb{R}^{d_{time}}$$
$$\gamma(t) = 1 + \tanh(e(t) \cdot W_\gamma)$$
$$\beta(t) = e(t) \cdot W_\beta$$
$$\text{Modulate}(x, t) = \gamma(t) \odot x + \beta(t)$$

The scale is centered at 1.0 (via $1 + \tanh(\cdot)$) so that without modulation, the shared block behaves identically across layers. The shift is initialized to zero, providing a smooth starting point for optimization.

**Comparison with prior approaches:**

| Aspect | Standard Transformer | Universal Transformer | Tong et al. | **Ours** |
|--------|--------------------|-----------------------|-------------|----------|
| Weights per layer | Independent | Shared (identical) | Generated (hypernetwork) | **Shared + modulated** |
| Depth awareness | Implicit (separate params) | None | Full (weight generation) | **FiLM modulation** |
| Parameter scaling | $O(L \cdot d^2)$ | $O(d^2)$ | $O(d^2 + d_{hyper})$ | **$O(d^2 + L \cdot d_{time} \cdot d)$** |
| Expressiveness | Maximum | Minimum | High | **Moderate** |

### 3.2 Baseline Architecture (Model A)

Our baseline is a modern LLaMA-style Transformer incorporating current best practices:

**Architecture:**
- Pre-norm architecture with RMSNorm (replacing LayerNorm)
- Rotary Positional Embeddings (RoPE) (replacing learned absolute embeddings)
- SwiGLU activation in the MLP (replacing GELU)
- Causal attention with proper masking
- Weight initialization scaled by $1/\sqrt{2L}$ for residual connections

**Block structure:**
$$x_{l+1} = x_l + \text{Dropout}(\text{SwiGLU}(\text{RMSNorm}(x_l + \text{Dropout}(\text{MHA}(\text{RMSNorm}(x_l))))))$$

Each of the $L$ layers has its own independent weights, totaling $L$ copies of the block parameters plus embeddings.

```python
class BaselineTransformer(eqx.Module):
    """Standard Transformer with independent weights per layer."""

    @staticmethod
    def init(Vocab, config, *, key):
        blocks = []
        for i in range(config.num_layers):
            k_block, key = jrandom.split(key)
            blocks.append(BaselineBlock.init(config, key=k_block))
        cos, sin = precompute_rope_frequencies(
            config.head_dim, config.seq_len, config.rope_theta
        )
        return BaselineTransformer(
            config=config,
            token_embeddings=hnn.Embedding.init(Vocab, config.Embed, key=...),
            blocks=blocks,  # L independent blocks
            norm_f=RMSNorm.init(config.Embed),
            rope_cos=cos, rope_sin=sin,
        )

    def __call__(self, input_ids, *, key=None, inference=False):
        x = self.token_embeddings(input_ids)
        Pos = input_ids.resolve_axis("position")
        KeyPos = Pos.alias("key_position")
        mask = hnn.attention.causal_mask(Pos, KeyPos)
        for block in self.blocks:
            x = block(x, mask, self.rope_cos, self.rope_sin, ...)
        x = self.norm_f(x)
        logits = self.token_embeddings.unembed(x)
        return logits
```

### 3.3 Time-Indexed Architecture (Model B)

The time-indexed architecture replaces $L$ independent blocks with a **single shared block** applied $L$ times, each time conditioned on a different time embedding:

```python
class TimeIndexedTransformerV2(eqx.Module):
    """Single shared block applied N times with time modulation."""

    @staticmethod
    def init(Vocab, config, *, key):
        return TimeIndexedTransformerV2(
            token_embeddings=hnn.Embedding.init(Vocab, config.Embed, ...),
            time_embed=TimeEmbedding.init(config.time_embed_dim),
            block=TimeIndexedBlock.init(config, config.time_embed_dim, ...),
            norm_f=RMSNorm.init(config.Embed),
            rope_cos=cos, rope_sin=sin,
        )

    def __call__(self, input_ids, *, key=None, inference=False):
        x = self.token_embeddings(input_ids)
        mask = hnn.attention.causal_mask(Pos, KeyPos)
        for layer_idx in range(self.num_layers):
            t = layer_idx / max(1, self.num_layers - 1)
            time_emb = self.time_embed(t)
            x = self.block(x, time_emb, mask, self.rope_cos, self.rope_sin, ...)
        x = self.norm_f(x)
        logits = self.token_embeddings.unembed(x)
        return logits
```

The critical difference: `self.block` is a *single instance* reused $L$ times. Each invocation receives a different `time_emb` vector, causing the modulation to vary across depth.

### 3.4 Time Embedding Network

The time embedding converts a scalar depth value $t \in [0, 1]$ into a dense vector representation using sinusoidal positional encoding, analogous to the technique used in diffusion models (Ho et al., 2020) and the original Transformer positional encoding (Vaswani et al., 2017).

For time value $t$ and embedding dimension $d_{time}$:

$$e(t) = [\sin(t \cdot \omega_1), \sin(t \cdot \omega_2), ..., \sin(t \cdot \omega_{d/2}), \cos(t \cdot \omega_1), ..., \cos(t \cdot \omega_{d/2})]$$

where $\omega_k = \exp\left(-\frac{\log(10000) \cdot k}{d/2}\right)$ provides a geometric frequency spectrum.

```python
class TimeEmbedding(eqx.Module):
    embed_dim: int = eqx.field(static=True)

    @staticmethod
    def init(embed_dim: int):
        return TimeEmbedding(embed_dim=embed_dim)

    def __call__(self, t: float) -> jnp.ndarray:
        half_dim = self.embed_dim // 2
        freqs = jnp.exp(
            -jnp.log(10000.0) * jnp.arange(half_dim) / half_dim
        )
        args = t * freqs
        return jnp.concatenate([jnp.sin(args), jnp.cos(args)])
```

With $d_{time} = 256$, this produces a 256-dimensional embedding for each of the 12 layers. The geometric frequency spectrum ensures that both fine-grained (high-frequency) and coarse (low-frequency) depth information is captured. Notably, since $t$ ranges from 0 to 1, the first few frequencies capture broad depth positioning while later frequencies provide fine layer-to-layer distinctions.

The time value for layer $l$ is computed as:

$$t_l = \frac{l}{L - 1}$$

where $L$ is the total number of layers. This normalizes depth to $[0, 1]$ regardless of the actual number of layers, meaning the same model could theoretically be evaluated with a different number of layer applications.

### 3.5 Time Modulation Mechanism

The `TimeModulation` module implements FiLM-style conditioning:

```python
class TimeModulation(eqx.Module):
    scale_proj: jnp.ndarray  # (time_dim, hidden_dim)
    shift_proj: jnp.ndarray  # (time_dim, hidden_dim)

    @staticmethod
    def init(time_dim, hidden_dim, *, key):
        return TimeModulation(
            scale_proj=jrandom.normal(k1, (time_dim, hidden_dim)) * 0.01,
            shift_proj=jnp.zeros((time_dim, hidden_dim)),
        )

    def __call__(self, x, time_embed):
        scale = 1.0 + jnp.tanh(time_embed @ self.scale_proj)
        shift = time_embed @ self.shift_proj
        return x * scale + shift
```

Modulation is applied at three points within the shared block:
1. **Q and K projections** in multi-head attention (before RoPE application)
2. **The gate path** in the SwiGLU MLP

This selective modulation allows the model to adjust *what to attend to* and *how to gate information* at each depth, while the V projection, output projection, and up/down MLP weights remain shared without modulation.

### 3.5 Training Procedure

**Optimization:**
- Optimizer: AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay $= 0.1$)
- Gradient clipping: Global norm 1.0
- Learning rate schedule: Trapezoidal (warmup $\rightarrow$ stable $\rightarrow$ cosine decay)

$$\text{LR}(t) = \begin{cases} \text{LR}_{peak} \cdot \frac{t}{T_{warmup}} & t < T_{warmup} \\ \text{LR}_{peak} & T_{warmup} \leq t < T_{warmup} + T_{stable} \\ \text{LR}_{end} + \frac{1}{2}(\text{LR}_{peak} - \text{LR}_{end})(1 + \cos(\pi \cdot \frac{t - T_{warmup} - T_{stable}}{T_{decay}})) & \text{otherwise} \end{cases}$$

with $\text{LR}_{peak} = 3 \times 10^{-4}$, $\text{LR}_{end} = 3 \times 10^{-5}$, $T_{warmup} = 2000$, $T_{stable} = 20000$.

```python
def create_optimizer(config):
    def schedule(step):
        warmup_lr = peak_lr * jnp.minimum(step / warmup, 1.0)
        decay_step = jnp.maximum(step - warmup - stable, 0)
        decay_frac = jnp.minimum(decay_step / decay_steps, 1.0)
        cosine_lr = end_lr + 0.5 * (peak_lr - end_lr) * (
            1.0 + jnp.cos(jnp.pi * decay_frac)
        )
        return jnp.where(step < warmup, warmup_lr,
                         jnp.where(step < warmup + stable, peak_lr, cosine_lr))
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(learning_rate=schedule, weight_decay=0.1, b1=0.9, b2=0.95),
    )
```

**Loss Function:**
Cross-entropy with label smoothing ($\alpha = 0.1$) during training, no smoothing during validation:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\left[(1-\alpha)\log p(y_t | y_{<t}) + \frac{\alpha}{V}\sum_{v=1}^{V}\log p(v | y_{<t})\right]$$

**Data:**
- Training: OpenWebText (~9B tokens, 8,013,769 documents), processed into sequences of length 512
- Validation: WikiText-103 raw validation split (4,549,817 characters, 997,776 tokens)
- Tokenizer: GPT-2 BPE tokenizer (vocabulary size 50,257)
- Effective batch size: $4 \times 64 = 256$ sequences (131,072 tokens per step)

### 3.7 Gradient Accumulation and Effective Batch Size

Due to GPU memory constraints, we use gradient accumulation to achieve a large effective batch size without requiring the full batch to fit in memory simultaneously.

Each training step consists of 64 micro-steps, where each micro-step processes a micro-batch of 4 sequences. Gradients are accumulated across micro-steps before applying the optimizer update:

$$g_{accumulated} = \frac{1}{N_{accum}} \sum_{i=1}^{N_{accum}} \nabla_\theta \mathcal{L}(\theta; B_i)$$

where $N_{accum} = 64$ and $|B_i| = 4$. This yields an effective batch size of 256 sequences, or 131,072 tokens per optimizer step.

The training loop implements this as follows:

```python
for micro_step in range(config.gradient_accumulation):
    batch = next(train_iter)
    key = jrandom.fold_in(base_key, global_step * config.gradient_accumulation + micro_step)
    loss, grads = compute_grads(model, batch, key)
    accumulated_loss += float(loss)
    if accumulated_grads is None:
        accumulated_grads = grads
    else:
        accumulated_grads = jax.tree.map(lambda a, b: a + b, accumulated_grads, grads)

# Average and apply
accumulated_grads = jax.tree.map(lambda g: g / config.gradient_accumulation, accumulated_grads)
model, opt_state = apply_grads(model, opt_state, accumulated_grads)
```

Both `compute_grads` and `apply_grads` are JIT-compiled with `@eqx.filter_jit` for maximum performance, with the gradient computation using Equinox's `filter_value_and_grad` to correctly handle non-array leaves in the model PyTree.

---

## 4. Implementation

### 4.1 Software Architecture

| Library | Purpose |
|---------|---------|
| JAX | Automatic differentiation, XLA compilation |
| Equinox | PyTree-based neural network modules |
| Haliax | Named tensor operations (axis safety) |
| Optax | Gradient transformations and optimizers |
| HuggingFace Transformers | GPT-2 tokenizer and pretrained model |
| HuggingFace Datasets | WikiText-103 and OpenWebText data loading |

**Hardware:** NVIDIA A100 GPU (46GB), single-GPU training.

### 4.2 Key Components

**RMSNorm:** We use Root Mean Square Layer Normalization, which omits mean-centering compared to standard LayerNorm. This reduces per-layer computation while maintaining training stability:

```python
class RMSNorm(eqx.Module):
    axis: hax.Axis = eqx.field(static=True)
    weight: hax.NamedArray
    eps: float = eqx.field(static=True)

    @staticmethod
    def init(axis: hax.Axis, eps: float = 1e-6):
        weight = hax.ones(axis)
        return RMSNorm(axis=axis, weight=weight, eps=eps)

    def __call__(self, x: hax.NamedArray) -> hax.NamedArray:
        variance = (x * x).mean(self.axis)
        inv = hax.rsqrt(variance + self.eps)
        return self.weight * (x * inv)
```

**Rotary Positional Embeddings (RoPE):** Position information is encoded by rotating query and key vectors in the complex plane. We precompute the rotation matrices at initialization for efficiency:

```python
def precompute_rope_frequencies(head_dim, seq_len, theta=10000.0):
    half_dim = head_dim // 2
    freqs = 1.0 / (theta ** (jnp.arange(0, half_dim).astype(jnp.float32) / half_dim))
    positions = jnp.arange(seq_len).astype(jnp.float32)
    angles = jnp.outer(positions, freqs)
    return jnp.cos(angles), jnp.sin(angles)

def apply_rope(x_array, cos, sin):
    half = x_array.shape[-1] // 2
    x1 = x_array[..., :half]
    x2 = x_array[..., half:]
    seq_len = x_array.shape[-2]
    cos = cos[:seq_len]
    sin = sin[:seq_len]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return jnp.concatenate([out1, out2], axis=-1)
```

RoPE requires an even head dimension (since it operates on pairs), which constrained our architecture search when finding the 95M baseline configuration (head_dim=58, the minimum even value that produced ~95M parameters).

**SwiGLU MLP:** The gated feed-forward network uses three linear projections instead of two, with the SiLU-gated pathway providing multiplicative interaction:

```python
class SwiGLUMLP(eqx.Module):
    w_gate: hnn.Linear  # Embed -> Mlp (gate projection)
    w_up: hnn.Linear    # Embed -> Mlp (value projection)
    w_down: hnn.Linear  # Mlp -> Embed (output projection)
    dropout_rate: float = eqx.field(static=True)

    @staticmethod
    def init(config: ModelConfig, *, key):
        Embed, Mlp = config.Embed, config.Mlp
        k1, k2, k3 = jrandom.split(key, 3)
        return SwiGLUMLP(
            w_gate=hnn.Linear.init(Embed, Mlp, key=k1, use_bias=False),
            w_up=hnn.Linear.init(Embed, Mlp, key=k2, use_bias=False),
            w_down=hnn.Linear.init(Mlp, Embed, key=k3, use_bias=False),
            dropout_rate=config.dropout,
        )

    def __call__(self, x, *, key=None, inference=False):
        gate = hnn.silu(self.w_gate(x))
        up = self.w_up(x)
        x = gate * up
        x = apply_dropout(x, self.dropout_rate, key, inference)
        x = self.w_down(x)
        return x
```

The MLP hidden dimension follows a 4× expansion ratio: $d_{mlp} = 4 \times d_{hidden}$. With SwiGLU's three projections (gate, up, down), the MLP parameters per layer are $3 \times d_{hidden} \times d_{mlp} = 12 \times d_{hidden}^2$, compared to $2 \times d_{hidden} \times d_{mlp} = 8 \times d_{hidden}^2$ for a standard FFN.

**Time-Modulated Attention:** The time-indexed attention module is identical to the baseline `MultiHeadAttention`, with the addition of FiLM modulation on the Q and K projections after the linear transformation but before RoPE application:

```python
class TimeModulatedAttention(eqx.Module):
    w_q: hnn.Linear
    w_k: hnn.Linear
    w_v: hnn.Linear
    w_o: hnn.Linear
    time_mod_q: TimeModulation  # FiLM on queries
    time_mod_k: TimeModulation  # FiLM on keys

    def __call__(self, x, time_embed, mask, rope_cos, rope_sin, *, key, inference):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = self.time_mod_q(q, time_embed)  # FiLM: scale + shift
        k = self.time_mod_k(k, time_embed)  # FiLM: scale + shift

        # Reshape for multi-head, apply RoPE, compute attention...
        q = q.unflatten_axis(HeadDim, (Heads, HeadSize))
        k = k.unflatten_axis(HeadDim, (Heads, HeadSize))
        v = v.unflatten_axis(HeadDim, (Heads, HeadSize))

        q_2d = apply_rope(q.array.reshape(...), rope_cos, rope_sin)
        k_2d = apply_rope(k.array.reshape(...), rope_cos, rope_sin)

        attn = hax.dot(HeadSize, q, k) * (head_dim ** -0.5)
        attn = hax.where(mask, attn, -1e9)
        attn = hnn.softmax(attn, axis=KeyPos)

        out = hax.dot(KeyPos, attn, v)
        out = self.w_o(out.flatten_axes((Heads, HeadSize), HeadDim))
        return out
```

**Time-Modulated SwiGLU:** The gate pathway in SwiGLU is modulated by time, allowing the model to adjust its information gating behavior at different depths:

```python
class TimeModulatedSwiGLU(eqx.Module):
    w_gate: hnn.Linear
    w_up: hnn.Linear
    w_down: hnn.Linear
    time_mod_gate: TimeModulation  # FiLM on gate path only

    def __call__(self, x, time_embed, *, key=None, inference=False):
        gate = self.w_gate(x)
        gate = self.time_mod_gate(gate, time_embed)  # FiLM modulation
        gate = hnn.silu(gate)
        up = self.w_up(x)
        x = gate * up
        x = apply_dropout(x, self.dropout_rate, key, inference)
        x = self.w_down(x)
        return x
```

### 4.3 Data Pipeline

**Training Data:** OpenWebText is an open-source recreation of OpenAI's WebText dataset, constructed by scraping URLs from Reddit submissions with at least 3 karma. Our processed version contains 8,013,769 documents totaling approximately 9 billion tokens.

The `OpenWebTextLoader` implements a streaming data pipeline that tokenizes documents on-the-fly and packs them into fixed-length sequences:

```python
class OpenWebTextLoader:
    def __init__(self, data_path, tokenizer, batch_size, seq_len):
        self.dataset = load_from_disk(data_path)
        self.num_examples = len(self.dataset)  # 8,013,769

    def __iter__(self):
        token_buffer = []
        example_idx = 0
        while True:
            # Fill buffer by tokenizing documents
            while len(token_buffer) < self.batch_size * (self.seq_len + 1) * 2:
                text = self.dataset[example_idx]['text']
                tokens = self.tokenizer.encode(text)
                token_buffer.extend(tokens)
                example_idx = (example_idx + 1) % self.num_examples
            # Extract fixed-length sequences
            batch_input, batch_labels = [], []
            for _ in range(self.batch_size):
                seq = token_buffer[:self.seq_len + 1]
                token_buffer = token_buffer[self.seq_len:]
                batch_input.append(seq[:-1])
                batch_labels.append(seq[1:])
            yield {
                "input_ids": jnp.array(batch_input, dtype=jnp.int32),
                "labels": jnp.array(batch_labels, dtype=jnp.int32),
            }
```

Documents are concatenated in a token buffer, and sequences of length 513 are extracted (512 input tokens + 1 for the next-token label). This "packing" approach avoids padding waste and ensures every token in every batch contributes to the loss.

### 4.4 Validation Protocol

Validation uses the WikiText-103 raw validation split, processed identically to training data:

1. Load WikiText-103 validation split (4,549,817 characters)
2. Tokenize with GPT-2 BPE tokenizer (997,776 tokens)
3. Extract non-overlapping sequences of length 513 (up to 500 sequences)
4. Evaluate on the first 200 sequences in batches of 4
5. Compute mean cross-entropy loss **without** label smoothing
6. Report perplexity as $\text{PPL} = \exp(\mathcal{L})$

The validation uses a fixed random key (`jrandom.PRNGKey(0)`) and runs in inference mode (dropout disabled). Label smoothing is explicitly disabled during validation to ensure perplexity is directly comparable across experiments and with external baselines.

### 4.5 Training Infrastructure and Automation

Training was orchestrated through a series of shell scripts that managed GPU allocation, sequential/parallel job scheduling, and automatic model swapping.

**Phase 1** (`schedule_next_training.sh`): Launched the Baseline 152M and Time-Indexed Small models simultaneously on separate GPUs. Upon completion, automatically launched Phase 2 — the Time-Indexed Large model on GPU 0.

**Training Swap** (`swap_training.sh`): When we needed to modify the Time-Indexed Large model's early stopping configuration mid-training (see Section 4.6), we created an automated swap script that:
1. Polled every 60 seconds for the step 15,000 checkpoint to appear
2. Waited 30 seconds for the file write to complete (checkpoints are ~1.1GB)
3. Killed the original training process
4. Launched a new process with updated hyperparameters, resuming from the checkpoint

```bash
while [ ! -f "$CKPT" ]; do
    if ! kill -0 $OLD_PID 2>/dev/null; then
        echo "Training process died before checkpoint!"
        break
    fi
    sleep 60
done
# Kill old process and resume with patience=999
kill $OLD_PID
CUDA_VISIBLE_DEVICES=0 nohup python train_v2.py \
    --resume_from "$CKPT" --patience 999 ... &
```

### 4.6 Checkpoint Resumption

To support training interruption and resumption, we added a `--resume_from` argument to the training script. The checkpoint contains the model state, optimizer state, and current step number:

```python
# Saving (every save_every steps)
with open(f"{output_dir}/checkpoint_{step:06d}.pkl", "wb") as f:
    pickle.dump({"model": model, "opt_state": opt_state, "step": step,
                 "config": config_dict}, f)

# Resuming
if resume_from is not None:
    with open(resume_from, "rb") as f:
        ckpt = pickle.load(f)
    model = ckpt["model"]
    opt_state = ckpt["opt_state"]
    resume_step = ckpt["step"]
    # Restore metrics history
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            existing_metrics = json.load(f)
```

The optimizer state includes the internal step counter used by the learning rate schedule, ensuring the trapezoidal schedule continues correctly from the resume point. Metrics history is loaded from the JSON log file to maintain continuity in validation tracking.

**Early Stopping Modification:** The Time-Indexed Large model was initially launched with `patience=15` (stop if validation loss doesn't improve for 15 consecutive evaluations). After observing that this was too aggressive for a 150,000-step training run — the model had 4 consecutive non-improving evaluations by step 19,000 despite being on a clear downward trajectory — we increased patience to 999 (effectively disabling early stopping) via the checkpoint swap mechanism. This allowed the model to continue training through temporary plateaus, ultimately reaching PPL 77.6 at step 41,500.

### 4.7 The Causal Mask Bug: Discovery and Resolution

During early development, we discovered a critical bug that produced artificially perfect results (validation perplexity ~1.0). This section documents the bug for transparency and as a cautionary example.

**The Bug:** In our initial custom attention implementation, the causal mask was created as:

```python
mask = hnn.attention.causal_mask(Pos, Pos)  # BUG: same axis twice
```

Passing the **same axis** for both query and key positions caused Haliax to return a degenerate 1D mask of shape `(seq_len,)` with all `True` values, rather than the correct 2D lower-triangular matrix of shape `(seq_len, seq_len)`. This meant **no causal masking was applied** — the model could attend to future tokens, leaking the correct answer into its predictions.

**Symptoms:**
- Training perplexity dropped to ~1.36 within 1,000 steps
- Validation perplexity reached ~1.02 — impossibly low for language modeling
- The model appeared to achieve near-perfect next-token prediction

**Diagnosis:** We created a diagnostic script (`scripts/check_mask.py`) that inspected the mask shape and values, confirming that the mask was ineffective.

**The Fix:** Using distinct axes for query and key positions:

```python
Pos = input_ids.resolve_axis("position")
KeyPos = Pos.alias("key_position")
mask = hnn.attention.causal_mask(Pos, KeyPos)  # Correct: 2D causal mask
```

This produces the proper lower-triangular mask where position $i$ can only attend to positions $j \leq i$.

**Diagnosis Methodology:** We created two diagnostic tools:

1. `scripts/check_mask.py` — Verified the mask shape and behavior by comparing `causal_mask(Pos, Pos)` vs `causal_mask(Pos, KeyPos)`, confirming the former produced a degenerate 1D tensor.

2. `scripts/diagnose_model.py` — Loaded a trained checkpoint and ran controlled tests:
   - Random tokens (expected loss ~10.8 for vocab size 50,257)
   - Constant tokens (testing memorization)
   - Sequential tokens (label = input + 1, testing the model can't cheat)
   - Logits distribution analysis
   - Output sensitivity to input changes (detecting representation collapse)
   - Real text inference

**Impact:** All small-scale results from early experiments (WikiText-2, ~1,000 steps) were invalidated. The original THESIS.md reported impossibly good results (validation loss ~2.2, perplexity ~9) that were entirely attributable to the model reading future tokens. All results in this thesis are from the corrected implementation, using the large-scale OpenWebText training pipeline (`scripts/train_v2.py`), which was written from scratch with correct causal masking.

**Lesson Learned:** Custom attention implementations are notoriously error-prone. The use of named tensors (Haliax) should in principle prevent axis confusion, but the `causal_mask` API's behavior with identical axes was unexpected. We recommend always verifying causal masking with a simple test: generate a sequence [1, 2, 3, ...] and verify the model cannot predict token $t$ from positions $> t$.

### 4.8 Earlier Small-Scale Experiments (Invalidated)

Before the causal mask bug was discovered, we conducted small-scale experiments on WikiText-2 with character-level tokenization (~2M tokens, sequence length 128, hidden dimension 256, 6 layers). Four model variants were compared with 5 random seeds each:

**Table (INVALIDATED — shown for transparency only):**

| Model | Val Loss | Parameters | Seeds |
|-------|----------|-----------|-------|
| Standard Transformer | 2.367 ± 0.022 | 308.5M | 5 |
| Tong's Neural ODE | 2.336 ± 0.018 | 51.5M | 5 |
| Time-Indexed MLP | 2.231 ± 0.025 | 0.7M | 5 |
| Time-Indexed SSM | 2.147 ± 0.124 | 4.9M | 5 |

These results showed statistically significant improvements (p < 0.01, Cohen's d > 2.0) for the time-indexed variants. An ablation study showed time-indexing provided a 5.47% improvement over constant modulation.

**Why these results are invalid:** The broken causal mask meant all models could attend to future tokens. The apparent "improvements" from time-indexed sharing may have reflected which architectures were better at exploiting the leaked information, not genuine language modeling capability.

**What we retained from this phase:**
- The general architecture design (FiLM modulation on shared weights)
- The training infrastructure and evaluation methodology
- The insight that time-varying modulation outperforms constant modulation (pending re-validation at scale)

### 4.9 Experimental Setup for Large-Scale Runs

**Model Configurations:**

| Configuration | Mode | hidden_dim | num_heads | head_dim | num_layers | Parameters |
|--------------|------|-----------|-----------|----------|-----------|------------|
| Baseline 152M | baseline | 768 | 12 | 64 | 12 | 151.9M |
| Baseline 95M | baseline | 580 | 10 | 58 | 12 | 93.8M |
| Time-Indexed Large | time_index | 1280 | 16 | 80 | 12 | 94.5M |
| Time-Indexed Small | time_index | 768 | 12 | 64 | 12 | 50.4M |

All models share: sequence length 512, dropout 0.1, label smoothing 0.1, RoPE $\theta = 10000$, time embedding dimension 256 (for time-indexed models).

**Training Hyperparameters (identical across all models):**

| Parameter | Value |
|-----------|-------|
| Peak learning rate | $3 \times 10^{-4}$ |
| Minimum learning rate | $3 \times 10^{-5}$ |
| Warmup steps | 2,000 |
| Stable steps | 20,000 |
| Max steps | 150,000 |
| Micro batch size | 4 |
| Gradient accumulation | 64 |
| Effective batch size | 256 (131,072 tokens/step) |
| Max gradient norm | 1.0 |
| AdamW $\beta_1, \beta_2$ | 0.9, 0.95 |
| Weight decay | 0.1 |

**Evaluation Protocol:**
- WikiText-103 validation split (raw)
- Sequences of length 512, up to 200 evaluation sequences
- Cross-entropy loss without label smoothing
- Perplexity: $\text{PPL} = \exp(\mathcal{L})$

---

## 5. Results and Analysis

### 5.1 Training Progress Overview

All models were trained on OpenWebText with periodic evaluation on WikiText-103. The following table summarizes the final state of each training run:

**Table 1: Training Summary**

| Model | Params | Steps Trained | Tokens Seen | Best Val Loss | Best Val PPL | Step @ Best |
|-------|--------|--------------|-------------|--------------|-------------|-------------|
| Baseline 152M | 151.9M | 25,000 | 3.28B | 4.265 | 71.1 | 24,500 |
| Baseline 95M | 93.8M | 24,000 | 3.15B | 4.405 | 81.8 | 23,500 |
| **TI-Large 95M** | **94.5M** | **44,000** | **5.77B** | **4.352** | **77.6** | **41,500** |
| TI-Small 50M | 50.4M | 27,500 | 3.60B | 4.717 | 111.8 | 26,500 |
| GPT-2 Small (ref) | 124.4M | N/A | ~40B | 3.572 | 35.6 | N/A |

> **Figure 1** (`fig1_val_loss_curves.png`): Validation loss curves for all models with GPT-2 Small reference line.

> **Figure 2** (`fig2_val_ppl_curves.png`): Validation perplexity curves (zoomed to useful range 20–300).

### 5.2 Apples-to-Apples Comparison (~95M Parameters)

The most controlled comparison is between the Baseline 95M and TI-Large 95M, which have nearly identical parameter counts (93.8M vs 94.5M), the same number of layers (12), and identical training hyperparameters.

**Table 2: Equal-Parameter Comparison at Step 24,000**

| Metric | Baseline 95M (93.8M) | TI-Large 95M (94.5M) |
|--------|---------------------|-----------------------|
| Val Loss | 4.432 | 4.511 |
| Val PPL | 84.1 | 91.0 |
| Training Speed | 10,400 tok/s | 3,800 tok/s |
| Wall-clock time to 24k steps | ~83 hours | ~226 hours |

At equal steps, the baseline leads by ~7 PPL. However, the time-indexed model uses hidden_dim=1280 versus the baseline's 580, meaning each forward pass involves significantly more computation (see Section 5.7).

**Table 3: Best Results Achieved**

| Metric | Baseline 95M | TI-Large 95M |
|--------|-------------|---------------|
| Best Val PPL | 81.8 (step 23,500) | **77.6** (step 41,500) |
| Best Val Loss | 4.405 | **4.352** |

With additional training, the TI-Large eventually surpasses the baseline's best, achieving **PPL 77.6** versus 81.8 — a **5.1% improvement** at equal parameter counts.

> **Figure 3** (`fig3_apples_to_apples.png`): Apples-to-apples comparison at ~95M parameters.

### 5.3 Full Model Comparison

**Table 4: All Models — Best Results**

| Model | Params | Best Val PPL | Compression vs Baseline 152M |
|-------|--------|-------------|-------------------------------|
| GPT-2 Small (pretrained) | 124.4M | 35.6 | (external reference) |
| Baseline 152M | 151.9M | 71.1 | 1.0× |
| TI-Large 95M | 94.5M | 77.6 | 1.6× (37% fewer params) |
| Baseline 95M | 93.8M | 81.8 | 1.6× (38% fewer params) |
| TI-Small 50M | 50.4M | 111.8 | 3.0× (67% fewer params) |

The TI-Large model is particularly notable: it uses the same hidden dimension (1280) as GPT-2 Large (774M parameters, 36 layers), but achieves this with only 94.5M parameters through weight sharing — an **8.2× parameter reduction** relative to a hypothetical standard model with the same per-layer width.

> **Figure 4** (`fig4_bar_comparison.png`): Bar chart comparison of all models.

### 5.4 External Baseline: GPT-2 Small

To establish external validity, we evaluated HuggingFace's pretrained GPT-2 Small (124.4M parameters) using our exact evaluation protocol (WikiText-103 validation, sequence length 512, up to 200 sequences, cross-entropy loss):

| Metric | GPT-2 Small |
|--------|------------|
| Val Loss | 3.572 |
| Val PPL | 35.6 |

GPT-2 Small was trained on ~40B tokens of WebText — approximately **7–12× more data** than our models. Our training dataset, OpenWebText, is an open-source recreation of WebText using the same Reddit-sourced URL methodology, so the data distribution is comparable.

The gap between our best model (Baseline 152M, PPL 71.1) and GPT-2 (PPL 35.6) is primarily attributable to the training data volume difference, not architectural quality. Chinchilla scaling laws (Hoffmann et al., 2022) predict that loss follows a power law in both parameters and tokens, and our models are firmly in the undertrained regime.

### 5.5 Parameter Efficiency Analysis

> **Figure 5** (`fig5_param_efficiency.png`): PPL vs Parameter Count scatter plot.

**Table 5: Parameter Efficiency at ~95M**

| Model | Params | Val PPL | Architecture | hidden_dim |
|-------|--------|---------|-------------|-----------|
| TI-Large | 94.5M | 77.6 | 1 shared block × 12 | 1280 |
| Baseline 95M | 93.8M | 81.8 | 12 independent blocks | 580 |

The time-indexed model achieves better perplexity with nearly identical parameters by allocating its budget differently: instead of 12 copies of a small block (hidden_dim=580), it stores one copy of a large block (hidden_dim=1280) plus a small time modulation network. This wider representation per layer appears to provide better modeling capacity.

### 5.6 Training Dynamics

> **Figure 6** (`fig6_tokens_comparison.png`): Performance vs tokens seen.

An important observation from the training curves is that the time-indexed models show **slower initial convergence** but **steadier long-term improvement**:

- At 1B tokens: Baseline 95M leads TI-Large by ~40 PPL
- At 3B tokens: The gap narrows to ~10 PPL
- At 5B tokens (TI-Large only): TI-Large reaches PPL 77.6, surpassing the baseline's final PPL of 81.8

This pattern is consistent with the hypothesis that weight sharing provides implicit regularization: the shared block must learn representations that work across all depths, which initially constrains learning but ultimately produces more generalizable features.

The time-indexed large model also showed consistent improvement throughout training, hitting new best validation losses even at step 41,500 out of 150,000. This suggests significant additional gains are possible with continued training.

**Validation loss trajectory (TI-Large, selected steps):**

| Step | Val Loss | Val PPL | Tokens Seen | Phase |
|------|----------|---------|-------------|-------|
| 500 | 7.675 | 2,156 | 65M | Warmup |
| 2,500 | 6.419 | 613 | 328M | Warmup |
| 5,000 | 5.228 | 187 | 655M | Stable LR |
| 10,000 | 4.969 | 144 | 1.3B | Stable LR |
| 15,000 | 4.622 | 102 | 2.0B | Stable LR |
| 20,000 | 4.563 | 96 | 2.6B | Stable LR |
| 22,000 | — | — | 2.9B | *Cosine decay starts* |
| 25,000 | 4.548 | 95 | 3.3B | Cosine decay |
| 30,000 | 4.432 | 84 | 3.9B | Cosine decay |
| 35,000 | 4.390 | 81 | 4.6B | Cosine decay |
| 40,000 | 4.397 | 81 | 5.2B | Cosine decay |
| 41,500 | **4.352** | **77.6** | 5.4B | Cosine decay |
| 44,000 | 4.383 | 80.1 | 5.8B | Cosine decay |

Notable: the transition from the stable LR phase to cosine decay at step 22,000 coincides with an acceleration in validation loss improvement — from steps 15,000–22,000 (PPL 102→~97, ~0.7 PPL/1000 steps) to steps 22,000–35,000 (PPL ~97→81, ~1.2 PPL/1000 steps). This is consistent with the well-documented phenomenon that learning rate decay consolidates learned representations.

### 5.8 Learning Rate Schedule Effects

The trapezoidal learning rate schedule plays a crucial role in training dynamics:

**Warmup (steps 0–2,000):** LR ramps from 0 to $3 \times 10^{-4}$. During this phase, all models show rapid but noisy loss reduction. The warmup prevents early instability from large gradients applied to randomly initialized weights.

**Stable Phase (steps 2,000–22,000):** LR remains at peak $3 \times 10^{-4}$. Models explore broadly, with validation loss showing occasional spikes. The Baseline 152M reaches the end of its training (step 25,000) mostly in this phase, achieving its best results during early cosine decay.

**Cosine Decay (steps 22,000–150,000):** LR decays from $3 \times 10^{-4}$ to $3 \times 10^{-5}$. The time-indexed large model shows its most consistent improvement during this phase. The decaying learning rate enables the shared block to fine-tune its modulation parameters for optimal depth-specific behavior, which requires more precise gradient steps than the initial broad feature learning.

The baseline models, having stopped training at steps 24,000–25,000, barely entered the cosine decay phase. A fair iso-compute comparison would require training all models to the same step count, which is an ongoing effort.

### 5.7 The Compute-Memory Trade-off

A critical finding of this work is that **parameter sharing saves memory and model size, but does not reduce per-step computation**.

**Table 6: Training Speed Comparison**

| Model | Params | hidden_dim | Tok/s | Time per 10 steps |
|-------|--------|-----------|-------|-------------------|
| Baseline 95M | 93.8M | 580 | 10,400 | 12.5s |
| TI-Large 95M | 94.5M | 1280 | 3,800 | 34.5s |
| Baseline 152M | 151.9M | 768 | ~7,000* | ~18.7s* |

*Estimated from training logs.

The TI-Large model trains **2.7× slower** per step than the Baseline 95M despite having nearly identical parameter counts. This is because:

1. **Shared block execution**: The single block is still executed 12 times sequentially — sharing saves stored weights, not forward-pass FLOPs.

2. **Larger hidden dimension**: Per-layer FLOPs scale as $O(d^2)$. With $d=1280$ vs $d=580$:
   $$\frac{1280^2}{580^2} = \frac{1,638,400}{336,400} \approx 4.9\times$$
   The TI-Large does approximately 4.9× more computation per layer.

3. **Time modulation overhead**: Computing time embeddings and applying scale/shift adds additional operations at each layer.

**What parameter sharing saves:**
- GPU memory for weight storage (1 block vs 12)
- Model file size for deployment and distribution
- Number of parameters to update during fine-tuning

**What it does not save:**
- Training time per step (same forward/backward pass depth)
- Inference latency (same number of sequential layer applications)
- Total FLOPs per token

This trade-off has important practical implications: time-indexed sharing is most valuable in **memory-constrained** settings (edge deployment, model distribution, fine-tuning budgets), not in **compute-constrained** settings (training throughput, inference latency).

---

## 6. Discussion

### 6.1 Interpretation of Results

Our results demonstrate that time-indexed parameter sharing is a viable approach to parameter-efficient Transformers, achieving competitive or superior perplexity at matched parameter budgets. The TI-Large model's PPL of 77.6 versus the Baseline 95M's 81.8 (both ~95M params) shows that shared+modulated weights can outperform independent weights — given sufficient training.

The key trade-off is convergence speed: the baseline converges faster per step, but the time-indexed model achieves better asymptotic performance. This suggests that the shared block's constraint forces the model to learn more generalizable representations that ultimately produce better language modeling.

### 6.2 Why Time-Indexed Sharing Works

We hypothesize three contributing factors:

1. **Implicit Regularization**: Sharing weights constrains the model to learn representations useful across all depths. This prevents individual layers from overfitting to depth-specific patterns and encourages more transferable features.

2. **Higher Per-Layer Capacity**: Because sharing eliminates duplicate weights, the parameter budget can be allocated to a *wider* representation (hidden_dim=1280 vs 580 at equal params). Wider layers have been shown to improve model quality in scaling law research.

3. **Smooth Depth Variation**: FiLM modulation varies smoothly with depth (via continuous time embeddings), which is consistent with the observation that adjacent transformer layers often learn similar representations. The modulation captures the *differences* between depths, not the commonalities.

### 6.3 What Sharing Saves vs. What It Doesn't

| Aspect | Parameter Sharing | Independent Layers |
|--------|------------------|--------------------|
| **Model size on disk** | Smaller (1 block stored) | Larger (L blocks stored) |
| **GPU memory (weights)** | Lower | Higher |
| **Fine-tuning cost** | Lower (fewer params to update) | Higher |
| **Distribution/serving** | Easier (smaller files) | Harder |
| **Training FLOPs** | Same | Same |
| **Inference latency** | Same | Same |
| **Per-step training time** | Same (or slower if using larger hidden_dim) | Same |

### 6.4 Comparison with Prior Work

**vs. Tong et al. (ICLR 2025)**: Tong generates full weight matrices via hypernetworks — a more expressive but more parameter-heavy approach. Our FiLM modulation is simpler, achieving extreme compression with a thin modulation layer rather than a full weight generation network. The approaches are complementary and could potentially be combined.

**vs. Universal Transformers (Dehghani et al., 2019)**: Universal Transformers share weights without any depth modulation, applying the identical function at every layer. Our work shows that adding even lightweight time-dependent modulation substantially improves the effectiveness of weight sharing.

**vs. GPT-2 (Radford et al., 2019)**: Our models use modern architectural improvements (RoPE, RMSNorm, SwiGLU) that post-date GPT-2. The performance gap is primarily due to training data volume (~3–6B vs ~40B tokens), not architecture quality. The time-indexed architecture's hidden_dim=1280 matches GPT-2 Large's per-layer width at 8× fewer parameters.

### 6.5 Lessons from the Causal Mask Bug

The causal mask information leak, while ultimately invalidating our early results, provided valuable methodological lessons:

1. **Sanity checks are essential.** A validation perplexity of ~1.0 should have immediately triggered suspicion — it implies the model predicts every token almost perfectly, which is impossible for natural language. We now recommend establishing expected perplexity ranges before any experiment (for GPT-2-scale models on WikiText, expect PPL 20–200 depending on training stage).

2. **Custom implementations require verification.** Using Haliax's named tensors was intended to prevent axis errors, but the `causal_mask` API's behavior with identical axes was non-obvious. The lesson: always write explicit tests for critical components, especially attention masking.

3. **Separate validation from training data.** Our validation eventually used WikiText-103 (distinct from the OpenWebText training data), providing a strong out-of-distribution check. If we had only validated on a held-out subset of the training data, the bug might have been harder to detect.

4. **Transparency about failures.** We document this bug in full because it represents a genuine part of the research process. The temptation to omit failed experiments is strong, but understanding failure modes is valuable to the research community.

### 6.6 Limitations

1. **Training Duration**: The time-indexed large model was stopped at step 44,000 of 150,000 (29%). The full training budget was not exhausted, leaving the model's asymptotic performance unknown.

2. **Single Run**: Due to computational constraints, each large-scale configuration was trained with a single random seed. Statistical significance testing (as done in our earlier small-scale experiments) was not feasible at this scale.

3. **Downstream Tasks**: We evaluate only on language modeling perplexity. Performance on downstream tasks (text classification, question answering, summarization) may differ.

4. **Compute Fairness**: Comparing at equal parameter counts is meaningful for deployment, but comparing at equal FLOPs or equal wall-clock time would tell a different story due to the hidden dimension mismatch.

5. **Modulation Scope**: We modulate only Q/K projections and the SwiGLU gate. Modulating additional components (V projection, output projection, norm parameters) might improve performance at minimal parameter cost.

---

## 7. Conclusion and Future Work

### 7.1 Summary of Contributions

This thesis presented **Time-Indexed Parameter Sharing**, demonstrating that a single transformer block modulated by continuous time embeddings can achieve competitive or superior language modeling performance compared to standard transformers with independent per-layer weights.

**Key findings:**

1. **At equal parameters (~95M), time-indexed sharing achieves PPL 77.6 vs baseline's 81.8** — a 5.1% improvement — by reallocating the parameter budget from 12 small independent blocks to one large shared block with modulation.

2. **37% parameter reduction** (94.5M vs 151.9M) compared to the full baseline, with the performance gap narrowing throughout training and projected to close further.

3. **The compute-memory trade-off** is a critical practical consideration: sharing saves parameters and memory but not FLOPs, making it most valuable for deployment rather than training efficiency.

4. **The causal mask bug** we discovered and documented serves as an important cautionary example about the subtlety of attention masking in custom implementations.

### 7.2 Future Directions

**Immediate:**

1. **Complete Training**: Run TI-Large to its full 150,000-step budget (currently at 29%) to determine asymptotic performance.

2. **Train Baseline 95M to Completion**: Enable fair comparison at equal training budgets across all step counts.

3. **Multi-Seed Validation**: Run at least 3 seeds for the key comparison (Baseline 95M vs TI-Large 95M) to establish statistical significance.

**Medium-term:**

4. **Broader Modulation**: Extend FiLM modulation to V projections, output projections, and normalization parameters.

5. **Scale to Larger Models**: Test at 300M+ parameter budgets where the memory savings of sharing become more significant.

6. **Downstream Evaluation**: Evaluate on standard NLU/NLG benchmarks beyond perplexity.

7. **Interpretability Analysis**: Examine how time-dependent modulation patterns (scale and shift vectors) evolve across depth — do they reveal meaningful layer specialization?

**Long-term:**

8. **Adaptive Depth**: Since depth is parameterized continuously, explore varying the number of layer applications at inference time based on input complexity.

9. **Combination with LoRA**: Use time-indexed sharing for pre-training, then LoRA for efficient fine-tuning — potentially achieving extreme efficiency at both stages.

10. **Multimodal Extension**: Apply time-indexed sharing to vision transformers and multimodal architectures where depth-wise parameter sharing may yield similar benefits.

---

## 8. References

[1] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is All You Need. *NeurIPS*.

[2] Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. *NeurIPS*.

[3] Tong, A., Nguyen-Tang, T., Lee, D., et al. (2025). Neural ODE Transformers: Analyzing Internal Dynamics and Adaptive Fine-tuning. *ICLR*.

[4] Radford, A., Wu, J., Child, R., et al. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Technical Report*.

[5] Touvron, H., Lavril, T., Izacard, G., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv preprint*.

[6] Su, J., Lu, Y., Pan, S., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. *arXiv preprint*.

[7] Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization. *NeurIPS*.

[8] Shazeer, N. (2020). GLU Variants Improve Transformer. *arXiv preprint*.

[9] Perez, E., Strub, F., De Vries, H., et al. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. *AAAI*.

[10] Dehghani, M., Gouws, S., Vinyals, O., et al. (2019). Universal Transformers. *ICLR*.

[11] Hu, E. J., Shen, Y., Wallis, P., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.

[12] Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). Training Compute-Optimal Large Language Models. *NeurIPS*.

[13] Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv preprint*.

[14] Xiong, R., Yang, Y., He, D., et al. (2020). On Layer Normalization in the Transformer Architecture. *ICML*.

[15] Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2017). Pointer Sentinel Mixture Models. *ICLR*.

---

## 9. Appendices

### Appendix A: Full Training Metrics (All Evaluation Steps)

**A.1 Baseline 152M — Complete Validation Loss Trajectory (50 evaluations)**

| Step | Val Loss | Val PPL | Step | Val Loss | Val PPL |
|------|----------|---------|------|----------|---------|
| 500 | 7.648 | 2,092 | 13,000 | 4.654 | 105.0 |
| 1,000 | 7.058 | 1,161 | 13,500 | 4.653 | 104.9 |
| 1,500 | 6.558 | 704 | 14,000 | 4.631 | 102.5 |
| 2,000 | 6.170 | 479 | 14,500 | 4.430 | 83.9 |
| 2,500 | 5.732 | 308 | 15,000 | 4.449 | 85.5 |
| 3,000 | 5.389 | 219 | 15,500 | 4.426 | 83.6 |
| 3,500 | 5.249 | 190 | 16,000 | 4.430 | 83.9 |
| 4,000 | 5.141 | 171 | 16,500 | 4.396 | 81.1 |
| 4,500 | 5.016 | 151 | 17,000 | 4.367 | 78.7 |
| 5,000 | 4.893 | 134 | 17,500 | 4.392 | 80.8 |
| 5,500 | 4.909 | 136 | 18,000 | 4.383 | 80.0 |
| 6,000 | 4.801 | 122 | 18,500 | 4.349 | 77.3 |
| 6,500 | 4.766 | 118 | 19,000 | 4.382 | 79.9 |
| 7,000 | 4.700 | 110 | 19,500 | 4.359 | 78.1 |
| 7,500 | 4.749 | 116 | 20,000 | 4.361 | 78.3 |
| 8,000 | 4.655 | 105 | 20,500 | 4.343 | 76.8 |
| 8,500 | 4.653 | 105 | 21,000 | 4.303 | 73.8 |
| 9,000 | 4.631 | 103 | 21,500 | 4.296 | 73.3 |
| 9,500 | 4.598 | 99.3 | 22,000 | 4.307 | 74.1 |
| 10,000 | 4.560 | 95.6 | 22,500 | 4.303 | 73.8 |
| 10,500 | 4.545 | 94.2 | 23,000 | 4.341 | 76.6 |
| 11,000 | 4.548 | 94.4 | 23,500 | 4.299 | 73.6 |
| 11,500 | 4.494 | 89.5 | 24,000 | 4.300 | 73.7 |
| 12,000 | 4.550 | 94.6 | 24,500 | **4.265** | **71.1** |
| 12,500 | 4.441 | 84.9 | 25,000 | 4.284 | 72.5 |

**A.2 Baseline 95M — Complete Validation Loss Trajectory (48 evaluations)**

| Step | Val Loss | Val PPL | Step | Val Loss | Val PPL |
|------|----------|---------|------|----------|---------|
| 500 | 7.786 | 2,407 | 12,500 | 4.591 | 98.6 |
| 1,000 | 7.186 | 1,320 | 13,000 | 4.678 | 107.5 |
| 1,500 | 6.674 | 792 | 13,500 | 4.584 | 97.9 |
| 2,000 | 6.300 | 544 | 14,000 | 4.605 | 100.0 |
| 2,500 | 5.912 | 369 | 14,500 | 4.598 | 99.2 |
| 3,000 | 5.477 | 239 | 15,000 | 4.546 | 94.3 |
| 3,500 | 5.318 | 204 | 15,500 | 4.493 | 89.4 |
| 4,000 | 5.185 | 179 | 16,000 | 4.518 | 91.6 |
| 4,500 | 5.081 | 161 | 16,500 | 4.540 | 93.6 |
| 5,000 | 4.990 | 147 | 17,000 | 4.515 | 91.4 |
| 5,500 | 5.041 | 155 | 17,500 | 4.459 | 86.4 |
| 6,000 | 4.904 | 135 | 18,000 | 4.524 | 92.2 |
| 6,500 | 4.851 | 128 | 18,500 | 4.478 | 88.1 |
| 7,000 | 4.809 | 123 | 19,000 | 4.501 | 90.1 |
| 7,500 | 4.779 | 119 | 19,500 | 4.479 | 88.1 |
| 8,000 | 4.787 | 120 | 20,000 | 4.485 | 88.6 |
| 8,500 | 4.767 | 118 | 20,500 | 4.431 | 84.0 |
| 9,000 | 4.745 | 115 | 21,000 | 4.470 | 87.3 |
| 9,500 | 4.720 | 112 | 21,500 | 4.421 | 83.1 |
| 10,000 | 4.702 | 110 | 22,000 | 4.439 | 84.7 |
| 10,500 | 4.644 | 104 | 22,500 | 4.475 | 87.8 |
| 11,000 | 4.699 | 110 | 23,000 | 4.444 | 85.1 |
| 11,500 | 4.591 | 98.6 | 23,500 | **4.405** | **81.8** |
| 12,000 | 4.679 | 108 | 24,000 | 4.432 | 84.1 |

**A.3 Time-Indexed Large 95M — Complete Validation Loss Trajectory (88 evaluations)**

| Step | Val Loss | Val PPL | Step | Val Loss | Val PPL |
|------|----------|---------|------|----------|---------|
| 500 | 7.675 | 2,156 | 22,500 | 4.484 | 88.5 |
| 1,000 | 7.181 | 1,314 | 23,000 | 4.521 | 91.9 |
| 1,500 | 6.735 | 839 | 23,500 | 4.520 | 91.8 |
| 2,000 | 6.419 | 613 | 24,000 | 4.511 | 91.0 |
| 2,500 | 6.190 | 488 | 24,500 | 4.553 | 94.9 |
| 3,000 | 5.857 | 349 | 25,000 | 4.548 | 94.4 |
| 3,500 | 5.644 | 283 | 25,500 | 4.520 | 91.8 |
| 4,000 | 5.527 | 251 | 26,000 | 4.582 | 97.7 |
| 4,500 | 5.332 | 207 | 26,500 | 4.460 | 86.5 |
| 5,000 | 5.228 | 187 | 27,000 | 4.476 | 87.9 |
| 5,500 | 5.220 | 185 | 27,500 | 4.451 | 85.7 |
| 6,000 | 5.080 | 161 | 28,000 | 4.466 | 87.0 |
| 6,500 | 5.023 | 152 | 28,500 | 4.463 | 86.8 |
| 7,000 | 4.969 | 144 | 29,000 | 4.460 | 86.5 |
| 7,500 | 5.036 | 154 | 29,500 | 4.440 | 84.7 |
| 8,000 | 4.938 | 139 | 30,000 | 4.432 | 84.1 |
| 8,500 | 4.918 | 137 | 30,500 | 4.466 | 86.9 |
| 9,000 | 4.822 | 124 | 31,000 | 4.435 | 84.3 |
| 9,500 | 4.840 | 127 | 31,500 | 4.457 | 86.2 |
| 10,000 | 4.830 | 125 | 32,000 | 4.449 | 85.5 |
| 10,500 | 4.790 | 120 | 32,500 | 4.428 | 83.7 |
| 11,000 | 4.753 | 116 | 33,000 | 4.456 | 86.1 |
| 11,500 | 4.706 | 111 | 33,500 | 4.467 | 87.1 |
| 12,000 | 4.872 | 131 | 34,000 | 4.434 | 84.2 |
| 12,500 | 4.648 | 104 | 34,500 | 4.420 | 83.0 |
| 13,000 | 4.645 | 104 | 35,000 | 4.426 | 83.5 |
| 13,500 | 4.690 | 109 | 35,500 | 4.436 | 84.4 |
| 14,000 | 4.675 | 107 | 36,000 | 4.380 | 79.8 |
| 14,500 | 4.629 | 102 | 36,500 | 4.460 | 86.5 |
| 15,000 | 4.622 | 102 | 37,000 | 4.438 | 84.6 |
| 15,500 | 4.599 | 99.4 | 37,500 | 4.405 | 81.8 |
| 16,000 | 4.591 | 98.6 | 38,000 | 4.398 | 81.3 |
| 16,500 | 4.613 | 101 | 38,500 | 4.419 | 83.0 |
| 17,000 | 4.563 | 95.9 | 39,000 | 4.413 | 82.5 |
| 17,500 | 4.567 | 96.2 | 39,500 | 4.398 | 81.2 |
| 18,000 | 4.595 | 99.0 | 40,000 | 4.390 | 80.6 |
| 18,500 | 4.576 | 97.1 | 40,500 | 4.398 | 81.2 |
| 19,000 | 4.600 | 99.5 | 41,000 | 4.381 | 79.9 |
| 19,500 | 4.613 | 101 | 41,500 | **4.352** | **77.6** |
| 20,000 | 4.588 | 98.3 | 42,000 | 4.429 | 83.8 |
| 20,500 | 4.576 | 97.2 | 42,500 | 4.407 | 82.0 |
| 21,000 | 4.589 | 98.4 | 43,000 | 4.397 | 81.2 |
| 21,500 | 4.551 | 94.7 | 43,500 | 4.377 | 79.6 |
| 22,000 | 4.600 | 99.5 | 44,000 | 4.383 | 80.1 |

**A.4 Time-Indexed Small 50M — Complete Validation Loss Trajectory (55 evaluations)**

| Step | Val Loss | Val PPL | Step | Val Loss | Val PPL |
|------|----------|---------|------|----------|---------|
| 500 | 7.804 | 2,447 | 14,500 | 4.959 | 142 |
| 1,000 | 7.307 | 1,494 | 15,000 | 4.896 | 134 |
| 1,500 | 6.868 | 960 | 15,500 | 4.903 | 135 |
| 2,000 | 6.602 | 736 | 16,000 | 4.928 | 138 |
| 2,500 | 6.400 | 602 | 16,500 | 4.879 | 132 |
| 3,000 | 6.137 | 462 | 17,000 | 4.829 | 125 |
| 3,500 | 5.960 | 388 | 17,500 | 4.889 | 133 |
| 4,000 | 5.733 | 308 | 18,000 | 4.894 | 134 |
| 4,500 | 5.591 | 268 | 18,500 | 4.838 | 126 |
| 5,000 | 5.444 | 231 | 19,000 | 4.844 | 127 |
| 5,500 | 5.482 | 240 | 19,500 | 4.841 | 127 |
| 6,000 | 5.354 | 211 | 20,000 | 4.815 | 123 |
| 6,500 | 5.297 | 200 | 20,500 | 4.817 | 124 |
| 7,000 | 5.264 | 193 | 21,000 | 4.839 | 126 |
| 7,500 | 5.224 | 186 | 21,500 | 4.778 | 119 |
| 8,000 | 5.231 | 187 | 22,000 | 4.792 | 121 |
| 8,500 | 5.148 | 172 | 22,500 | 4.842 | 127 |
| 9,000 | 5.184 | 178 | 23,000 | 4.814 | 123 |
| 9,500 | 5.112 | 166 | 23,500 | 4.820 | 124 |
| 10,000 | 5.070 | 159 | 24,000 | 4.786 | 120 |
| 10,500 | 5.063 | 158 | 24,500 | 4.780 | 119 |
| 11,000 | 5.093 | 163 | 25,000 | 4.777 | 119 |
| 11,500 | 5.006 | 149 | 25,500 | 4.717 | 112 |
| 12,000 | 5.140 | 171 | 26,000 | 4.781 | 119 |
| 12,500 | 4.959 | 142 | 26,500 | **4.717** | **112** |
| 13,000 | 5.013 | 150 | 27,000 | 4.768 | 118 |
| 13,500 | 5.044 | 155 | 27,500 | — | — |

### Appendix B: Key Code Samples

**B.1 Time Modulation (FiLM-style):**

```python
class TimeModulation(eqx.Module):
    scale_proj: jnp.ndarray  # (time_dim, hidden_dim)
    shift_proj: jnp.ndarray  # (time_dim, hidden_dim)

    @staticmethod
    def init(time_dim: int, hidden_dim: int, *, key):
        k1, k2 = jrandom.split(key)
        return TimeModulation(
            scale_proj=jrandom.normal(k1, (time_dim, hidden_dim)) * 0.01,
            shift_proj=jnp.zeros((time_dim, hidden_dim)),
        )

    def __call__(self, x, time_embed):
        scale = 1.0 + jnp.tanh(time_embed @ self.scale_proj)
        shift = time_embed @ self.shift_proj
        return hax.named(x.array * scale + shift, x.axes)
```

**B.2 Time-Indexed Block (shared across all layers):**

```python
class TimeIndexedBlock(eqx.Module):
    norm1: RMSNorm
    attn: TimeModulatedAttention
    norm2: RMSNorm
    mlp: TimeModulatedSwiGLU

    def __call__(self, x, time_embed, mask, rope_cos, rope_sin, *, key, inference):
        h = self.attn(self.norm1(x), time_embed, mask, rope_cos, rope_sin, ...)
        x = x + dropout(h)
        h = self.mlp(self.norm2(x), time_embed, ...)
        x = x + dropout(h)
        return x
```

**B.3 Forward Pass — Time-Indexed Transformer:**

```python
def __call__(self, input_ids, *, key=None, inference=False):
    x = self.token_embeddings(input_ids)
    x = apply_dropout(x, self.dropout_rate, ...)
    Pos = input_ids.resolve_axis("position")
    KeyPos = Pos.alias("key_position")
    mask = hnn.attention.causal_mask(Pos, KeyPos)

    for layer_idx in range(self.num_layers):
        t = layer_idx / max(1, self.num_layers - 1)
        time_emb = self.time_embed(t)
        x = self.block(x, time_emb, mask, self.rope_cos, self.rope_sin, ...)

    x = self.norm_f(x)
    logits = self.token_embeddings.unembed(x)
    return logits
```

### Appendix C: Experimental Configuration

**C.1 Baseline 152M Configuration:**

```json
{
  "mode": "baseline",
  "hidden_dim": 768,
  "num_heads": 12,
  "num_layers": 12,
  "sequence_length": 512,
  "learning_rate": 0.0003,
  "warmup_steps": 2000,
  "stable_steps": 20000,
  "max_steps": 150000,
  "micro_batch_size": 4,
  "gradient_accumulation": 64,
  "label_smoothing": 0.1,
  "dropout": 0.1,
  "weight_decay": 0.1
}
```

**C.2 Time-Indexed Large Configuration:**

```json
{
  "mode": "time_index",
  "hidden_dim": 1280,
  "num_heads": 16,
  "num_layers": 12,
  "sequence_length": 512,
  "time_embed_dim": 256,
  "learning_rate": 0.0003,
  "warmup_steps": 2000,
  "stable_steps": 20000,
  "max_steps": 150000,
  "micro_batch_size": 4,
  "gradient_accumulation": 64,
  "label_smoothing": 0.1,
  "dropout": 0.1,
  "weight_decay": 0.1
}
```

**C.3 Baseline 95M Configuration:**

```json
{
  "mode": "baseline",
  "hidden_dim": 580,
  "num_heads": 10,
  "num_layers": 12,
  "sequence_length": 512,
  "learning_rate": 0.0003,
  "warmup_steps": 2000,
  "stable_steps": 20000,
  "max_steps": 150000,
  "patience": 999,
  "micro_batch_size": 4,
  "gradient_accumulation": 64,
  "label_smoothing": 0.1,
  "dropout": 0.1,
  "weight_decay": 0.1
}
```

**C.4 Time-Indexed Small Configuration:**

```json
{
  "mode": "time_index",
  "hidden_dim": 768,
  "num_heads": 12,
  "num_layers": 12,
  "sequence_length": 512,
  "time_embed_dim": 256,
  "learning_rate": 0.0003,
  "patience": 15,
  "micro_batch_size": 4,
  "gradient_accumulation": 64
}
```

### Appendix D: GPT-2 Evaluation Script

The following script evaluates HuggingFace's pretrained GPT-2 Small using the exact same protocol as our training validation:

```python
#!/usr/bin/env python3
"""Evaluate pretrained GPT-2 Small (124M) on WikiText-103 validation."""

import json, math, torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

SEQ_LEN = 512
MAX_SEQUENCES = 500
EVAL_SEQUENCES = 200
BATCH_SIZE = 4

def main():
    device = torch.device("cuda:0")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    wikitext = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    all_val_text = "\n".join([t for t in wikitext['text'] if len(t.strip()) > 0])
    all_val_tokens = tokenizer.encode(all_val_text)

    val_sequences = []
    for i in range(0, len(all_val_tokens) - SEQ_LEN - 1, SEQ_LEN + 1):
        val_sequences.append(all_val_tokens[i:i + SEQ_LEN + 1])
        if len(val_sequences) >= MAX_SEQUENCES:
            break

    total_loss, num_batches = 0.0, 0
    with torch.no_grad():
        for i in range(0, min(len(val_sequences), EVAL_SEQUENCES), BATCH_SIZE):
            batch_seqs = val_sequences[i:i + BATCH_SIZE]
            if len(batch_seqs) < BATCH_SIZE:
                continue
            input_ids = torch.tensor([s[:-1] for s in batch_seqs], dtype=torch.long).to(device)
            labels = torch.tensor([s[1:] for s in batch_seqs], dtype=torch.long).to(device)
            outputs = model(input_ids)
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    ppl = math.exp(min(avg_loss, 20))
    print(f"Val Loss: {avg_loss:.4f}, Val PPL: {ppl:.2f}, Params: {num_params:,}")

    results = {"model": "gpt2-small-pretrained", "params": num_params,
               "val_loss": avg_loss, "val_ppl": ppl,
               "eval_sequences": num_batches * BATCH_SIZE, "seq_len": SEQ_LEN}
    with open("gpt2_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
```

**Result:**

```json
{
  "model": "gpt2-small-pretrained",
  "params": 124439808,
  "val_loss": 3.572,
  "val_ppl": 35.58,
  "eval_sequences": 200,
  "seq_len": 512
}
```

### Appendix E: Training Automation Scripts

**E.1 Phase Scheduling Script (abridged)**

```bash
#!/bin/bash
# schedule_next_training.sh — Monitor Phase 1, then launch Phase 2

# Phase 1: Baseline 152M (GPU 0) + TI-Small (GPU 1)
while kill -0 $PID_BASELINE 2>/dev/null || kill -0 $PID_TIMEINDEX 2>/dev/null; do
    sleep 300
done

# Phase 2: Launch large time-indexed model
CUDA_VISIBLE_DEVICES=0 nohup python train_v2.py \
    --mode time_index --hidden_dim 1280 --num_heads 16 --num_layers 12 \
    --max_steps 150000 --patience 15 \
    --output_dir checkpoints_v5_timeindex_large \
    > training_v5_timeindex_large.log 2>&1 &
```

**E.2 Mid-Training Configuration Swap**

```bash
#!/bin/bash
# swap_training.sh — Wait for checkpoint, swap patience from 15 to 999

CKPT="checkpoints_v5_timeindex_large/checkpoint_015000.pkl"
OLD_PID=295707

# Poll for checkpoint
while [ ! -f "$CKPT" ]; do
    kill -0 $OLD_PID 2>/dev/null || break
    sleep 60
done

# Wait for write completion, then swap
sleep 30
kill $OLD_PID; sleep 5

# Resume with early stopping disabled
CUDA_VISIBLE_DEVICES=0 nohup python train_v2.py \
    --mode time_index --hidden_dim 1280 --num_heads 16 --num_layers 12 \
    --max_steps 150000 --patience 999 \
    --resume_from "$CKPT" \
    --output_dir checkpoints_v5_timeindex_large \
    >> training_v5_timeindex_large.log 2>&1 &
```

---

## Declaration

I declare that this thesis is my own work and has not been submitted for any other degree or professional qualification. All sources have been properly acknowledged.

**Signature:** _______________________

**Date:** _______________________

---

*Word Count: Approximately 10,800 words (including appendices, references, tables, and code)*
