---
layout: post
title: DeepSeekV4技术报告阅读笔记
date: 2026-08-12 21:30:00 +0800
description: DeepSeekV4技术报告阅读笔记
tags: LLM Architecture
categories: notes
featured: true
---

## **1. 总览**

DeepSeek-V4 共有 2 个大小的模型，均为 MoE 语言模型。分别为 DeepSeek-V4-Flash 和 DeepSeek-V4-Pro。两模型的具体参数如下表：

|             | DeepSeek-V4-Flash | DeepSeek-V4-Pro | 
| :-----------: | :-----------: | :-----------: |
| 总参数量   | 284B       | 1.6T       |
| 激活参数量   |    13B     | 49B        |
| 上下文长度   |    1M     | 1M        |


DeepSeek-V4 在**模型架构**方面的主要技术路线为：
- 混合 Attention 架构：包含 **Compressed Sparse Attention (CSA)** 和 **Heavily Compressed Attention (HCA)**，这两种架构帮助提升了在长上下文情形下的推理效率。
- **Manifold Constrained Hyper-Connections (mHC)**：通过扩展表征的维度尝试提升模型的表达能力。
- **Muon 优化器**：能更快地收敛并提高训练稳定性。

预训练阶段：（还没细看）


后训练阶段：（还没细看）



相比于 DeepSeek-V3.2，V4 在长上下文场景下的效率显著提升。在 1M token 的长上下文 setting 下，与 V3.2 相比，V4 的 single-token inference FLOPs 只有 V3.2 的 27%，KV cache 只有 V3.2 的 10%。论文开源代码网址为 <https://huggingface.co/collections/deepseek-ai/deepseek-v4>。


## **2. 模型架构部分**

### 2.1 与 DeepSeek-V3 的异同
- 使用 MoE 架构，采用 DeepSeekMoE 范式
- MoE 负载均衡部分，仍然采用 V3 的思路：auxiliary-loss-free strategy 加上一个针对极端情形的惩罚项 loss。但是 V4 相比 V3 移去了对 routing target nodes 的限制，重新设计了并行策略来保持训练效率。
- V3 前几层使用 dense Transformer，V4 将其中的 dense FFN layer 换成带有 Hash routing 的 MoE layer。
- V4 使用与 V3 相同的 **multi-token prediction** 策略。

#### 2.1.1 Multi-token Prediction
原本是 DeepSeek-V3 技术报告中的内容，有空我在这里介绍一下。

### 2.2 mHC
mHC 是 Hyper-Connections (HC) 的改进版本，介绍 mHC 之前，先介绍一下 HC。

HC 希望通过扩展残差数据流的宽度来增加网络的拓扑复杂度，认为这样可以提升模型的表达能力，从而提升性能。比如我们在第 $l$ 层原本有一个残差函数 $\mathcal{F}_l: \mathbb{R}^d \to \mathbb{R}^d$，假设我们第 $l$ 层的输入为 $x_l \in \mathbb{R}^d$，那么我们一般的残差网络的做法是：

$$
x_{l+1} = x_l + \mathcal{F}_l(x_l).
$$

HC 的做法则是尝试将 $\mathbb{R}^d$ 中的特征扩展到 $\mathbb{R}^{n_\mathrm{hc} \times d}$，从而我们第 $l$ 层的特征变成 $X_l = [\mathbf{x}_ {l,1}, \mathbf{x}_ {l,2}, \dots, \mathbf{x}_ {l,n_ \mathrm{hc}}]^T \in \mathbb{R}^{n_ \mathrm{hc} \times d}$。由于 $\mathcal{F}_ l$ 是从 $\mathbb{R}^d$ 到 $\mathbb{R}^d$ 的映射，数据在进入 $\mathcal{F}_ l$ 前需被处理为 $\mathbb{R}^d$ 空间中的数据，在从 $\mathcal{F}_ l$ 中出来后需被处理为 $\mathbb{R}^{n_ \mathrm{hc} \times d}$ 维的数据，于是 HC 中引入 input mapping $A_ l \in \mathbb{R}^{1 \times n_ \mathrm{hc}}$ 与 output mapping $C_ l \in \mathbb{R}^{n_ \mathrm{hc} \times 1}$ 分别在特征进入残差函数之前和之后进行处理。此外，HC还引入了一个 residual transformation $B_ l \in \mathbb{R}^{n_ \mathrm{hc} \times n_ \mathrm{hc}}$ 来对 indentity map 的输出进行变换。HC 的公式为：

$$
X_ {l+1}
=
B_ l X_ l
+
C_ l \mathcal{F}_ l\left(A_ l X_ l\right).
\tag{1}
$$

其中，$A_ l X_ l = (A_ {l,1} , A_ {l,2}, \dots, A_ {l, n_ \mathrm{hc}}) \cdot (\mathbf{x}_ {l,1}, \mathbf{x}_ {l,2}, \dots, \mathbf{x}_ {l,n_ \mathrm{hc}})^T = \sum_{i=1}^{n_ \mathrm{hc}} A_ {l,i} \mathbf{x}_ {l,i}$ 可以理解为使用 $A_l $ 中的各个元素作为权重，对 $X_ l$ 各行进行加权求和；记 $\mathcal{F}_ l = \mathcal{F}_ l\left(A_ l X_ l\right) \in \mathbb{R}^{1 \times d}$，则 $C_ l \mathcal{F}_ l$ 为 $n_ \mathrm{hc} \times d$ 的矩阵，其中第 $i$ 行为 $C_ {l, i} \mathcal{F}_ l$。

原始的残差连接可以保证 identify 信号可以传递到最后一层，即：

$$
x_ L = x_ {L-1} + \mathcal{F}_ {L-1}(x_ {L-1}) = \cdots = x_ l + \sum_ {i=l}^{L-1} \mathcal{F}_ i(x_ i),
$$

其中 $L > l$。可以看到，即使传输到第 $L$ 层，第 $l$ 层的信号 $x_l$ 仍然被保持。而 identity mapping 给出的信号 $x_l$ 并没有随着层数的加深而消失，这一点看起来正好符合 resnet 的设计初衷。然而，对于 HC，通过递推可以得到：

$$
X_ L = \left(\prod_{i=1}^{L-l} B_ {L-i}\right)X_ l + \sum_{i=l}^{L-1}\left(\prod_{j=1}^{L-1-i} B_ {L-j}\right)C_i  \mathcal{F}_ i(A_ i X_ i).
$$

可以看到，第一项为 $\left(\prod_{i=1}^{L-l} B_ {L-i}\right)X_ l$ 而不是 $X_ l$，已经不再是 identity mapping 了，这一点实际上已经有些背离了 residual network 的设计初衷了。

未完待续。。。。