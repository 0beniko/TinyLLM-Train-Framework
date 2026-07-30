# 知辰大模型训练框架 ✨

> 从零构建 0.1B 参数级中文语言模型训练框架，覆盖 Tokenizer 构建、Decoder-only Transformer 建模、分布式预训练、监督微调、Benchmark 评测与对话验证全流程。

知辰是一个面向中文语言建模与指令跟随能力训练的轻量级大模型项目。项目不依赖现成大模型结构封装，而是从 Tokenizer、模型结构、训练循环、学习率调度、loss 计算、SFT 掩码、分布式训练到评测流程进行完整手写实现，形成一条可复现、可解释、可扩展的中文大模型训练链路。

“知辰”之名取意于：**洞悉星辰轨迹，通晓时序轮回。**“辰”指代天地时序，象征模型不仅学习语言表层规律，也尝试在连续上下文中把握因果、节奏与趋势，具备面向复杂任务的远见格局，预判万物走向。

## 项目概览 🚀

知辰的目标是从零训练一个约 0.1B 参数规模的中文语言模型，使其具备基础中文语言建模、一次性对话和指令跟随能力。当前仓库负责预训练与普通 SFT；后续 CoT-SFT 和 GRPO 位于 [TinyAudit-GRPO](https://github.com/0beniko/TinyAudit-GRPO)。完整链路如下：

```text
原始语料
  -> Byte-level BPE Tokenizer 训练
  -> 数据清洗与 token 化
  -> 预训练二进制样本构建
  -> Decoder-only Transformer 预训练
  -> 中文 Benchmark 评测
  -> 单轮 SFT 数据筛选
  -> SFT 指令微调
  -> 单轮对话验证
  -> TinyAudit-GRPO：CoT-SFT 与 GRPO
```

![训练流程](assets/README/training-pipeline.png)

核心特性：

- 🧩 从零训练 15K 词表 BBPE（Byte-level BPE）Tokenizer，统一支撑预训练、SFT 和推理。
- 🧠 手写类 Qwen3 Dense 的 Decoder-only Transformer，默认配置约 82M 参数，属于 0.1B 参数级中文模型。
- ⚙️ 实现 RMSNorm、Grouped Query Attention、RoPE、SwiGLU、KV Cache、Embedding/LM Head 权重绑定等现代 LLM 关键结构。
- 📉 手写 token 级自回归交叉熵 loss、梯度累积、混合精度、Warmup + Cosine 学习率调度与 checkpoint 保存。
- 🖥️ 支持多卡 DDP 预训练和 SFT，SwanLab 记录 loss、学习率和训练过程。
- 🎯 SFT 阶段仅对 assistant 回答区域计算 loss，指令和用户输入部分通过 `label=-100` 屏蔽。
- 📊 使用中文因果推断/逻辑推理 Benchmark 进行能力验证，包括 XCOPA 与 CLUE C3。

## 整体架构 🏗️

![模型结构](assets/README/model-architecture.png)

知辰采用标准 Decoder-only Transformer 架构。每个 Transformer Block 使用 Pre-Norm 结构：

```text
x = x + SelfAttention(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

默认模型配置如下：

| 配置项 | 默认值 | 说明 |
| --- | ---: | --- |
| `vocab_size` | 15000 | 15K Byte-level BPE 词表 |
| `hidden_size` | 768 | 隐藏层维度，需能被 Query 头数整除 |
| `num_hidden_layers` | 12 | Transformer Block 层数 |
| `num_attention_heads` | 12 | Query 头数 |
| `num_key_value_heads` | 4 | Key/Value 头数，GQA 分组比为 3:1 |
| `intermediate_size` | 2048 | SwiGLU 前馈网络中间维度 |
| `max_seq_len` | 512 | 训练默认序列长度 |
| `max_position_embeddings` | 32768 | RoPE 位置编码最大配置能力 |
| `rope_theta` | 10000.0 | RoPE 基础频率 |
| `dropout` | 0.0 | 默认不使用 dropout |
| 参数规模 | 约 82M | 默认配置下的近似参数量 |

### RMSNorm：更稳定的 Pre-Norm 归一化

项目使用 RMSNorm 替代 LayerNorm。RMSNorm 只按均方根进行缩放，不做均值中心化，计算时先转为 `float32`，再转回原始精度，以降低混合精度训练中的数值误差。

与 Post-Norm 相比，Pre-Norm 将归一化放在 Attention 和 FFN 之前，残差路径更直接，深层训练更稳定：

```python
hidden_states = hidden_states + Attention(RMSNorm(hidden_states))
hidden_states = hidden_states + MLP(RMSNorm(hidden_states))
```

### GQA 分组查询注意力：降低 KV 参数与推理缓存

知辰默认使用 12 个 Query 头、4 个 Key/Value 头。Key/Value 头数只有 Query 的三分之一，前向传播时通过 `repeat_kv` 动态扩展 KV，使其匹配 Query 头数。

这种 Grouped Query Attention 设计保留多头 Query 的表达能力，同时减少 K/V 投影参数量，并显著降低推理阶段 KV Cache 的显存占用。默认配置下分组比为：

```text
num_attention_heads / num_key_value_heads = 12 / 4 = 3
```

### RoPE 旋转位置编码：无参数注入相对位置信息

项目手写 RoPE 旋转位置编码，将每个注意力头的 hidden 维度按两两一组视为复平面上的旋转分量。模型预先计算不同位置的 `cos/sin` 频率表，前向时对 Q/K 施加旋转变换，使注意力分数天然感知相对位置差。

RoPE 不引入额外可训练参数，适合 Decoder-only 自回归语言模型，也便于在推理时结合 KV Cache 增量生成。

### SwiGLU 前馈层：门控信息流

知辰使用 SwiGLU 替代传统 ReLU FFN。前馈层包含两条并行线性分支：

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

其中 `gate_proj` 控制每个维度的信息流，`up_proj` 提供候选特征，二者逐元素相乘后再投影回隐藏维度。相比普通 ReLU FFN，SwiGLU 的梯度更平滑，表达能力更强，是现代 LLM 中常用的前馈结构。

### KV Cache 与权重绑定

推理时，模型支持 `past_key_values`。新 token 只需要和历史 KV 拼接，不必重复计算完整前缀，从而降低自回归生成的重复计算成本。

同时，项目将输入 Embedding 与输出 LM Head 权重绑定：

```python
embed_tokens.weight = lm_head.weight
```

在 15K 词表、768 hidden size 下，权重绑定可减少约 `15000 x 768 = 11.5M` 参数，使小参数模型更紧凑。

## Tokenizer 构建 🧩

Tokenizer 是知辰训练链路的第一步。项目从中文和英文语料中训练 BBPE（Byte-level BPE）Tokenizer，词表大小为 15K，其中包含 15 个特殊 token。

特殊 token 设计如下：

| Token | 用途 |
| --- | --- |
| `<|endoftext|>` | 文本结束、padding/unknown 兜底 |
| `<|im_start|>` / `<|im_end|>` | 对话边界控制 |
| `<|system|>` / `<|user|>` / `<|assistant|>` | 对话角色标识 |
| `<think>` / `</think>` | CoT 推理片段标识 |
| `<tool_call>` / `</tool_call>` | 工具调用边界 |
| `<function>` / `</function>` | 函数调用边界 |
| `<pad>` / `<unk>` / `<unused_0>` | 填充、未知和保留位 |

训练入口：

```bash
python train/train_tokenizer.py
```

输出目录：

```text
tokenizer_15k/
  tokenizer.json
  tokenizer_config.json
  vocab.json
  merges.txt
```

该 Tokenizer 在预训练、SFT 和推理阶段保持统一，避免不同阶段编码空间不一致带来的分布偏移。

## 预训练流程 🔥

预训练阶段使用约 1.5B tokens 多源开源语料。数据预处理后保存为 `.bin + .meta` 格式，训练时通过 `numpy.memmap` 读取，避免一次性加载大规模 token 数据到内存。

![预训练数据分布](assets/README/pretrain-data-distribution.png)

预训练数据集返回 `(input_ids, labels)`，二者初始相同。模型内部执行 shift：

```text
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
loss = CrossEntropy(shift_logits, shift_labels)
```

也就是使用当前位置之前的上下文预测下一个 token，符合标准自回归语言模型训练目标。

单机多卡预训练示例：

```bash
torchrun --nproc_per_node=4 train/pretrain.py \
  --data_path /path/to/pretrain_data \
  --save_dir ./pretrain_out \
  --hidden_size 768 \
  --num_hidden_layers 12 \
  --batch_size 128 \
  --accumulation_steps 4 \
  --learning_rate 1e-3 \
  --dtype bfloat16 \
  --use_compile 1 \
  --max_seq_len 512 \
  --use_swanlab 1
```

训练代码包含：

- DDP 分布式初始化和 `DistributedSampler` 数据切分。
- `torch.amp.autocast` 引入 bfloat16 混合精度训练。
- `torch.compile` 对模型前向进行动静态图编译优化，减少 Python 调度开销。
- 梯度累积、梯度裁剪与 AdamW 参数更新，突破单卡显存对 batch size 的限制。
- AdamW 优化器与 Warmup + Cosine Decay 学习率调度。
- 主进程 checkpoint 保存，支持后续 resume。
- 周期性运行中文 Benchmark，并将结果记录到 SwanLab。

### 训练工程优化：AMP、torch.compile 与梯度累积

在硬件显存有限的情况下，知辰没有简单降低模型结构复杂度，而是从训练工程侧进行优化，尽量在稳定性、吞吐和显存之间取得平衡。

![学习率调度曲线](assets/README/learning-rate-schedule.png)

| 优化项 | 实现方式 | 作用 |
| --- | --- | --- |
| bfloat16 AMP | `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)` | 降低激活显存占用，提高 Tensor Core 计算吞吐，同时保留比 fp16 更大的指数范围 |
| torch.compile | `model = torch.compile(model)` | 对前向图进行编译优化，减少 Python 解释器调度和小算子开销 |
| 梯度累积 | `loss / accumulation_steps` 后分步反传 | 在显存不足以容纳大 batch 时，用多次 micro-batch 近似更大的全局 batch |
| 梯度裁剪 | `clip_grad_norm_` | 降低训练早期或异常 batch 带来的梯度爆炸风险 |
| Warmup + Cosine | `get_lr(current_step, total_steps, lr, warmup_steps)` | 训练初期平滑升温，后期余弦衰减，兼顾收敛速度与稳定性 |

其中，bfloat16 AMP 是本项目训练稳定性的重要选择。相比 fp16，bfloat16 拥有更大的动态范围，在大模型训练中更不容易出现溢出；相比 fp32，则能明显降低显存占用并提升计算效率。代码中通过 `--dtype bfloat16` 控制混合精度上下文，默认在 CUDA 环境下启用 `autocast`。

梯度累积用于解决“模型、序列长度和 batch size 同时增大时显存不足”的问题。每个 micro-batch 先计算 `loss / accumulation_steps` 并反向传播，累计多步梯度后再统一执行一次 optimizer step。这样可以在不改变模型结构的情况下扩大等效 batch size，使训练曲线更平滑。

训练过程中会定期保存 checkpoint，并记录 loss、learning rate、global step 等信息，保证长时间训练出现中断时可以继续恢复。

![训练终端与 checkpoint](assets/README/training-checkpoint-terminal.png)

## SFT 指令微调 🎯

原始 SFT 数据包含 2,023,725 条对话。当前阶段只考虑一次性对话，因此不再把原始数据直接送入训练，而是先筛选为严格的一问一答，再将预训练模型继续训练为具备中文单轮对话与指令跟随能力的模型。

SFT 数据格式为单轮对话：

```json
{
  "conversations": [
    {"role": "user", "content": "请解释什么是注意力机制。"},
    {"role": "assistant", "content": "注意力机制是一种..."}
  ]
}
```

### 单轮数据筛选

原始数据中混有多轮样本、重复问答、异常控制字符和超过训练窗口的长样本。对于监督数据，直接把所有样本截断到 512 tokens 可能截掉 assistant 答案尾部和 EOS，使模型学习到不完整回答。因此当前采用：

- 只保留严格的一条 `user` 和一条 `assistant`；
- 清除首尾空白；
- 丢弃空文本、损坏字符和嵌入的对话控制 token；
- 对 `(user, assistant)` 精确去重；
- 使用项目 Tokenizer 计算完整样本长度；
- 完整长度超过 512 tokens 时整条丢弃，不截断；
- 短样本补齐到 512，padding 不参与 loss；
- user 区域全部 mask，只计算 assistant 内容和 assistant EOS 的 loss。

本次筛选统计记录在 `data/spongebob_sft_single_512.manifest.json`：

| 项目 | 数量 |
| --- | ---: |
| 原始样本 | 2,023,725 |
| 最终保留 | 1,576,371 |
| 非单轮样本 | 418,511 |
| 精确重复 | 25,522 |
| 超过 512 tokens | 3,046 |
| 损坏内容 | 270 |
| 空内容 | 5 |
| 保留比例 | 77.89% |
| 平均长度 | 148.58 tokens |

生成清洗后的数据：

```bash
python dataset/prepare_single_turn_sft.py \
  --input data/spongebob_sft.jsonl \
  --output data/spongebob_sft_single_512.jsonl \
  --tokenizer tokenizer_15k \
  --max-length 512
```

脚本同时生成 manifest，记录筛选规则、数量和输入输出 SHA-256。完整训练数据体积较大，可不提交到 Git；建议提交处理脚本与 manifest，使筛选口径可以复现。

项目将对话编码为：

```text
<|im_start|><|user|>用户问题<|im_end|><|assistant|>模型回答<|im_end|>
```

其中 user 指令部分全部设置为 `label=-100`，只对 assistant 回答部分计算交叉熵：

```text
input_ids: <|user|> question <|im_end|> <|assistant|> answer <|im_end|>
labels:    -100     -100     -100       assistant_tokens...
```

这样模型不会被训练去复读用户问题，而是专注学习在给定指令后的回答分布。

SFT 训练示例：

```bash
torchrun --nproc_per_node=4 train/train_sft.py \
  --data_path ./data/spongebob_sft_single_512.jsonl \
  --tokenizer_path ./tokenizer_15k \
  --from_weight /path/to/pretrain_768.pth \
  --save_dir ./out_sft \
  --hidden_size 768 \
  --num_hidden_layers 12 \
  --batch_size 128 \
  --accumulation_steps 4 \
  --learning_rate 2e-5 \
  --dtype bfloat16 \
  --use_compile 1 \
  --max_seq_len 512 \
  --use_swanlab 1
```

训练参数仍集中在 `train/train_sft.py` 底部的 `argparse` 中。可以直接修改对应 `default=...` 后运行，也可以使用同名命令行参数临时覆盖。README 不写入任何真实 API Key；公开提交前必须清空脚本中的密钥。

### SFT Benchmark：自建指令评测与 LLM-as-Judge

为了更直接观察 SFT 后模型的基础对话能力，项目额外构建了一个轻量级自建 SFT Benchmark。该 Benchmark 不追求覆盖所有复杂知识场景，而是聚焦小模型最基础、最容易暴露问题的交互能力：回答是否自然、是否符合事实、是否真正理解并执行了用户指令。

评测数据通过 LLM 对预定义的 9 种类别进行 prompt 生成，共 100 条样本：

| 类别 | 样本数 | 评估重点 |
| --- | ---: | --- |
| 开放问答 | 30 | 日常对话、常见问题回答 |
| 基础常识 | 29 | 基础事实与生活常识 |
| 简单指令 | 10 | 是否按用户要求执行 |
| 简单逻辑 | 10 | 基础推理与条件判断 |
| 上下文提取 | 5 | 从给定上下文中提取关键信息 |
| 角色扮演 | 5 | 是否保持设定角色与语气 |
| 简单的代码 | 5 | 基础代码理解与生成 |
| 算术与数字 | 5 | 简单数字计算与比较 |
| 结束语 | 1 | 对话结束场景 |

评估方式采用 **LLM-as-Judge**：每个 prompt 生成 3 个候选回答，再使用 DeepSeek V3.2 快思考版本作为 Judge，从三个维度进行二值判分。

| 维度 | 含义 |
| --- | --- |
| `fluency` | 回答是否流畅、语言是否自然 |
| `factuality` | 回答是否准确、是否符合事实 |
| `instruction_following` | 是否正确理解并遵循指令，回答用户真正提出的问题 |

Judge Prompt 如下：

```text
请根据问题对以下回答进行评分（0-1 二值）：

【问题】{question}
【回答】{response}

请从三个维度评分（0=不通过，1=通过）：
1. fluency: 回答是否流畅、语言自然
2. factuality: 回答是否准确、符合事实
3. instruction_following: 是否正确理解并遵循了指令并回答用户问题

请务必严格，如果无法判断，则视为不通过。

以 JSON 格式输出：
{
  "fluency": 0 或 1,
  "factuality": 0 或 1,
  "instruction_following": 0 或 1
}
```

SFT 阶段训练曲线如下，可以看到 loss 在微调早期快速下降，并在后续阶段进入相对稳定区间；learning rate 使用 warmup 后余弦衰减，ETA 曲线反映了长时间训练过程中的耗时变化。

![SFT 训练曲线](assets/README/sft-training-curves.png)

三维 Judge 指标用于分别观察模型回答的语言自然度、事实正确性和指令遵循能力：

![SFT Judge 三维指标](assets/README/sft-judge-rubric-metrics.png)

最终综合指标通过 `mean_avg3` 和 `mean_pass3` 汇总，其中 `avg3` 表示 3 个候选回答的平均通过率，`pass3` 表示 3 个候选回答中至少有 1 个通过的比例，更适合观察小模型在采样生成场景下的可用回答概率。

![SFT Judge 综合指标](assets/README/sft-judge-mean-score.png)

## 评测与实验结果 📊

项目在预训练和 SFT 阶段均设计了评测闭环：

- 预训练阶段：使用中文因果推断/逻辑推理 Benchmark 评估模型基础理解能力。
- SFT 阶段：使用自建 `benchmark/mini_bench` 抽样生成回答，并结合 DeepSeek Judge 进行对话质量评估。
- 训练过程：使用 SwanLab 记录 loss、learning rate、ETA 和评测指标，便于观察收敛趋势。

![训练指标](assets/README/swanlab-training-metrics.png)

核心实验结果：

| 指标 | 结果 |
| --- | ---: |
| 预训练数据规模 | 约 1.5B tokens |
| 原始 SFT 数据 | 2,023,725 条 |
| 筛选后单轮 SFT 数据 | 1,576,371 条 |
| 训练 loss | 约 8 收敛至约 2.3 |
| XCOPA 中文因果推断 Benchmark | 约 0.55 |
| CLUE C3 中文阅读理解/逻辑推理 Benchmark | 约 0.48 |

经过 SFT 后，知辰能够完成基础中文单轮对话，并对常见指令给出较完整的回答。

![SFT 对话效果](assets/README/sft-dialogue-demo.png)

### 交互效果展示 🖥️

除离线 Benchmark 外，项目还提供了本地命令行对话入口，用于验证模型权重、Tokenizer、推理设备和对话模板是否能够完整串联。启动时会依次加载知辰 Tokenizer、初始化 0.1B 模型结构、载入 SFT 权重，并进入单轮对话模式。

<p align="center">
  <img src="assets/README/zhichen-cli-startup.png" alt="知辰大模型启动界面" width="760">
</p>

在交互验证中，模型能够围绕自我介绍、城市美食推荐等开放问题生成连续回答。该截图与上方 SFT Judge 曲线共同构成 SFT 后的效果展示：曲线用于量化评估，终端样例用于直观看到回答形态。

<p align="center">
  <img src="assets/README/sft-evaluation-transcript.png" alt="SFT 评测与对话展示" width="900">
</p>

评测相关入口：

```bash
python eval.py
python benchmark/test_xcopa_improved.py
python benchmark/mini_bench/eval.py
```

## CoT-SFT 与 GRPO 后训练 🧠

完成普通 SFT 后，可将兼容权重交给
[TinyAudit-GRPO](https://github.com/0beniko/TinyAudit-GRPO) 继续进行 CoT-SFT 与
GRPO 后训练。两个仓库共用 Tokenizer 和模型结构，职责划分如下：

```text
TinyLLM-Train-Framework
  └─ 预训练 → 普通单轮 SFT
                  ↓
TinyAudit-GRPO
  └─ CoT 数据清洗 → CoT-SFT → 混合奖励 GRPO → 独立评测
```

<p align="center">
  <img src="assets/README/post-training/model-interface.png" alt="知辰大模型启动界面" width="82%">
</p>

<p align="center"><sub>加载 CoT-SFT 权重后的知辰单轮对话界面。</sub></p>

### CoT-SFT 训练与效果

CoT-SFT 只对 assistant 区域计算 loss，并使用经过格式、长度、重复和身份过滤的单轮
`<think>...</think>` 数据。训练曲线采用通栏展示，两张生成案例采用等宽双栏，便于
在桌面端对比，同时保持移动端可读性。

<p align="center">
  <img src="assets/README/post-training/cot-sft-training-curves.png" alt="CoT-SFT 训练曲线" width="96%">
</p>

<p align="center"><sub>CoT-SFT 的 loss、learning rate 与 ETA；第二轮开始时 ETA 会重新估算。</sub></p>

<table>
  <tr>
    <td width="50%" align="center">
      <img src="assets/README/post-training/cot-sft-dialogue-1.png" alt="CoT-SFT 晚餐建议案例" width="100%">
    </td>
    <td width="50%" align="center">
      <img src="assets/README/post-training/cot-sft-dialogue-2.png" alt="CoT-SFT 北京美食案例" width="100%">
    </td>
  </tr>
  <tr>
    <td align="center"><sub>开放建议与结构化回答</sub></td>
    <td align="center"><sub>常识问答与列表生成</sub></td>
  </tr>
</table>

身份数据用于稳定模型名称、开发者、学校与参数规模等基础事实：

<p align="center">
  <img src="assets/README/post-training/identity-dialogue.png" alt="知辰大模型身份识别案例" width="96%">
</p>

### GRPO 效果

GRPO 使用本地确定性奖励处理可验证任务，并仅对少量开放任务调用组内 Judge。以下截图
用于定性观察推理格式、信息组织和指令遵循；正式权重选择仍应以独立验证集为准。

<table>
  <tr>
    <td width="50%" align="center">
      <img src="assets/README/post-training/grpo-dialogue-1.png" alt="GRPO 北京美食案例" width="100%">
    </td>
    <td width="50%" align="center">
      <img src="assets/README/post-training/grpo-dialogue-2.png" alt="GRPO 周末放松建议案例" width="100%">
    </td>
  </tr>
  <tr>
    <td align="center"><sub>常识信息组织</sub></td>
    <td align="center"><sub>日常建议与指令遵循</sub></td>
  </tr>
</table>

## 图片资产分布 🖼️

README 中的图片统一放在 `assets/README/` 下，并使用英文语义化文件名，避免中文路径或空格导致 GitHub 渲染异常。图片按展示位置分布如下：

| 展示位置 | 文件 | 原始尺寸 | 文件大小 |
| --- | --- | ---: | ---: |
| 项目概览：训练全链路 | `assets/README/training-pipeline.png` | 1190 x 1322 | 1.34 MB |
| 整体架构：模型结构 | `assets/README/model-architecture.png` | 1036 x 1519 | 1.05 MB |
| 预训练流程：数据分布 | `assets/README/pretrain-data-distribution.png` | 1448 x 1086 | 1.15 MB |
| 训练工程优化：学习率调度 | `assets/README/learning-rate-schedule.png` | 646 x 716 | 39.5 KB |
| 训练工程优化：checkpoint 终端 | `assets/README/training-checkpoint-terminal.png` | 2888 x 1850 | 856.9 KB |
| SFT Benchmark：训练曲线 | `assets/README/sft-training-curves.png` | 2350 x 642 | 110.9 KB |
| SFT Benchmark：三维 Judge 指标 | `assets/README/sft-judge-rubric-metrics.png` | 2272 x 1182 | 174.0 KB |
| SFT Benchmark：综合 mean/pass 指标 | `assets/README/sft-judge-mean-score.png` | 1616 x 634 | 67.1 KB |
| 评测与实验结果：SwanLab 指标 | `assets/README/swanlab-training-metrics.png` | 1810 x 722 | 107.4 KB |
| 评测与实验结果：SFT 对话效果 | `assets/README/sft-dialogue-demo.png` | 1280 x 464 | 344.6 KB |
| 交互效果展示：模型启动界面 | `assets/README/zhichen-cli-startup.png` | 1402 x 894 | 166.3 KB |
| 交互效果展示：SFT 评测输出 | `assets/README/sft-evaluation-transcript.png` | 1742 x 530 | 334.2 KB |
| CoT-SFT：模型启动界面 | `assets/README/post-training/model-interface.png` | 1404 x 864 | 166.0 KB |
| CoT-SFT：训练曲线 | `assets/README/post-training/cot-sft-training-curves.png` | 2000 x 712 | 94.7 KB |
| CoT-SFT：晚餐建议 | `assets/README/post-training/cot-sft-dialogue-1.png` | 1764 x 1228 | 445.0 KB |
| CoT-SFT：北京美食 | `assets/README/post-training/cot-sft-dialogue-2.png` | 1776 x 878 | 446.4 KB |
| 身份数据：身份识别 | `assets/README/post-training/identity-dialogue.png` | 1878 x 174 | 52.4 KB |
| GRPO：北京美食 | `assets/README/post-training/grpo-dialogue-1.png` | 1746 x 1046 | 380.6 KB |
| GRPO：周末建议 | `assets/README/post-training/grpo-dialogue-2.png` | 1760 x 722 | 357.9 KB |

## 项目结构 🗂️

```text
.
├── model/
│   ├── config.py                 # 模型配置：hidden size、层数、头数、词表等
│   └── model_spongebob_pro.py    # Decoder-only Transformer 主体实现
├── train/
│   ├── train_tokenizer.py        # 15K Byte-level BPE Tokenizer 训练与验证
│   ├── pretrain.py               # 支持 DDP 的预训练脚本
│   ├── pretrain_without_ddp.py   # 单进程预训练脚本
│   ├── train_sft.py              # SFT 指令微调脚本
│   └── utils.py                  # 学习率调度、DDP 初始化、日志工具
├── dataset/
│   ├── preprocess_data.py        # 预训练数据预处理
│   ├── pretrain_dataset.py       # memmap 预训练数据集
│   ├── prepare_single_turn_sft.py # SFT 单轮筛选与 manifest
│   └── sft_dataset.py            # assistant-only loss 的 SFT 数据集
├── data/
│   └── *.manifest.json           # 数据筛选规则、统计与哈希
├── benchmark/
│   ├── evaluator.py              # Benchmark 评测工具
│   ├── test_xcopa_improved.py    # XCOPA 中文因果推断评测
│   └── mini_bench/               # SFT 生成式评测样例
├── tokenizer_15k/                # 已训练 Tokenizer 文件
└── assets/README/                # README 展示图片
    └── post-training/            # CoT-SFT 与 GRPO 展示图片
```

## 快速开始 ⚡

### 1. 准备环境

建议使用 Python 3.10+ 和 PyTorch 2.x。根据机器 CUDA 版本安装对应 PyTorch 后，再安装项目所需依赖，例如：

```bash
pip install torch transformers tokenizers datasets swanlab
```

### 2. 验证 Tokenizer

```bash
python train/train_tokenizer.py
```

该脚本当前默认执行 Tokenizer 验证逻辑，会加载 `tokenizer_15k/` 并打印词表大小、编码解码一致性和对话模板。

### 3. 启动预训练

```bash
torchrun --nproc_per_node=4 train/pretrain.py \
  --data_path /path/to/pretrain_data \
  --save_dir ./pretrain_out \
  --hidden_size 768 \
  --num_hidden_layers 12 \
  --batch_size 128 \
  --learning_rate 1e-3
```

### 4. 启动 SFT

先生成严格单轮数据：

```bash
python dataset/prepare_single_turn_sft.py \
  --input data/spongebob_sft.jsonl \
  --output data/spongebob_sft_single_512.jsonl \
  --tokenizer tokenizer_15k \
  --max-length 512
```

再启动训练：

```bash
torchrun --nproc_per_node=4 train/train_sft.py \
  --data_path ./data/spongebob_sft_single_512.jsonl \
  --tokenizer_path ./tokenizer_15k \
  --from_weight /path/to/pretrain_768.pth \
  --save_dir ./out_sft \
  --learning_rate 2e-5
```

### 5. 运行评测

```bash
python eval.py
python benchmark/test_xcopa_improved.py
```

## 工程亮点 🌟

- ✅ **从零实现链路完整**：Tokenizer、模型、训练循环、loss、调度器、SFT 掩码和评测均在项目内实现。
- ✅ **结构贴近现代 LLM**：RMSNorm、RoPE、GQA、SwiGLU、KV Cache 和权重绑定均为当前主流 Decoder-only LLM 的关键设计。
- ✅ **训练工程闭环**：支持 DDP、多卡训练、混合精度、checkpoint、SwanLab 追踪和 Benchmark 回传。
- ✅ **中文任务导向**：Tokenizer、预训练语料、SFT 数据和评测任务均围绕中文语言能力构建。
- ✅ **小规模可验证**：约 0.1B 参数规模便于在有限算力下完成端到端实验，同时保留大模型训练框架的核心复杂度。

## 总结 🧭

知辰并不是简单调用现成大模型接口的应用项目，而是一个从底层组件到训练闭环完整搭建的中文大模型训练框架。项目以 15K BBPE Tokenizer 为统一编码基础，手写类 Qwen3 Dense 的 Decoder-only Transformer，并通过多卡预训练和经过筛选的单轮 SFT，让约 0.1B 参数级模型获得基础中文理解、因果推断、逻辑推理和一次性对话能力。CoT-SFT 与 GRPO 后训练由 [TinyAudit-GRPO](https://github.com/0beniko/TinyAudit-GRPO) 继续维护。

该项目的价值在于：通过可读、可改、可训练的代码，把大模型训练中的核心机制拆解为可以验证的工程模块，完整展示从数据到模型、从训练到评测、从 loss 收敛到对话效果的端到端能力。
