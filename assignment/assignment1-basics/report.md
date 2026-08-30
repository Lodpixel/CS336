# CS336 Assignment 1 Report

Spring 2026 自学记录。这里只记录书面题和实验题的完成状态、结果与回答；不替代最终排版的 `writeup.pdf`。

## 使用方式

- 书面题：在“我的回答/记录”下填写答案。
- 实验题：保存配置、日志、曲线、checkpoint 和生成样例，并在对应位置记录结果。
- 代码实现和本地测试不在此文件中记录。

## 进度总览

- [ ] Section 2：BPE Tokenizer 的书面题与实验
- [ ] Section 3：Transformer resource accounting
- [ ] Section 4：Training 的书面题与 resource accounting
- [ ] Section 7：Experiments

---

## 2. Byte-Pair Encoding (BPE) Tokenizer

### Problem `unicode1` — Understanding Unicode（1 point）

- [ ] (a) 说明 `chr(0)` 返回的 Unicode 字符。
- [ ] (b) 说明该字符的 `__repr__()` 与打印表示之间的区别。
- [ ] (c) 说明该字符出现在文本中时的表现。

**我的回答：**

- (a) '\x00'
- (b) print 的时候就不显示，string representation 是 '\x00'
- (c) 不显示

### Problem `unicode2` — Unicode Encodings（3 points）

- [ ] (a) 说明相比 UTF-16/UTF-32，为什么适合在 UTF-8 字节上训练 tokenizer。
- [ ] (b) 给出错误 UTF-8 解码函数产生错误结果的输入，并解释原因。
- [ ] (c) 给出一个无法解码为 Unicode 字符的两字节序列，并解释原因。

**我的回答：**

- (a) UTF-8 的初始词表比较小，只有 256 个，同时也能组合出所有可能的字符，比较简单
- (b) 有些字符是组合而成的，不能逐字节解读
- (c) 10011100 10111000 
- bytes: b'\x9c\xb8'
    hex: 9c b8
    UnicodeDecodeError: 'utf-8' codec can't decode byte 0x9c in position 0: invalid start byte

### Problem `train_bpe_tinystories` — BPE Training on TinyStories（2 points）

- [ ] (a) 用 TinyStories 训练 10K vocabulary，并加入 `<|endoftext|>`；保存 vocabulary 和 merges。
- [ ] (a) 记录训练耗时、内存占用和 vocabulary 中最长 token，并说明是否合理。
- [ ] (b) profile BPE 训练，找出最耗时的部分。

**实验配置与结果：**

**我的回答：**

### Problem `train_bpe_expts_owt` — BPE Training on OpenWebText（2 points）

- [ ] (a) 用 OpenWebText 训练 32K vocabulary，并保存 vocabulary 和 merges。
- [ ] (a) 找出 vocabulary 中最长 token，并说明是否合理。
- [ ] (b) 对比 TinyStories tokenizer 和 OpenWebText tokenizer。

**实验配置与结果：**

**我的回答：**

### Problem `tokenizer_experiments` — Experiments with tokenizers（4 points）

- [ ] (a) 对 TinyStories 和 OpenWebText 各抽取 10 个文档，计算 compression ratio（bytes/token）。
- [ ] (b) 用 TinyStories tokenizer 编码 OpenWebText，比较 compression ratio 或文本表现。
- [ ] (c) 测量 tokenizer throughput，并估算处理 Pile 所需时间。
- [ ] (d) 编码训练集和验证集，说明为什么 token IDs 适合保存为 `uint16`。

**实验配置与结果：**

**我的回答：**

---

## 3. Transformer Language Model Architecture

### Problem `transformer_accounting` — Transformer LM resource accounting（5 points）

- [ ] (a) 计算 GPT-2 XL 配置的参数量和模型占用内存。
- [ ] (b) 列出一次 forward pass 中的矩阵乘法及总 FLOPs。
- [ ] (c) 判断各模型部分中 FLOPs 最大的部分。
- [ ] (d) 对 GPT-2 small/medium/large 做 FLOPs 比例分析。
- [ ] (e) 将 context length 增加到 16,384，分析 FLOPs 和比例变化。

**我的回答：**

---

## 4. Training a Transformer LM

### Problem `learning_rate_tuning` — Tuning the learning rate（1 point）

- [ ] 在 toy SGD 示例中测试学习率 `1e1`、`1e2`、`1e3`。
- [ ] 记录 loss 衰减、变慢或发散的现象。

**实验结果：**

**我的回答：**

### Problem `adamw_accounting` — Resource accounting for training with AdamW（2 points）

- [ ] (a) 分析 parameters、activations、gradients 和 optimizer state 的 peak memory。
- [ ] (b) 将表达式代入 GPT-2 XL 配置，估算 80GB 内存下的最大 batch size。
- [ ] (c) 计算一次 AdamW step 的 FLOPs。
- [ ] (d) 根据 MFU 估算单张 H100 训练 GPT-2 XL 所需时间。

**我的回答：**

---

## 7. Experiments

### Problem `experiment_log` — Experiment logging（3 points）

- [ ] 建立实验追踪和 logging infrastructure。
- [ ] 记录 gradient steps 和 wall-clock time 对应的训练/验证 loss。
- [ ] 建立覆盖本节实验的 experiment log 文档。

**日志位置：**

**记录：**

### Problem `learning_rate` — Tune the learning rate（3 points）

- [ ] 对多个 learning rates 做 sweep，并记录最终 loss 或发散情况。
- [ ] 保存 learning curves，并说明搜索策略。
- [ ] 训练一个 TinyStories 模型并记录 validation loss；标准目标为不超过 1.45，CPU/MPS 可使用手册中的低资源配置并记录实际结果。

**实验配置与结果：**

**我的分析：**

### Problem `batch_size_experiment` — Batch size variations（1 point）

- [ ] 将 batch size 从 1 测试到 GPU memory limit。
- [ ] 至少包含若干中间值，例如 64 和 128，并在必要时重新调 learning rate。
- [ ] 保存 learning curves。
- [ ] 讨论 batch size 对训练的影响。

**实验配置与结果：**

**我的分析：**

### Problem `generate` — Generate text（1 point）

- [ ] 使用训练好的 checkpoint 和 decoder 生成文本。
- [ ] 记录至少 256 tokens 的文本，或记录到第一个 `<|endoftext|>` 为止。
- [ ] 评论文本流畅度，并说明至少两个影响生成质量的因素。

**生成文本：**

**我的分析：**

### Problem `layer_norm_ablation` — Remove RMSNorm and train（1 point）

- [ ] 移除 Transformer blocks 中的 RMSNorm 并训练。
- [ ] 测试之前的最优 learning rate。
- [ ] 尝试更低学习率以恢复稳定性。
- [ ] 保存对应 learning curves，并评论 RMSNorm 的影响。

**实验配置与结果：**

**我的分析：**

### Problem `pre_norm_ablation` — Implement post-norm and train（1 point）

- [ ] 将 pre-norm Transformer 修改为 post-norm。
- [ ] 训练 post-norm 模型。
- [ ] 保存与 pre-norm 模型的对比 learning curve。

**实验配置与结果：**

**我的分析：**

### Problem `no_pos_emb` — Implement NoPE（1 point）

- [ ] 移除 RoPE，得到不使用 position embedding 的 NoPE 模型。
- [ ] 对比 RoPE 和 NoPE 的训练表现。
- [ ] 保存对比 learning curve。

**实验配置与结果：**

**我的分析：**

### Problem `swiglu_ablation` — SwiGLU vs. SiLU（1 point）

- [ ] 实现不带 GLU 的 SiLU feed-forward network。
- [ ] 调整结构，使其与 SwiGLU 具有近似参数量。
- [ ] 对比两者的 learning curves。
- [ ] 讨论实验结果。

**实验配置与结果：**

**我的分析：**

### Problem `main_experiment` — Experiment on OpenWebText（2 points）

- [ ] 使用与 TinyStories 相同的模型架构和训练迭代数在 OpenWebText 上训练。
- [ ] 保存 OpenWebText learning curve，并比较两种数据集上的 loss。
- [ ] 生成 OpenWebText 模型的文本并记录流畅度。
- [ ] 分析为什么相同模型和 compute budget 下，OpenWebText 的输出质量可能更差。

**实验配置与结果：**

**我的分析：**

### Problem `leaderboard` — Leaderboard（6 points）

- [ ] 在 leaderboard 规则下训练模型，并记录 wall-clock time 小于 45 分钟的曲线。
- [ ] 记录最终 validation loss 和所做的改动。
- [ ] 确认实验满足数据和资源规则。
- [ ] 如参加 leaderboard，准备并提交对应结果；如暂不参加，在备注中记录跳过原因。

**实验配置与结果：**

**我的分析：**

---

## 最终整理

- [ ] 所有书面题和实验结果已填写。
- [ ] 图表、生成文本和 checkpoint 路径已整理。
- [ ] 将书面内容排版为 `writeup.pdf`。
