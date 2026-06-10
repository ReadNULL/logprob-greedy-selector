# Logprob Greedy Selector

中文 | [English](README.en.md)

本项目是一个面向开源复现的 **Logprob-Delta Greedy 上下文选块器** 工程骨架。它的目标是：用强模型的答案 logprob 增益构造监督信号，训练一个较小的 selector 模型，使其能够在长上下文问答任务中选择更有用的 context chunks。

核心思想不是判断某个 chunk 与问题是否“语义相关”，而是判断：

```text
把这个 candidate chunk 加入当前 selected context 后，gold answer 的条件 logprob 是否提升？
```

## 方法定义

设问题为 $q$，标准答案为 $a = (a_1, \dots, a_T)$，上下文为 $C$。Teacher 模型对该上下文的答案支持度定义为 gold answer token logprob 之和：

$$
\mathrm{score}(C, q, a) = \sum_{t=1}^{T} \log p_\theta(a_t \mid C, q, a_{\lt t})
$$

对于加入候选 chunk 前的上下文 $C^-$，以及加入候选 chunk 后的上下文 $C^+$，定义增量分数：

$$
\Delta(C^-, C^+, q, a)
= \mathrm{score}(C^+, q, a) - \mathrm{score}(C^-, q, a)
$$

当 $\Delta > 0$ 时，说明该 candidate chunk 提升了 Teacher 对 gold answer 的条件概率，因此在 greedy 构造中应被接受；否则应被拒绝。

## 当前状态

本仓库已经实现从数据获取到小模型选块的主流程：

```text
LongBench raw samples
  -> Teacher greedy paths
  -> Delta training dataset
  -> Train scalar score model
  -> Run greedy selector
```

已实现：

- LongBench 数据抓取与标准化；
- Teacher gold-answer logprob 打分接口；
- Teacher greedy path 构造；
- greedy path replay 成 `(minus_context, plus_context, target_delta)` 训练样本；
- Transformer backbone + scalar score head 的 delta 训练；
- 推理阶段的 greedy selector；
- `candidate_top_p`、`solo_rank_cap`、`delta_offset` 三个核心推理参数。

暂未实现：

- 4B reranker baseline；
- Reader 模型问答；
- LLM-as-judge 准确率评估；
- 完整论文级实验报告自动生成。

这些模块与具体模型、API 和判分策略强相关，因此当前先保留为后续可插拔扩展。

## 项目结构

```text
configs/
  experiment.yaml                 实验配置草案

selector/
  data/
    schema.py                     JSONL 数据结构
    io.py                         JSONL 读写工具
    context.py                    prompt 构造与 chunk 拼接
    greedy_teacher.py             Teacher greedy path 构造
    build_delta_dataset.py        replay path 生成 delta 数据集

  scoring/
    base.py                       scorer 抽象接口
    hf_logprob_scorer.py          本地 Hugging Face gold-answer logprob scorer
    openai_logprob_scorer.py      OpenAI-compatible completions logprob scorer
    delta_model_predictor.py      训练后 selector 模型推理接口

  model/
    delta_model.py                Transformer backbone + scalar score head
    dataset.py                    Delta JSONL dataset 与 collator

  select/
    greedy_selector.py            推理阶段 greedy selector

scripts/
  00_fetch_longbench_raw.py       获取 LongBench 子集并生成 raw JSONL
  01_build_teacher_paths.py       构造 Teacher greedy paths
  02_build_delta_dataset.py       构造 delta training JSONL
  03_train_delta_model.py         训练 selector score model
  04_run_greedy_selector.py       使用训练后模型执行 greedy 选块

docs/
  *.md                            实验说明与复现规划文档
```

生成的数据、模型权重和输出目录已经通过 `.gitignore` 排除，不会默认提交到 GitHub。

## 安装

推荐 Python 3.10+。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

也可以用 editable mode 安装：

```powershell
pip install -e .
```

## 数据准备

默认从 LongBench 中抽取三个子集：

- `hotpotqa`：80 条；
- `2wikimqa`：80 条；
- `hotpotqa_e`：80 条。

总计 240 条。

运行：

```powershell
python scripts/00_fetch_longbench_raw.py `
  --output data/raw/samples.jsonl `
  --dataset-count hotpotqa=80 `
  --dataset-count 2wikimqa=80 `
  --dataset-count hotpotqa_e=80
```

标准化后的数据格式：

```json
{
  "sample_id": "...",
  "dataset": "hotpotqa",
  "question": "...",
  "answer": "...",
  "chunks": ["chunk 0", "chunk 1"],
  "metadata": {}
}
```

当前 chunk 切分采用工程化的段落合并策略，由 `--max-chunk-words` 控制。若要严格复现某个已有实验，需要替换为原实验的 chunking 规则。

## Teacher Logprob 打分

严格复现需要 Teacher scorer 能计算：

$$
\mathrm{score}(C, q, a) = \sum_{t=1}^{T} \log p_\theta(a_t \mid C, q, a_{\lt t})
$$

注意：这里要求 API 或本地模型能对 **给定的 gold answer** 返回 token logprob，而不是只返回模型自己生成文本的 logprob。

### 方案 A：本地 Hugging Face 模型

这是目前推荐的严格路径。只要能本地加载 causal LM，就可以直接计算 gold answer 的条件 logprob。

```powershell
python scripts/01_build_teacher_paths.py `
  --backend hf `
  --input data/raw/samples.jsonl `
  --output data/processed/teacher_greedy_paths.jsonl `
  --model Qwen/Qwen3-0.6B `
  --device cuda `
  --max-candidates 20
```

真实实验中应替换为更强的 Teacher 模型。`hf_logprob_scorer.py` 会拼接 `Context / Question / Answer`，并只累加 answer token 的 logprob。

### 方案 B：OpenAI-Compatible Completions

如果某个服务商支持 legacy `/completions` 接口，并且同时支持：

```text
echo=True
logprobs=N
```

则可以使用：

```powershell
python scripts/01_build_teacher_paths.py `
  --backend openai-compatible-completions `
  --input data/raw/samples.jsonl `
  --output data/processed/teacher_greedy_paths.jsonl `
  --model YOUR_MODEL `
  --openai-base-url YOUR_BASE_URL `
  --openai-api-key-env YOUR_API_KEY_ENV
```

实测注意事项：

- DeepSeek Chat 和 Qwen/DashScope Chat 可以返回生成 token 的 logprob；
- DeepSeek beta completions 在本地 smoke test 中拒绝 `echo + logprobs`；
- Qwen/DashScope OpenAI-compatible completions 在本地 smoke test 中不支持 `qwen-plus` / `qwen-turbo` 走该严格路径。

因此，DeepSeek/Qwen Chat logprobs 可以用于生成答案置信度分析，但不等价于本项目所需的严格 gold-answer Teacher scoring。

## 构造 Delta 训练数据

当 Teacher greedy paths 构造完成后，运行：

```powershell
python scripts/02_build_delta_dataset.py `
  --input data/processed/teacher_greedy_paths.jsonl `
  --output data/processed/delta_train.jsonl
```

每个非 seed 节点会生成一条训练样本：

```json
{
  "question": "...",
  "answer": "...",
  "minus_context": "...",
  "plus_context": "...",
  "minus_context_ids": [0],
  "plus_context_ids": [0, 2],
  "target_delta": 1.234,
  "action": "accept"
}
```

Replay 规则：

- `seed` 只初始化 selected context，不产生训练样本；
- `accept` 产生训练样本，并更新 selected context；
- `reject` 产生训练样本，但不更新 selected context。

Replay 阶段的监督目标为：

$$
y_\Delta = s_{\text{step}} - s_{\text{current}} = \mathrm{score}(C^+, q, a) - \mathrm{score}(C^-, q, a)
$$

其中 $s_{\text{current}}$ 是测试 candidate 前当前 selected context 的 Teacher score， $s_{\text{step}}$ 是加入 candidate 后的 Teacher score。

## 训练 Selector

训练 Transformer backbone + scalar score head：

```powershell
python scripts/03_train_delta_model.py `
  --train data/processed/delta_train.jsonl `
  --base-model Qwen/Qwen3-0.6B `
  --output-dir outputs/delta_model `
  --device cuda `
  --batch-size 1 `
  --grad-accum 8 `
  --epochs 1 `
  --lr 1e-5
```

训练时，小模型学习一个标量函数 $f_\phi(C, q, a)$。对于每条训练样本：

$$
\hat{s}^- = f_\phi(C^-, q, a)
$$

$$
\hat{s}^+ = f_\phi(C^+, q, a)
$$

$$
\hat{\Delta}
= \hat{s}^+ - \hat{s}^-
$$

默认损失函数为：

$$
\mathcal{L}
= \mathrm{HuberLoss}(\hat{\Delta}, y_\Delta)
$$

低资源 smoke test 可以加 `--freeze-backbone`，只训练 scalar head。真实实验中应根据 GPU 显存调整 batch size、max length、学习率、训练轮数和验证集比例。

## 推理阶段 Greedy 选块

使用训练好的模型执行选块：

```powershell
python scripts/04_run_greedy_selector.py `
  --input data/raw/samples.jsonl `
  --output data/processed/selector_outputs.jsonl `
  --model-dir outputs/delta_model `
  --device cuda `
  --candidate-top-p 0.95 `
  --solo-rank-cap 20 `
  --delta-offset 0.0
```

推理逻辑：

1. 对每个单 chunk 计算相对于 empty context 的 delta；
2. 按 single-chunk delta 排序；
3. 用 softmax top-p 和 `solo_rank_cap` 构造候选池；
4. 用最高 single-delta chunk 作为 seed；
5. 对候选池剩余 chunk 做 greedy 增量判断；
6. 当预测增量不低于 `delta_offset` 时 accept。

单块排序分数为：

$$
\hat{\Delta}_i
= f_\phi(c_i, q, a) - f_\phi(\varnothing, q, a)
$$

对于当前 selected context $C_{\text{sel}}$ 和候选 chunk $c_j$，greedy 增量为：

$$
\hat{\Delta}_j = f_\phi(C_{\text{sel}} \cup \{c_j\}, q, a) - f_\phi(C_{\text{sel}}, q, a)
$$

接受规则为：

$$
\hat{\Delta}_j \ge \tau
$$

其中 $\tau$ 即 `delta_offset`。

## 完整流程

```powershell
python scripts/00_fetch_longbench_raw.py --output data/raw/samples.jsonl

python scripts/01_build_teacher_paths.py `
  --backend hf `
  --input data/raw/samples.jsonl `
  --output data/processed/teacher_greedy_paths.jsonl `
  --model YOUR_TEACHER_MODEL

python scripts/02_build_delta_dataset.py `
  --input data/processed/teacher_greedy_paths.jsonl `
  --output data/processed/delta_train.jsonl

python scripts/03_train_delta_model.py `
  --train data/processed/delta_train.jsonl `
  --base-model YOUR_0P8B_BASE_MODEL `
  --output-dir outputs/delta_model

python scripts/04_run_greedy_selector.py `
  --input data/raw/samples.jsonl `
  --output data/processed/selector_outputs.jsonl `
  --model-dir outputs/delta_model
```

## 复现边界

原始实验说明中，目标是在一个 bucket-sampled 的 224 条评估集上，让 0.8B selector 超过 4B reranker top-k。本仓库当前不直接声称复现该数值。

要复现该结果，还需要明确：

- 强 Teacher 模型；
- 原始 chunking 规则；
- 具体 0.8B base model 和训练超参；
- 4B reranker baseline；
- Reader 模型；
- answer judge 策略；
- 224 条 bucket-sampled 评估协议。

本仓库的目标是先提供稳定、清晰、可扩展的工程骨架，使这些组件后续可以逐步接入。

## 开发与检查

语法检查：

```powershell
python -m compileall selector scripts
```

不要提交生成文件：

```text
data/raw/
data/processed/
outputs/
```

这些目录已在 `.gitignore` 中排除。

## License

MIT. See [LICENSE](LICENSE).
