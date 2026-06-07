## 实验步骤

1.  **数据构造（本文档）**：用 Teacher (强模型) 的 logprob 作为信号，通过 Greedy 流程生成 `(minus, plus, target_delta)` 训练样本。
2.  **模型训练（报告 8 节）**：用这些样本训练 0.8B 模型，使其学会预测 `delta`。
3.  **推理评估（主实验）**：训练好的 0.8B 模型独立执行 Greedy 选块，并与 4B Reranker 对比。

因此，复现实验需要覆盖这 **三个阶段**。下面我将整合所有信息，制定一份从数据构造到最终评估的完整复现计划，并依然保留本地模型与 API 调用的双重路径选择。

---

## 0.8B Logprob Selector 全流程复现实验计划

### 核心目标
完整复现“基于 Logprob 增量打分的 Greedy 选块”方法，包括：
1.  使用 Teacher 模型构造训练数据。
2.  训练一个 0.8B 的 logprob selector。
3.  在 224 条测试样本上评估该 selector，验证其是否优于 4B Reranker top-k 基线。

---

### 阶段 0：全局准备与模型资源规划

此阶段需确定整个流程的模型方案，并完成环境与数据准备。

#### 0.1 模型方案选择
根据你拥有的资源，为三个角色选择模型。**必须保持同一角色在整个流程中模型一致**。

| 角色         | 任务                            | 方案 A：严格复现 (本地)          | 方案 B：API 近似替代                                         |
| :----------- | :------------------------------ | :------------------------------- | :----------------------------------------------------------- |
| **Teacher**  | 数据构造阶段，计算 logprob      | 本地加载 122B 模型               | **支持 `logprobs` 的 API** (如 OpenAI GPT-4o, DashScope Qwen-Max) |
| **Student**  | 待训练的 0.8B 模型              | 本地加载基座模型并训练           | 本地加载基座模型并训练 **(不可替换)**                        |
| **Selector** | 推理评估阶段，执行选块          | 加载训练好的 Student 权重        | 加载训练好的 Student 权重 **(不可替换)**                     |
| **Reranker** | 推理评估阶段，提供基线          | 本地加载 4B Reranker             | 本地加载 4B Reranker (或其 API 等价物)                       |
| **Reader**   | 推理评估阶段，最终问答          | 本地加载 35B 模型                | **LLM API** (如 GPT-4o, Qwen-Max)                            |
| **裁判**     | 推理评估阶段，精细 logprob 对比 | 本地 122B 模型 (可与 Teacher 同) | **支持 `logprobs` 的 API** (可与 Teacher 同)                 |

-   **最低资源要求**：必须能够本地训练和运行 0.8B 模型。Teacher 和 Reader 可用 API 替代。

#### 0.2 全局参数固化
将以下贯穿全流程的参数写入配置文件，确保一致性：
-   `TEACHER_MODEL`：Teacher 模型名称或路径。
-   `STUDENT_MODEL`：Student 模型基座名称或路径。
-   `RERANKER_MODEL`：Reranker 模型名称或路径。
-   `READER_MODEL`：Reader 模型名称或 API 地址。
-   `CHUNK_SEPARATOR`：上下文拼接分隔符 (如 `"\n\n"` )。
-   **推理阶段参数** (报告中的)：
    -   `CANDIDATE_TOP_P`
    -   `SOLO_RANK_CAP`
    -   `DELTA_OFFSET`

#### 0.3 数据准备
-   **训练集**：用于数据构造的原始样本，格式为 `{"question": ..., "answer": ..., "chunks": [...]}`。数量需足以训练 0.8B 模型。
-   **测试集**：用于最终评估的 224 条固定样本，结构与训练集一致，并标注好数据集来源 (2wikimqa等)。

---

### 阶段 1：训练数据构造

本阶段完全依照你提供的文档，目标是生成 `(minus_context, plus_context, target_delta)` 格式的训练样本。

#### 步骤 1.1：单块打分与排序
-   **输入**：训练集的一条样本 `{question, answer, chunks}`。
-   **操作**：
    1.  对每个 `chunk_i`，拼接 `[chunk_i, question, answer]` 作为 prompt。
    2.  调用 **Teacher 模型** (本地或 API)，传入 prompt，请求返回 `answer` 部分的 token-level `logprobs`。
    3.  对 `logprobs` 求和，得到 `s_i = score(chunk_i, question, answer)`。
-   **输出**：所有 chunk 的分数列表，按分数从高到低排序，得到 `ranked_chunks`。
-   **API 调用注意事项**：
    -   **Prompt 格式**：必须固定 chunk 和 question/answer 的顺序和分隔符。
    -   **Logprob 获取**：需要仔细阅读 API 文档，确保能拿到每个 token 的 logprob。如 OpenAI API 需设置 `logprobs=True`。
    -   **成本/速率控制**：此步调用量大，可先在小样本上测试，确认代码正确和成本可控。

#### 步骤 1.2：Greedy 选块并记录路径
-   **输入**：`{question, answer, ranked_chunks}`。
-   **操作**：
    1.  初始化 `selected = []`, `current_score = -inf`。
    2.  选择 `ranked_chunks[0]` 作为 seed，拼接 `selected_context = [seed]`。
    3.  计算 `current_score = score(selected_context, question, answer)`。
    4.  记录 seed 节点：`{"chunk": seed_id, "action": "seed", "score": current_score}`。
    5.  遍历 `ranked_chunks` 中剩余的 chunk `c_j`:
        -   构造 `plus_context = selected_context + [c_j]`。
        -   计算 `s_plus = score(plus_context, question, answer)`。
        -   若 `s_plus > current_score`，则 **accept**：`selected_context.append(c_j)`，`current_score = s_plus`。记录 `{"chunk": c_j_id, "action": "accept", "score": s_plus}`。
        -   否则 **reject**。记录 `{"chunk": c_j_id, "action": "reject", "score": s_plus}`。
-   **输出**：该条样本的 `greedy_path` (节点列表)。

#### 步骤 1.3：Replay 路径生成训练样本
-   **输入**：`{question, answer, chunks, greedy_path}`。
-   **操作**：
    1.  初始化 `selected_so_far = []`, `current_score = -inf`。
    2.  遍历 `greedy_path` 中的每个节点 `node`:
        -   若 `node.action == "seed"`: 更新 `selected_so_far` 和 `current_score`。
        -   否则 (`accept` 或 `reject`):
            -   构造 `minus_context = selected_so_far` (按 chunk 原始顺序拼接)。
            -   构造 `plus_context = selected_so_far + [chunks[node.chunk]]` (按 chunk 原始顺序拼接)。
            -   计算 `target_delta = node.score - current_score`。
            -   **生成一条训练样本**: `{"question": ..., "answer": ..., "minus_context": ..., "plus_context": ..., "target_delta": ...}`。
            -   若 `node.action == "accept"`: 更新 `selected_so_far` 和 `current_score`。
            -   若 `node.action == "reject"`: 不更新状态。
-   **输出**：由所有非 seed 节点生成的 **训练样本集**。

---

### 阶段 2：模型训练

本阶段对应报告第 8 节，目标是训练 0.8B 模型来拟合 `target_delta`。

#### 步骤 2.1：训练目标与Loss选择
-   **目标**：让 Student 模型学会预测 `delta`。
-   **模型结构**：0.8B 基座模型 + 一个输出单个标量值的 `score head` (可以复用 LM head 或新增线性层，具体参考原始实现)。
-   **Loss 函数**：使用 **Huber Loss**，对异常值更鲁棒。
    `loss = HuberLoss(student_delta, target_delta)`
    其中 `student_delta = student_score(plus) - student_score(minus)`。

#### 步骤 2.2：训练配置与执行
-   **实验设置**：根据报告建议，可尝试两组实验进行消融：
    -   **实验一：仅拟合绝对分数**。Loss = `HuberLoss(student_score(context), teacher_score)`。
    -   **实验二：联合优化**。Loss = `w1*Huber(delta) + w2*Huber(score) + w3*BCE(sign)`。**注意**：报告指出此方法存在多任务冲突风险，需仔细调整权重或作为消融研究。
-   **超参数**：根据报告记录和你的资源调整学习率、batch size、训练轮次等。
-   **验证**：在单独的验证集上监控 `delta_mae`、`delta_sign_acc` 等指标。

---

### 阶段 3：推理与评估

本阶段完全依照复现计划文档中的步骤，使用训练好的 0.8B 模型对测试集进行选块评估。

#### 步骤 3.1：0.8B Selector 流水线
对测试集的每条样本执行以下操作：
1.  **单块排序与候选池**：使用 **Student 模型** 计算所有 chunk 的单块 `delta`，按 softmax 概率和 `CANDIDATE_TOP_P`、`SOLO_RANK_CAP` 构建候选池。
2.  **Greedy 增量选块**：执行 Greedy 选块，使用 **Student 模型** 计算增量 `delta`，并与 `DELTA_OFFSET` 比较来决定 accept/reject。得到 `qwen_selected_chunks`。

#### 步骤 3.2：构建 Reranker 基线
-   使用 **Reranker 模型** 对同一样本的所有 chunk 打分并排序。
-   取前 `k = len(qwen_selected_chunks)` 个 chunk，得到 `reranker_selected_chunks`。

#### 步骤 3.3：Reader 问答准确率评估
-   使用 **Reader 模型** (本地或 API)，分别基于 `qwen_selected_context` 和 `reranker_selected_context` 生成答案。
-   与 `gold_answer` 比对，统计准确率，并分类到 `qwen_win`、`rer_win`、`tie`。
-   **目标**：复现 `0.8B accuracy > 4B reranker accuracy`。

#### 步骤 3.4：(可选) 裁判模型精细评估
-   **如果拥有可靠的 logprob API 或本地裁判模型**：
    -   **压缩增益**：计算 `score(selected) - score(full)` 并比较。
    -   **同数量 Logprob 对比**：在相同 chunk 数量下比较 `score`。
-   **如果无法获得**：可跳过此步，仅基于 QA 准确率得出结论，但需在报告中说明缺失了该部分证据。

---

### 阶段 4：结果汇总与分析
1.  汇总测试集上的准确率、压缩增益等指标，与原始报告对比。
2.  重点分析 `rer_win` 的样本，排查是候选池召回问题还是 Greedy 决策问题。
3.  撰写复现报告，清晰说明每一阶段使用的模型方案（本地/API）、参数及任何差异点。

这个全流程计划现在完整了。它从原始语料出发，覆盖数据构造、模型训练到最终评估，并根据你的实际情况保留了灵活的 API 替代空间。现在可以按此计划逐步推进。
