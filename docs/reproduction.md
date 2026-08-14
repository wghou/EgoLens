# EgoLens 复现记录

从 Qwen2.5-VL-3B 出发，在 EgoAfford 上完整训练一轮 EgoLens 并在两个测试集上评测。
本文记录配置、结果、与论文的差距，以及过程中发现的代码问题。

- 训练：2026-08-11 21:58 → 08-12 16:33，4× NVIDIA H200
- 评测：2026-08-12 21:21 → 22:20，1× NVIDIA H20-3e
- 代码基线：`44358df`（Initial commit, clean snapshot）
- 原始数据：[`results/eval_metrics.json`](results/eval_metrics.json)、
  [`results/eval_generated.log`](results/eval_generated.log)、
  [`results/eval_real.log`](results/eval_real.log)

## 1. 结论摘要

文本规划指标复现到论文的 5–8% 以内，**分割指标低 0.09**，是唯一实质性缺口。

| 指标 | 本次（生成测试集） | 论文 | 差距 | 本次（Real） | 论文 | 差距 |
|---|---:|---:|---:|---:|---:|---:|
| gIoU | 0.607 | 0.700 | **−0.093** | 0.640 | 0.666 | −0.026 |
| cIoU | 0.392 | 0.486 | **−0.094** | 0.347 | 0.455 | −0.108 |
| First-Step Sim | 0.628 | 0.666 | −0.038 | 0.585 | 0.631 | −0.046 |
| Semantic F1 | 0.458 | 0.500 | −0.042 | 0.403 | 0.426 | −0.023 |
| CSR | 0.592 | 0.624 | −0.032 | 0.604 | 0.609 | −0.005 |
| Coverage | 0.534 | 0.566 | −0.032 | 0.448 | 0.481 | −0.033 |

差距来自训练而非评测：训练期的 `mask_dice_loss` 全程比官方 checkpoint 高 4–5 倍，
且从 epoch 5 就分叉，测试集 gIoU 的落差与之同向（见 §3.2、§5.2）。

## 2. 环境

镜像自包含，构建期从 GitHub 克隆源码并编译 SAM2 CUDA 扩展，运行时不挂载宿主机内容。
详见 [`../docker/Dockerfile`](../docker/Dockerfile)。

| | |
|---|---|
| Python / PyTorch / CUDA | 3.11.15 / 2.5.1+cu124 / 12.4 |
| transformers / trl / accelerate | 4.51.3 / 0.17.0 / 1.6.0 |
| deepspeed / flash-attn | 0.15.4 / 2.7.0.post2 |
| 分割器 | SAM2-Hiera-Large（官方 072824，md5 `08083462423be3260cd6a5eef94dc01c`） |

两个环境约束值得记录，它们决定了准备工作的形态：

- **无外网**。计算节点只能访问内网 PyPI 镜像，`github.com`、`huggingface.co`、
  `dl.fbaipublicfiles.com` 均不可达。SAM2 权重和文本 cross-encoder 都需在有外网的机器
  下载后放到共享盘。
- **容器 overlay 仅 20 GB**，而单个 checkpoint 约 8.9 GB。所有产物必须写到共享的
  JuiceFS 卷，否则第二个 checkpoint 就会写满文件系统。

数据与权重通过软链接组织成 README 要求的布局，实体全部位于共享盘：

```
data/EgoAfford            2000 scenes (scene_1~100 测试, 101~2000 训练)
data/EgoAfford_real       26 scenes / 102 张真机图像
pretrained/Qwen/Qwen2.5-VL-3B-Instruct
pretrained/sam2_hiera_large.pt
```

`train_qwen2p5_3b_full_auto.sh` 和 `eval_full_auto.sh` 会在启动时自动建立这些软链、
按实际显卡重编 SAM2 扩展、并做前置检查，可在全新容器里作为唯一命令运行。

## 3. 训练

### 3.1 配置

保持论文的有效 batch 512，因此 lr / 调度 / epoch 数均沿用论文值，结果可比。

| | 论文 | 本次 |
|---|---|---|
| GPU | 8× H200 | 4× H200 |
| per-device batch × 累积 | 16 × 4 | 16 × **8** |
| 有效 batch | 512 | 512 |
| epoch / 总步数 | 40 / — | 40 / 1160 |
| lr / 调度 / warmup | 3e-5 / cosine / 100 | 同 |
| 精度 / 并行 | bf16 / ZeRO-3 | 同 |
| 墙钟 | 12 h | **18 h 42 m** |

`TORCH_CUDA_ARCH_LIST` 必须按实际显卡设置（H200 为 `9.0`）。镜像里烘死的值会被
DeepSpeed 运行时 JIT 继承，在不同代显卡上产生 `no kernel image is available` 。

### 3.2 结果

```
status      SUCCESS      1160/1160 步, epoch 39.98
wall time   18h42m39s    58 s/step (train_runtime 66890 s)
final       loss 0.1017  lm_loss 3.24e-4  dice 0.1796  bce 0.01527  res 0.1061
train_loss  0.2578（全程平均）
产物        34 GB, final model 2 shard + checkpoint-1102/1131/1160
显存峰值    gpu0 98%  gpu1 98%  gpu2 98%  gpu3 73%   (141 GB/卡)
```

与官方 checkpoint 的 `trainer_state.json` 按 epoch 对齐：

| epoch | 官方 lm_loss | 官方 dice | 官方 bce | 本次 lm_loss | 本次 dice | 本次 bce |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.08362 | 0.1808 | 0.01668 | 0.07489 | 0.4131 | 0.05334 |
| 10 | 0.04449 | 0.0725 | 0.00736 | 0.06155 | 0.3877 | 0.05665 |
| 20 | 0.00855 | 0.0535 | 0.00508 | 0.02792 | 0.2682 | 0.02278 |
| 30 | 0.00018 | 0.0446 | 0.00259 | 0.00187 | 0.2147 | 0.01697 |
| 36 | 0.00010 | 0.0335 | 0.00242 | 0.00043 | 0.1593 | 0.01224 |

`lm_loss` 基本追平（同量级），`mask_dice_loss` 全程高 4–5 倍且 **epoch 5 即分叉** ——
不是训练时长不足，而是分割分支从一开始就在另一条曲线上。

显存峰值 98% 说明 per-device 16 在 141 GB 卡上贴着上限，前两步出现过
`pytorch allocator cache flushes` 警告（之后消失，属启动期瞬态）。同配置重跑建议
`PER_DEVICE_BS=8`（累积自动变 16，有效 batch 不变）留余量。

## 4. 评测

单卡 H20-3e，生成测试集 488 样本 48 分 34 秒，Real 102 样本 8 分 53 秒（6–8 s/样本，
显存约 12 GB）。评测流程是两遍前向：先 `generate` 出计划文本，再把生成序列连同
learnable query 喂回 `forward` 取 mask。

### 4.1 整体指标

| | giou | ciou | first_step_sim | sim(F1) | order | coverage | constraint(CSR) | hard_constraint | dag_f1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 生成测试集 | 0.6071 | 0.3919 | 0.6277 | 0.4578 | 0.8334 | 0.5342 | 0.5918 | 0.4980 | 0.6549 |
| Real（零样本） | 0.6397 | 0.3474 | 0.5849 | 0.4029 | 0.8177 | 0.4480 | 0.6042 | 0.5294 | 0.6497 |

### 4.2 分角色拆解（生成测试集）

| 角色 | gIoU | cIoU | non_empty_mIoU | presence_F1 | GT 出现率 | tp/fp/fn/tn |
|---|---:|---:|---:|---:|---:|---|
| direct_object | 0.547 | 0.370 | **0.534** | 0.970 | 90.6% | 430/15/12/31 |
| instrument | 0.661 | 0.232 | **0.336** | 0.727 | 37.1% | 129/45/52/262 |
| destination | 0.613 | 0.418 | **0.544** | 0.864 | 66.0% | 277/42/45/124 |

读这张表要注意 **gIoU 会奖励"经常缺省"的角色**：它把"正确预测为空"记作 IoU=1，而
instrument 有 63% 的样本 GT 为空。所以 instrument 的 gIoU（0.661）虽然高于
direct_object（0.547），实际却是最差的角色 —— 看 `non_empty_mIoU` 才准，
instrument 只有 0.336，另两个是 0.53–0.54。

instrument 的 `presence_F1` 也最低（0.727，45 假阳 + 52 假阴），说明模型对"这一步要不要
用工具"判断得最不准。这与 §5.2 的损失权重问题吻合。

### 4.3 分角色拆解（Real，零样本）

| 角色 | gIoU | cIoU | non_empty_mIoU | presence_F1 | GT 出现率 |
|---|---:|---:|---:|---:|---:|
| direct_object | 0.445 | 0.428 | 0.440 | 1.000 | 99.0% |
| instrument | 0.915 | 0.494 | 0.531 | 0.800 | 11.8% |
| destination | 0.558 | 0.278 | 0.405 | 0.910 | 67.6% |

Real 集上 instrument 的 GT 出现率只有 11.8%（102 个样本里 12 个），gIoU 0.915 几乎全部
来自"正确弃权"的白送分，参考价值很低。真正下降的是 direct_object：
non_empty_mIoU 从生成集的 0.534 掉到 0.440，这是真实图像域偏移的代价。

## 5. 发现的问题

### 5.1 `eval/metrics.py` 用错了文本模型（阻塞级，已修）

```python
_CE_MODEL = CrossEncoder("sentence-transformers/stsb-roberta-base", device='cuda')
```

`sentence-transformers/stsb-roberta-base` 是**双塔编码器**（`architectures:
["RobertaModel"]`，带 `sentence_bert_config.json`）。用 `CrossEncoder` 加载它会构造
`RobertaForSequenceClassification`，其回归头**随机初始化**：

```
newly initialized: ['classifier.dense.weight', 'classifier.out_proj.weight', ...]
```

实测无区分能力，而 `compute_text_planning_score` 的阈值是 0.6：

| 句对 | 上游 id | `cross-encoder/stsb-roberta-base` |
|---|---:|---:|
| 完全相同 | 0.5582 | 0.9968 |
| 同义 | 0.5569 | 0.9967 |
| 毫不相关 | 0.4763 | 0.0024 |

上游 id 的所有分数都低于阈值，会使 **Semantic F1 / Coverage / CSR 恒为 0**。论文报的是
0.500 / 0.566 / 0.624，说明其实际使用的不是这个模型。

旁证：`metrics.py` 第 7 行 `from sentence_transformers import SentenceTransformer,
util, CrossEncoder` 中 `SentenceTransformer` 和 `util` 从未被使用 —— 早期版本应是用双塔
加余弦相似度，改成 CrossEncoder 时漏改了 repo id。

**修法**：改为 `cross-encoder/stsb-roberta-base`，并支持 `EGOLENS_CE_MODEL` 环境变量覆盖
以便复现原行为。注意 gIoU / cIoU 完全不依赖文本模型，不受此问题影响。

### 5.2 `mask_empty_loss` 漏了归一化（疑似分割差距的根因，未修）

`trainer_sft.py` 的 `compute_loss` 里：

```python
mask_bce_loss   = mask_bce_loss  / num_elements    # num_elements = 3B
mask_dice_loss  = mask_dice_loss / num_elements
mask_empty_loss = mask_empty_loss                  # ← 唯独这一项没有除

res_loss = (mask_bce_loss + mask_dice_loss) / (non_empty_count + 1e-8) * batch_size
         + mask_empty_loss / (empty_count + 1e-8) * batch_size
```

化简后：

```
res_loss = 平均(BCE + Dice) / 3  +  B × 平均(空 mask BCE)
```

第二项被乘了 batch size（16），第一项被除了 3，**空 mask 项的有效权重高约 48 倍**。
从代码结构看（唯独这一项漏掉 `/num_elements`）更像归一化疏漏而非刻意设计。

可观测的后果：训练日志里 `res_loss` 六步之内从 3.87 崩到 0.47，而 `mask_dice_loss`
在同期几乎不动 —— 模型优先把"该弃权就输出全负 logit"学到位。代价落在出现频率最低的
instrument 上（`non_empty_mIoU` 仅 0.336、`presence_F1` 0.727）。

这条与 §3.2 的训练曲线分叉、§4.2 的分角色表构成一致的叙事，但**尚未验证**：手上只有官方
的 checkpoint 和 `trainer_state.json`，没有他们的实际训练脚本，无法确认官方是否修了这里。
建议的对照实验是补上归一化重训一轮（4×H200 约 18 小时）：

```python
mask_empty_loss = mask_empty_loss / num_elements
```

### 5.3 `<think>` teacher forcing 带来的训练/推理不一致（方法层面）

训练时 `<think>` 段里放的是**真实**剩余计划，作为 teacher-force 的语义前缀（不计 loss）。
64 个 learnable query 排在序列末尾，因此始终能读到标准答案。推理时则是先 `generate`
出模型自己的计划，再做第二遍 forward 取 mask。

这解释了论文里 EgoLens-Seg（给定参考动作，gIoU 0.839）与完整任务（gIoU 0.700）之间的
落差。属于设定固有的 exposure bias，要改需要动训练目标（例如按概率用模型自己的生成替换
`<think>` 内容做 scheduled sampling）。

### 5.4 训练/推理的 mask 选择逻辑不一致（已做对照实验，影响很小）

`build_sam2` 默认 `apply_postprocessing=True`，注入 `dynamic_multimask_via_stability`。
而 `mask_decoder.py`：

```python
elif self.dynamic_multimask_via_stability and not self.training:
    masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
else:
    masks = masks[:, 0:1, :, :]
```

训练时恒取第 0 个 mask token；评测脚本调了 `model.eval()`，于是走稳定性筛选，**可能换成
另一个 mask token 的输出**。即训的是 token 0，测的时候未必是 token 0。

**对照实验**（同一 checkpoint，生成测试集 488 样本，仅关闭该开关，
`EGOLENS_DYNAMIC_MULTIMASK=0`）：

| 指标 | 开（上游默认） | 关（钉住 token 0） | 差值 |
|---|---:|---:|---:|
| gIoU | 0.6071 | 0.6099 | +0.0028 |
| cIoU | 0.3919 | 0.3950 | +0.0031 |
| First-Step Sim | 0.6277 | 0.6290 | +0.0012 |
| Semantic F1 / Coverage / CSR / order / dag_f1 | — | — | 完全相同 |

| 角色 | non_empty_mIoU（开→关） | cIoU（开→关） |
|---|---|---|
| direct_object | 0.5336 → 0.5346 (+0.0010) | 0.3701 → 0.3671 (−0.0030) |
| instrument | 0.3357 → 0.3479 (**+0.0121**) | 0.2319 → 0.2522 (**+0.0203**) |
| destination | 0.5439 → 0.5485 (+0.0046) | 0.4183 → 0.4243 (+0.0060) |

**结论：钉住 token 0 确实带来一致的小幅提升，但只有 +0.003 gIoU，仅能解释 0.093 差距的
约 3%。** 该假设基本被排除，剩余差距指向训练侧（§5.2）。收益集中在 instrument
（cIoU +0.020），与"最弱角色对输出选择最敏感"相符。

三个纯文本指标完全不变，符合预期。但 First-Step Sim 变了 0.0012 —— 因为它用的参考句来自
`compute_best_mask_iou_multi` 按 **mask IoU** 选出的最佳候选，mask 一变，选中的候选可能就
变了。这是一处 mask 与文本指标之间的隐式耦合，读指标时值得留意。

对照实验的数据：[`results/eval_metrics_dynmask_off.json`](results/eval_metrics_dynmask_off.json)，
共享盘上目录布局为 `evaluations_multi`（上游默认行为）、`evaluations_multi.dynmask_off`
（对照组）、`evaluations_multi.baseline_dynmask_on`（基线备份）。

实现方式是在 `evaluate_egoaff_multi.py` 的模型加载后加一个环境变量开关，默认保持上游行为：

```python
if os.environ.get("EGOLENS_DYNAMIC_MULTIMASK", "1") == "0":
    self.model.sam.sam_mask_decoder.dynamic_multimask_via_stability = False
```

### 5.5 其它已确认的小问题

| 位置 | 问题 |
|---|---|
| `trainer_sft.py` | `if_freeze_llm` 永远不生效：`sft_vllm_sam.py` 构造 Trainer 时没传该参数；且实现写的是 `model.visual.requires_grad = False`，对 `nn.Module` 无效（应为 `requires_grad_(False)`） |
| `trainer_sft.py` | `build_answer_only_labels` 实际监督范围是 `<answer>` 到序列末尾，与函数名和注释不符（span 版本被注释掉了）。三条分支的行为见 §5.6，实测本数据集不触发问题分支 |
| `samr1.py` + `trainer_sft.py` | SAM2 被构建两次：`__init__` 里先建一个随机初始化的并 `del` 掉三个 memory 模块，trainer 再整体替换成从权重加载的实例 —— 后者**带回了** `memory_attention` / `memory_encoder` / `maskmem_tpos_enc`。这三个模块前向不用、冻结无梯度，但占显存且被写进 checkpoint |
| `arguments.py` | `prompt_difficulty` 默认值 `"easy"` 不在 `dataset.py` 的分支里，用默认值会因 `self.question_template` 未定义而 `AttributeError`；`res_loss_ratio` 定义了但从未使用 |
| `sft_vllm_sam.py` | `eval_dataset` 用的是 `split='val'`，即 scene_1~100（官方测试集）。因未设 `eval_strategy` 实际从未执行，没有产生泄漏，但要加验证监控时不能直接用它 |
| 性能 | SAM2 解码是纯 Python 串行双层循环（每 micro-batch 16×3 = 48 次 decoder 调用），且 loss 在原图分辨率（约 768×1024）用 fp32 计算。两处都可优化一个数量级 |

### 5.6 截断假设：已实测排除

`build_answer_only_labels` 有三条分支，其中两条在样本被 `max_length=2048` 截断时才会命中，
而截断是从序列尾部开始的（顺序为 system → user+图像 → assistant 的 `<think>…</think>
<answer>…</answer>`），所以 `</answer>` 最先被切掉：

| 情况 | 分支 | 后果 |
|---|---|---|
| `<answer>` 和 `</answer>` 都在 | `start != -1 and end != -1` | 正常，从 `<answer>` 之后开始监督 |
| `<answer>` 在、`</answer>` 被切 | **两条分支都不满足** | labels 全为 −100，该样本**完全没有语言监督**；若整个 micro-batch 都如此，`F.cross_entropy` 会返回 nan |
| `<answer>` 也被切 | `start == -1 and end == -1` | `labels[:] = ids`，该样本被训练去复现 prompt 模板 |

用 [`results/truncation_stats.py`](results/truncation_stats.py) 在全部 15049 个训练样本上
实测（复现了 collator 的 tokenize 流程，并直接复用 `build_answer_only_labels`）：

```
token 长度（未截断）  p50 1553  p90 1644  p95 1671  p99 1722  max 1936   mean 1558
> 2048               0 / 15049  (0.00%)
分支分布             ok 15049 (100%)   no_super 0   full 0
零监督 token 的样本    0
```

**结论：零截断，问题分支从未命中，该假设排除。** 完整输出见
[`results/truncation_stats.txt`](results/truncation_stats.txt)。

但余量只有 112 个 token（1936 vs 2048），属于"恰好没出事"。视觉 token 数由
`max_pixels=1000000` 决定（约 1275 个），一旦提高图像分辨率、加长 prompt 模板，或换一个
步骤更多的数据集，就会**静默**落入上面两条问题分支。建议给这两条分支各加一条计数日志。

## 6. 复现方式

全新容器里各一条命令即可，脚本自己完成软链、扩展重编、卡数适配和前置检查：

```bash
bash scripts/train_qwen2p5_3b_full_auto.sh          # 训练，卡数自适应，有效 batch 恒为 512
SPLIT=both bash scripts/eval_full_auto.sh           # 评测两个 split
SETUP_ONLY=1 bash scripts/eval_full_auto.sh         # 只做准备并打印配置
```

无外网环境需先把两个模型放到共享盘（在有外网的机器上下载后拷贝）：

```bash
# SAM2 权重，897952466 字节, md5 08083462423be3260cd6a5eef94dc01c
curl -L -A 'Mozilla/5.0' -o sam2_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
#   → $DISK/models/facebook/sam2-hiera-large/sam2_hiera_large.pt

# 文本 cross-encoder（hf-mirror.com 可用时）
HF_ENDPOINT=https://hf-mirror.com HF_HOME=/tmp/hf python -c \
  "from huggingface_hub import snapshot_download as d; d('cross-encoder/stsb-roberta-base')"
#   → $DISK/cache/huggingface/hub/models--cross-encoder--stsb-roberta-base
```

评测结果写在 `${MODEL_PATH}/evaluations_multi[_real]/`，与 checkpoint 同目录，包含
`egoaff_metrics.json`、`egoaff_metrics_by_scene.json` 和 `--vis` 每 50 个样本的分割可视化。

## 7. 后续建议

已排除两个候选解释：

- §5.4 训练/推理的 mask 选择不一致 —— 对照实验只值 +0.003 gIoU（占差距 3%）
- §5.6 `max_length=2048` 截断 —— 全量实测零截断

**`mask_empty_loss` 的归一化（§5.2）现在是唯一仍然成立的假设。** 剩余按预期收益排序：

1. **补上归一化后重训一轮**（§5.2）。一行改动，4×H200 约 18 小时，可验证或排除：

   ```python
   mask_empty_loss = mask_empty_loss / num_elements
   ```

   判据是 `mask_dice_loss` 的轨迹能否贴近官方（epoch 5 时 0.18 而非 0.41）。
2. 给 `build_answer_only_labels` 的两条问题分支加计数日志（§5.6）。当前配置余量只有
   112 token，改动数据或分辨率就会静默失效。
3. 性能优化：SAM2 解码批量化、loss 移到 256×256 计算。当前 58 s/step 中相当一部分开销
   在这两处。
4. 若 §5.2 也不能解释差距，再与作者核对训练配置 —— 我们手上只有 checkpoint 和
   `trainer_state.json`，没有他们实际使用的训练脚本，无法排除超参或数据处理上的其它差异。
