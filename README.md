<div align="center">

![logo](./images/logo_miniwin.png)

</div>

<div align="center">

中文 | [English](./README_en.md)

</div>

---

# MiniWin

**MiniWin** 是在 [MiniMind](https://github.com/jingyaogong/minimind) 基础上修改得到的小型语言模型项目，核心改动来自论文 [Periodic RoPE for Infinite Context LLMs](https://arxiv.org/abs/2605.27980)（P-RoPE），受[Iwin Transformer](https://github.com/Cominder/Iwin-Transformer)启发不断改进得到。

当序列长度超过预训练位置编码范围时，标准 RoPE 会出现**位置编码耗尽**（position exhaustion），长上下文性能随之下降。MiniWin 通过 **Periodic RoPE（P-RoPE）** 与 **Sliding Window Attention（SWA）** 处理局部依赖，并配合 **No Positional Encoding（NoPE）** 的全局注意力层，在理论上支持更长的上下文窗口。

## 主要特性

- **P-RoPE**：周期性位置编码，缓解 RoPE 外推失效问题
- **SWA + NoPE 混合层**：局部窗口内建模相对位置，全局层实现无位置约束的跨序列交互
- 保留 MiniMind 的训练流程：预训练、SFT、LoRA、DPO、RL 等
- 原生 PyTorch 实现，支持单卡 / 多卡训练

## 快速开始

### 环境

```bash
pip install -r requirements.txt
```

### 数据集

使用 MiniMind 开源数据集，下载后放到 `./dataset/` 目录：

- [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)
- [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)

快速复现推荐：`pretrain_hq.jsonl` + `sft_mini_512.jsonl`。

### 训练

进入 `trainer` 目录：

```bash
# 预训练
python train_pretrain.py

# 监督微调
python train_full_sft.py
```

多卡训练：

```bash
torchrun --nproc_per_node N train_pretrain.py
torchrun --nproc_per_node N train_full_sft.py
```

断点续训：添加 `--from_resume 1`。

其他训练脚本（LoRA、DPO、PPO/GRPO 等）见 `trainer/` 目录。

### 预训练模型

已训好的 SFT 权重（768 dim）：

- [full_sft_768.pth](https://github.com/Cominder/miniwin/releases/download/models/full_sft_768.pth)

下载后放到 `./out/` 目录：

```bash
mkdir -p out
wget -O out/full_sft_768.pth https://github.com/Cominder/miniwin/releases/download/models/full_sft_768.pth
```

### 推理

```bash
python eval_llm.py --weight full_sft
```

## 项目结构

```
miniwin/
├── model/          # MiniWin 模型（P-RoPE、SWA、NoPE）
├── dataset/        # 数据集加载
├── trainer/        # 训练脚本
└── scripts/        # 工具脚本
```

## 引用

若本工作对你有帮助，请引用：

```bibtex
@article{huo2026periodic,
  title={Periodic RoPE for Infinite Context LLMs},
  author={Huo, Simin},
  journal={arXiv preprint arXiv:2605.27980},
  year={2026}
}
```

MiniMind 基础项目：

```bibtex
@misc{minimind,
  title={MiniMind: Train a Tiny LLM from scratch},
  author={Jingyao Gong},
  year={2024},
  howpublished={https://github.com/jingyaogong/minimind}
}
```

## License

本项目采用 [Apache-2.0 License](LICENSE)。
