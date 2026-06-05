<div align="center">

![logo](./images/logo_miniwin.png)

</div>

<div align="center">

[中文](./README.md) | English

</div>

---

# MiniWin

**MiniWin** is a small language model project built on top of [MiniMind](https://github.com/jingyaogong/minimind). Its core changes come from the paper [Periodic RoPE for Infinite Context LLMs](https://arxiv.org/abs/2605.27980) (P-RoPE).

When sequence length exceeds the pre-trained range of positional encodings, standard RoPE suffers from **position exhaustion**, and long-context performance degrades. MiniWin addresses this with **Periodic RoPE (P-RoPE)** combined with **Sliding Window Attention (SWA)** for local dependencies, plus **No Positional Encoding (NoPE)** global attention layers for unbounded cross-sequence interaction—supporting theoretically infinite context windows.

## Key Features

- **P-RoPE**: Periodic positional encoding to mitigate RoPE extrapolation failure
- **SWA + NoPE hybrid layers**: relative positions within local windows; position-free global interaction across the full sequence
- MiniMind training pipeline retained: pretrain, SFT, LoRA, DPO, RL, etc.
- Native PyTorch implementation; single-GPU and multi-GPU training supported

## Quick Start

### Environment

```bash
pip install -r requirements.txt
```

### Dataset

Use the MiniMind open-source dataset. Download files into `./dataset/`:

- [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)
- [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)

For a quick reproduction: `pretrain_hq.jsonl` + `sft_mini_512.jsonl`.

### Training

From the `trainer` directory:

```bash
# Pretrain
python train_pretrain.py

# Supervised fine-tuning
python train_full_sft.py
```

Multi-GPU:

```bash
torchrun --nproc_per_node N train_pretrain.py
torchrun --nproc_per_node N train_full_sft.py
```

Resume from checkpoint: add `--from_resume 1`.

Other scripts (LoRA, DPO, PPO/GRPO, etc.) are in `trainer/`.

### Pre-trained Model

Pre-trained SFT checkpoint (768 dim):

- [full_sft_768.pth](https://github.com/Cominder/miniwin/releases/download/models/full_sft_768.pth)

Download into `./out/`:

```bash
mkdir -p out
wget -O out/full_sft_768.pth https://github.com/Cominder/miniwin/releases/download/models/full_sft_768.pth
```

### Inference

```bash
python eval_llm.py --weight full_sft
```

## Project Structure

```
miniwin/
├── model/          # MiniWin model (P-RoPE, SWA, NoPE)
├── dataset/        # Dataset loading
├── trainer/        # Training scripts
└── scripts/        # Utility scripts
```

## Citation

If you find this work helpful, please cite:

```bibtex
@article{huo2026periodic,
  title={Periodic RoPE for Infinite Context LLMs},
  author={Huo, Simin},
  journal={arXiv preprint arXiv:2605.27980},
  year={2026}
}
```

MiniMind base project:

```bibtex
@misc{minimind,
  title={MiniMind: Train a Tiny LLM from scratch},
  author={Jingyao Gong},
  year={2024},
  howpublished={https://github.com/jingyaogong/minimind}
}
```

## License

This repository is licensed under the [Apache-2.0 License](LICENSE).
