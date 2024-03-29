# ChatGLM-6b with LoRA Fine-tuning

本项目通过LoRA (Low-Rank Adaptation) 技术对ChatGLM-6b模型进行微调，旨在提高特定任务的性能而不显著增加参数数量。适合需要针对特定领域或任务优化预训练大模型的研究者和开发者。

## 环境需求

- Python 3.x
- transformers==4.28.1
- peft==0.3.0
- deepspeed==0.9.2
- icetk
- mpi4py
- accelerate
- cpm_kernels
- sentencepiece==0.1.99
- peft=0.3.0
- torch=2.0.0 

## 安装指南

确保您已经安装了上述所需库。如果没有安装，可以通过以下命令安装：

```bash
pip install -r requirements.txt
```
