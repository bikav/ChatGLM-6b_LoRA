# ChatGLM-6b with LoRA Fine-tuning

本项目通过LoRA (Low-Rank Adaptation) 技术对ChatGLM-6b模型进行微调，旨在提高特定任务的性能而不显著增加参数数量。微调使用睡眠理疗师模拟对话数据集。

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

## 使用说明

### 1、数据准备

**放置数据集**：首先，将原始数据集放置在项目目录下的`data/preprocess`文件夹中。确保你的文件结构如下所示：

```bash
your-project/
├── data/
    └── preprocess/
```

**预处理数据集**：使用下列脚本对数据集进行预处理。这些步骤将帮助你分割数据集、修改标签、调整数据内容，并转换数据格式。

 **· 分割数据集**

```python
python split_dataset.py
```

将数据集放在`./data/preprocess`中，使用split_dataset.py、modify_labels.py、modify_data_content.py和data_conversion_format.py进行预处理，并将预处理后的数据集保存在`./data`中。


## 引用
> https://github.com/taishan1994/ChatGLM-LoRA-Tuning
> 
> https://github.com/hkust-nlp/ceval
> 
> https://juejin.cn/post/7249626243125231677
