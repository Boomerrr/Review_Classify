# 快速开始
## step1：下载模型
预训练模型下载地址    
https://huggingface.co/google-bert/bert-base-uncased   
https://huggingface.co/FacebookAI/roberta-base

### 模型文件存放位置  
../bert  
../roberta 

## step2：安装依赖环境 
### 创建conda环境
`conda create -n Review_Classify python=3.10`
### 激活conda环境
`conda activate Review_Classify `
### 安装依赖包
`pip install -r requirements.txt `  
  
## step3：运行代码文件  
### 本地加载模型 运行程序
`python Review_Classify.py`

### 参数说明
--model_name 使用的backbone，（选项：ROBERTA_CLS  BERT_CLS）  
--seed 随机数   
--batch_size 批次数（设置为64 24G显卡会爆显存）  
--learning_rate 学习率   
--max_seq_len 字符填充长度  
--hidden_size 图卷积网络输出节点维度  

