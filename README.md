# RAG 问答系统

一个基于检索增强生成（Retrieval-Augmented Generation, RAG）的简单唐诗检索系统，支持从自定义文档库中检索唐诗并结合LLM模型生成回答。

## 功能概述
- **文档加载**：支持 CSV、TXT、JSON等格式的文档。
- **文本分块与嵌入**：将文档分块并生成向量嵌入（Embeddings）。
- **检索增强生成**：根据用户问题检索相关文档片段，并结合LLM模型生成回答。
- **交互式问答**：支持命令行交互。

## 环境配置

### 1. 系统依赖
- Python 3.10+
- [Pip](https://pip.pypa.io/en/stable/)（Python 包管理器）
- [Conda](https://docs.conda.io/en/latest/)（Python 环境管理器）
- [OpenAI API Key](https://beta.openai.com/account/api-keys)（可选）

### 2. 安装 Python 依赖
运行以下命令安装所需库(建议使用 Conda 虚拟环境)，以python3.10为例：
```bash
conda create -n poems_rag python=3.10 pandas tqdm pyarrow fastparquet
conda activate poems_rag
pip install openai chromadb sentence-transformers
```

### 3. 配置选项
编辑 `build_model.py` 文件(可选)：
```python
# 配置参数
CONFIG = {
    "data_path": "./data/tang_poems.csv", # 默认文档路径
    "db_path": "./poem_db", # 文档库路径
    "collection_name": "tang_poems", # 文档库名称
    "embedding_model": "BAAI/bge-small-zh-v1.5", # 默认嵌入模型
    "batch_size": 4096, # 批处理大小，经测试接近上限
    "check_batch_size": 999, # 检查批处理大小，已经设置为最大值
    "hnsw_space": "cosine" # 空间索引算法
}
```

编辑 `llm_query.py` 文件(必须提供 API 密钥)：
```python
# 配置参数
CONFIG = {
    "data_path": "tang_poems.csv", # 默认文档路径
    "db_path": "./poem_db", # 文档库路径
    "collection_name": "tang_poems", # 文档库名称
    "embedding_model": "BAAI/bge-small-zh-v1.5", # 默认嵌入模型，需要与生成模型匹配
    "batch_size": 4096, # 批处理大小，经测试接近上限
    "check_batch_size": 999, # 检查批处理大小，已经设置为最大值
    "hnsw_space": "cosine", # 空间索引算法
    "llm_config": {
        "provider": "openai",  # 可选: "openai", "local"
        "model_name": "",  # 请替换为你的模型名称
        "api_key": "",  # 请替换为你的 API 密钥
        "base_url": "",  # 请替换为你的 API 提供商地址
        "local_model_path": None,  # 对于本地模型
        "temperature": 0.7,
        "max_tokens": 1000
    },
    "retrieval_config": {
        "top_k": 3,  # 默认检索最相关的top_k文本块
        "score_threshold": 0.5  # 相关性分数阈值
    }
}
```

## 快速开始

### 1. 运行步骤
```bash
# 克隆仓库（如果是首次使用）
git clone https://github.com/your-repo/rag-qa-system.git
cd rag-qa-system

# 安装依赖
conda create -n poems_rag python=3.10 pandas tqdm pyarrow fastparquet
conda activate poems_rag
pip install openai chromadb sentence-transformers

# 构建文档库
python build_model.py

# 运行程序
python client.py
```

### 2. 交互示例
```python
>>> 系统提示："=== 唐诗检索系统 ===
支持语义搜索和过滤条件，输入'q'退出"
>>> 请输入查询内容: "李白"
>>> 按作者过滤(可选，直接回车跳过): 
>>> 按标题过滤(可选，直接回车跳过): 
>>> 返回结果数(默认3): 
>>> 系统提示："检索到了3首唐诗。"
>>> 查询结果：
"根据检索到的内容，关于李白的信息如下：

1. 李白的作品《古风 五十二》中写道：“青春流惊湍，朱明骤回薄。不忍看秋蓬，飘扬竟何托。”这首诗表达了时光流逝和漂泊无依的感慨。

2. 李白的另一首诗《白纻辞三首 二》中提到：“月寒江清夜沈沈，美人一笑千黄金。愿作天池双鸳鸯，一朝飞去青云上。”这首诗描绘了美人的风姿和诗人对美好境界的向往。

此外，杜甫在《春日忆李白》中评价李白：“白也诗无敌，飘然思不群。清新庾开府，俊逸鲍参军。”这体现了李白诗歌的卓越成就和独特风格。"
```

## 文件结构
```
.
├── data/                  # 存放数据的目录
├── poem_db/               # 存放文档库的目录
│   ├── */                 # 文档库子目录
│   └── chroma.sqlite3     # 文档库函数
├── build_model.py         # 构建文档库代码
├── get_data.py            # 下载数据集代码
├── client.py              # 交互式问答代码
├── llm_query.py           # 语言模型查询代码
├── README.md              # 说明文档
└── test_model.py          # 测试模型代码
```



## 常见问题
1. **建立文档库阶段python库报错？**  
   使用推荐的虚拟环境设置，并确保安装了所有依赖库。

2. **检索到结果但返回结果为空？**  
   使用参数量更大的LLM模型。

3. **遇到 API 限速错误？**  
   尝试降低请求频率或使用本地模型（如 Llama.cpp）。

3. **交互式查询速度慢？**  
   尝试使用本地模型（如 Llama.cpp）。不要使用深度思考模型。
   


## 许可证
本项目采用 [MIT License](LICENSE)。

