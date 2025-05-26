import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import re
import os
from tqdm import tqdm
import logging
from typing import List, Dict, Any
from get_data import get_poems_data

# 配置参数
CONFIG = {
    "data_path": "./data/tang_poems.csv",
    "db_path": "./poem_db",
    "collection_name": "tang_poems",
    "embedding_model": "BAAI/bge-small-zh-v1.5",
    "batch_size": 4096,
    "check_batch_size": 999,
    "hnsw_space": "cosine"
}

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_text_with_punctuation(text: str) -> str:
    """清洗文本但保留标点符号"""
    if not isinstance(text, str):
        return ""
    # 只移除特殊字符和多余空格，保留中文标点
    text = re.sub(r'[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', '', text)
    # 合并连续空格但保留其他空白字符
    return re.sub(r'\s+', ' ', text).strip()

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """加载和预处理数据（保留标点符号）"""
    try:
        df = pd.read_csv(file_path)
        df.dropna(inplace=True)
        
        # 应用文本清洗（保留标点）
        text_columns = ['paragraphs', 'title', 'author']
        for col in text_columns:
            df[col] = df[col].apply(clean_text_with_punctuation)
        
        # 创建完整文本字段（保留原始格式）
        df['full_text'] = df.apply(
            lambda x: f"《{x['title']}》 {x['author']} {x['paragraphs']}", 
            axis=1
        )
        return df
    except Exception as e:
        logger.error(f"加载和预处理数据失败: {str(e)}")
        raise

def initialize_chroma_collection() -> chromadb.Collection:
    """初始化ChromaDB集合"""
    try:
        client = chromadb.PersistentClient(path=CONFIG["db_path"])
        
        # 使用自定义嵌入函数
        sbert_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=CONFIG["embedding_model"]
        )
        
        collection = client.get_or_create_collection(
            name=CONFIG["collection_name"],
            embedding_function=sbert_ef,
            metadata={"hnsw:space": CONFIG["hnsw_space"]}
        )
        return collection
    except Exception as e:
        logger.error(f"初始化ChromaDB集合失败: {str(e)}")
        raise

def batch_process_data(
    collection: chromadb.Collection,
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str]
) -> None:
    """批量处理数据到ChromaDB"""
    try:
        # 获取现有ID集合
        existing_ids = set()
        logger.info("检查已存在的文档ID...")
        
        for i in tqdm(range(0, len(ids), CONFIG["check_batch_size"])):
            end_idx = min(i + CONFIG["check_batch_size"], len(ids))
            batch_ids = ids[i:end_idx]
            
            try:
                batch_existing = collection.get(ids=batch_ids)['ids']
                existing_ids.update(batch_existing)
            except Exception as e:
                logger.warning(f"获取ID批次 {i}-{end_idx} 失败: {str(e)}")
                continue
        
        # 批量添加新数据
        logger.info("开始批量添加新文档...")
        added_count = 0
        
        for i in tqdm(range(0, len(documents), CONFIG["batch_size"])):
            end_idx = min(i + CONFIG["batch_size"], len(documents))
            batch_ids = ids[i:end_idx]
            
            # 筛选新文档
            new_data = [
                (doc, meta, doc_id)
                for doc, meta, doc_id in zip(
                    documents[i:end_idx],
                    metadatas[i:end_idx],
                    batch_ids
                )
                if doc_id not in existing_ids
            ]
            
            if not new_data:
                continue
                
            new_docs, new_metas, new_ids = zip(*new_data)
            
            try:
                collection.add(
                    documents=list(new_docs),
                    metadatas=list(new_metas),
                    ids=list(new_ids)
                )
                added_count += len(new_ids)
            except Exception as e:
                logger.error(f"添加批次 {i}-{end_idx} 失败: {str(e)}")
                continue
        
        logger.info(f"处理完成! 共添加 {added_count} 个新文档")
        
    except Exception as e:
        logger.error(f"批量处理数据失败: {str(e)}")
        raise

def main():
    """主处理流程"""
    if os.path.exists(CONFIG["db_path"]):
        choice = input("ChromaDB数据库已存在，是否覆盖？(y/n)\n选择：")
        if choice.lower() == "y":
            os.system(f"rm -rf {CONFIG['db_path']}")
        else:
            print("未覆盖，build程序退出!")
            return
    try:
        try:
            # 1. 加载和预处理数据（保留标点）
            logger.info("开始加载和预处理数据（保留标点符号）...")
            df = load_and_preprocess_data(CONFIG["data_path"])
        except Exception as e:
            logger.error(f"加载和预处理数据失败: {str(e)}, 尝试重新下载数据...")
            # 检查data文件夹是否存在
            if not os.path.exists("./data"):
                os.mkdir("./data")
            # 下载数据并保存到data文件夹
            df = get_poems_data(save_path=CONFIG["data_path"])
            df["full_text"] = df.apply(
                lambda x: f"{x['title']} {x['author']} {x['paragraphs']}", 
                axis=1
            )
            logger.info("数据下载完成!")
            df.to_csv(CONFIG["data_path"], index=False)
            logger.info("数据保存成功!")
        # 2. 准备数据
        documents = df['full_text'].tolist()
        metadatas = [
            {
                "title": row['title'],
                "author": row['author'],
                "content": row['paragraphs'],
                "full_text": row['full_text']  # 将完整文本也存入metadata
            }
            for _, row in df.iterrows()
        ]
        ids = [str(idx) for idx in range(len(df))]
        
        # 3. 初始化ChromaDB集合
        logger.info("初始化ChromaDB集合...")
        collection = initialize_chroma_collection()
        
        # 4. 批量处理数据
        batch_process_data(collection, documents, metadatas, ids)
        
        logger.info("数据处理完成!")
        
    except Exception as e:
        logger.error(f"主流程执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()
