import chromadb
from chromadb.utils import embedding_functions
import logging
from typing import List, Dict, Any, Optional

# 配置参数（与索引代码保持一致）
CONFIG = {
    "db_path": "./poem_db",
    "collection_name": "tang_poems",
    "embedding_model": "BAAI/bge-small-zh-v1.5",
    "default_results": 5
}

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_chroma_collection() -> chromadb.Collection:
    """初始化ChromaDB集合"""
    try:
        client = chromadb.PersistentClient(path=CONFIG["db_path"])
        
        # 使用与索引相同的嵌入函数
        sbert_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=CONFIG["embedding_model"]
        )
        
        collection = client.get_collection(
            name=CONFIG["collection_name"],
            embedding_function=sbert_ef
        )
        return collection
    except Exception as e:
        logger.error(f"初始化ChromaDB集合失败: {str(e)}")
        raise

def query_poems(
    query_text: str,
    collection: chromadb.Collection,
    n_results: int = CONFIG["default_results"],
    author_filter: Optional[str] = None,
    title_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    查询唐诗
    
    参数:
        query_text: 查询文本
        collection: ChromaDB集合
        n_results: 返回结果数量
        author_filter: 作者过滤条件
        title_filter: 标题过滤条件
    
    返回:
        包含诗歌信息的字典列表
    """
    try:
        # 构建查询过滤器
        where_filter = {}
        if author_filter:
            where_filter["author"] = {"$eq": author_filter}
        if title_filter:
            where_filter["title"] = {"$eq": title_filter}
        
        # 执行查询
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"]
        )
        
        # 格式化结果
        formatted_results = []
        for i in range(len(results["ids"][0])):
            poem_data = {
                "id": results["ids"][0][i],
                "score": float(results["distances"][0][i]),
                "title": results["metadatas"][0][i]["title"],
                "author": results["metadatas"][0][i]["author"],
                "content": results["metadatas"][0][i]["content"],
                "full_text": results["metadatas"][0][i].get("full_text", ""),
                "document": results["documents"][0][i]
            }
            formatted_results.append(poem_data)
        
        return formatted_results
    except Exception as e:
        logger.error(f"查询失败: {str(e)}")
        raise

def display_results(results: List[Dict[str, Any]]) -> None:
    """美观地显示查询结果"""
    if not results:
        print("未找到匹配的诗歌")
        return
    
    print("\n=== 查询结果 ===")
    for i, poem in enumerate(results, 1):
        print(f"\n结果 {i} (相似度: {1-poem['score']:.2f})")
        print(f"标题: 《{poem['title']}》")
        print(f"作者: {poem['author']}")
        print("内容:")
        print(poem['content'])
        print("-" * 40)

def interactive_search(collection: chromadb.Collection):
    """交互式查询界面"""
    print("=== 唐诗检索系统 ===")
    print("支持语义搜索和过滤条件，输入'q'退出")
    
    while True:
        try:
            # 获取用户输入
            query = input("\n请输入查询内容: ").strip()
            if query.lower() == 'q':
                break
                
            author = input("按作者过滤(可选，直接回车跳过): ").strip() or None
            title = input("按标题过滤(可选，直接回车跳过): ").strip() or None
            n_results = input(f"返回结果数(默认{CONFIG['default_results']}): ").strip()
            n_results = int(n_results) if n_results.isdigit() else CONFIG['default_results']
            
            # 执行查询
            results = query_poems(
                query_text=query,
                collection=collection,
                n_results=n_results,
                author_filter=author,
                title_filter=title
            )
            
            # 显示结果
            display_results(results)
            
        except Exception as e:
            logger.error(f"查询过程中出错: {str(e)}")
            continue

def main():
    """主查询流程"""
    try:
        # 初始化集合
        logger.info("初始化ChromaDB集合...")
        collection = initialize_chroma_collection()
        
        # 启动交互式查询
        interactive_search(collection)
        
        logger.info("退出检索系统")
        
    except Exception as e:
        logger.error(f"主流程执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()
