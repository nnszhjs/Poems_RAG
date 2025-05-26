import chromadb
from chromadb.utils import embedding_functions
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI


# 配置参数
CONFIG = {
    "data_path": "tang_poems.csv",
    "db_path": "./poem_db",
    "collection_name": "tang_poems",
    "embedding_model": "BAAI/bge-small-zh-v1.5",
    "batch_size": 4096,
    "check_batch_size": 999,
    "hnsw_space": "cosine",
    "llm_config": {
        "provider": "openai",  # 可选: "openai", "local"
        "model_name": "",
        "api_key": "",  
        "base_url": "",  
        "local_model_path": None,  # 对于本地模型
        "temperature": 0.7,
        "max_tokens": 1000
    },
    "retrieval_config": {
        "top_k": 3,  # 检索最相关的top_k文本块
        "score_threshold": 0.5  # 相关性分数阈值
    }
}

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 保留原有函数: clean_text_with_punctuation, load_and_preprocess_data, 
# initialize_chroma_collection, batch_process_data, main

class LLMIntegration:
    """LLM集成类，支持多种LLM提供者"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_provider = config["provider"]
        self.client = self._initialize_llm_client()
    
    def _initialize_llm_client(self) -> Any:
        """初始化LLM客户端"""
        try:
            if self.llm_provider == "openai":
                return OpenAI(
                    api_key=self.config["api_key"],
                    base_url=self.config.get("base_url", "https://api.openai.com/v1")
                )
            elif self.llm_provider == "local":
                # 初始化本地模型（如使用transformers）
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.config["local_model_path"])
                model = AutoModelForCausalLM.from_pretrained(self.config["local_model_path"])
                return {"model": model, "tokenizer": tokenizer}
            else:
                raise ValueError(f"不支持的LLM提供者: {self.llm_provider}")
        except Exception as e:
            logger.error(f"初始化LLM客户端失败: {str(e)}")
            raise
    
    def generate_response(self, prompt: str) -> str:
        """生成LLM响应"""
        try:
            if self.llm_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.config["model_name"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config["temperature"],
                    max_tokens=self.config["max_tokens"]
                )
                return response.choices[0].message.content
            elif self.llm_provider == "local":
                inputs = self.client["tokenizer"](prompt, return_tensors="pt")
                outputs = self.client["model"].generate(
                    **inputs,
                    max_new_tokens=self.config["max_tokens"],
                    temperature=self.config["temperature"]
                )
                return self.client["tokenizer"].decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"生成LLM响应失败: {str(e)}")
            return "抱歉，生成回答时出现错误。"

class PoemRAGSystem:
    """唐诗RAG系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = LLMIntegration(config["llm_config"])
        self.collection = self._initialize_collection()
    
    def _initialize_collection(self) -> chromadb.Collection:
        """初始化ChromaDB集合"""
        try:
            client = chromadb.PersistentClient(path=self.config["db_path"])
            sbert_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config["embedding_model"]
            )
            return client.get_or_create_collection(
                name=self.config["collection_name"],
                embedding_function=sbert_ef,
                metadata={"hnsw:space": self.config["hnsw_space"]}
            )
        except Exception as e:
            logger.error(f"初始化ChromaDB集合失败: {str(e)}")
            raise
    
    def retrieve_relevant_poems(self, query: str, top_k: Optional[int] = None, author: Optional[str] = None, title: Optional[str] = None) -> List[Dict[str, Any]]:
        """检索相关唐诗"""
        try:
            # 构建查询过滤器
            where_filter = {}
            if author:
                where_filter["author"] = {"$eq": author}
            if title:
                where_filter["title"] = {"$eq": title}
            
            # 执行查询
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )
            print("检索到了{}首唐诗。".format(len(results["ids"][0])))
            
            retrieved_docs = []
            for i in range(len(results["ids"][0])):
                doc = {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": results["distances"][0][i]
                }
                retrieved_docs.append(doc)
            
            return retrieved_docs
        except Exception as e:
            logger.error(f"检索唐诗失败: {str(e)}")
            return []
    
    def construct_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """构建Prompt模板"""
        context_str = "\n\n".join([
            f"相关文本 {i+1} (相关性分数: {ctx['score']:.2f}):\n{ctx['content']}"
            for i, ctx in enumerate(contexts)
        ])
        
        prompt = f"""你是一个专业的唐诗知识助手，请根据以下检索到的唐诗相关内容回答问题。
        
用户问题: {query}

检索到的相关内容:
{context_str}

请根据以上信息回答用户的问题。要求:
1. 回答要准确、简洁，尽量引用原文
2. 如果检索到的内容与问题无关，请明确告知"根据现有资料无法回答此问题"
3. 如果问题与唐诗无关，请礼貌地指出
4. 回答请使用中文
5. 如果询问是哪首诗，请完整回答诗名和内容
6. 如果询问诗歌的情感、意境等，请结合诗句简单回答
7. 请不要用markdown格式，用自然语言即可，如果有多个结果分点回答

请开始你的回答:"""
        return prompt
    
    def rewrite_query(self, original_query: str) -> str:
        """使用LLM重写查询以改进检索效果"""
        try:
            prompt = f"""你是一个唐诗专家助手，你的任务是根据用户问题改写或扩展查询，以便更好地检索唐诗数据库。
            
原始问题: {original_query}

请根据以下要求改写或扩展查询：
1. 改写尽量在保持信息的情况下，只留下关键词，尽可能简短
2. 如果问题涉及特定诗人，确保包含诗人姓名
3. 如果问题模糊，尝试添加内容
4. 保留原始查询的关键词
5. 用中文返回改写后的查询
6. 请不要使用标点符号，只需要返回句子或短语即可

改写后的查询:"""
            
            rewritten = self.llm.generate_response(prompt)
            return rewritten.strip('"\'') or original_query  # 确保有回退
        except Exception as e:
            logger.warning(f"查询重写失败，使用原始查询: {str(e)}")
            return original_query

    
    def ask_question(self, query: str, top_k: Optional[int] = None, 
                author: Optional[str] = None, 
                title: Optional[str] = None,
                enable_query_rewrite: bool = True) -> str:
        """提问并获取回答"""
        try:
            # 0. (新增)查询重写
            final_query = self.rewrite_query(query).strip() if enable_query_rewrite else query
            logger.info(f"查询重写: '{query}' -> '{final_query}'")
            
            # 1. 检索相关唐诗
            contexts = self.retrieve_relevant_poems(final_query, top_k, author, title)
            # 注意这里用原始query
            # contexts = self.retrieve_relevant_poems(query, top_k, author, title)
            
            if not contexts:
                return "未能找到与问题相关的唐诗内容。"

            # 2. 构建Prompt
            prompt = self.construct_prompt(query, contexts)  # 注意这里用原始query
            
            # 3. 调用LLM生成回答
            response = self.llm.generate_response(prompt)
            
            return response
        except Exception as e:
            logger.error(f"提问处理失败: {str(e)}")
            return "处理问题时出现错误。"


# 示例用法
if __name__ == "__main__":
    # 初始化RAG系统
    rag_system = PoemRAGSystem(CONFIG)
    
    # 示例问题
    questions = [
        "李白写过哪些关于月亮的诗？",
        "李白",
        "杜甫的《春望》表达了什么情感？",
        "请解释'床前明月光'的意境",
        "唐代最著名的三位诗人是谁？",
        "量子力学的基本原理是什么？"  # 测试无关问题
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        answer = rag_system.ask_question(question, top_k=3, enable_query_rewrite=True)
        print(f"回答: {answer}")
