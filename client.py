from llm_query import PoemRAGSystem

from llm_query import CONFIG
    

def main():
    # 实例化PoemRAGSystem
    system = PoemRAGSystem(config=CONFIG)
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
            n_results = input(f"返回结果数(默认{CONFIG['retrieval_config']['top_k']}): ").strip()
            n_results = int(n_results) if n_results.isdigit() else CONFIG['retrieval_config']['top_k']
            
            # 执行查询
            result = system.ask_question(query, top_k=n_results, author=author, title=title)
            print(f"\n查询结果: {result}")
        except Exception as e:
            print(f"查询失败: {e}")

if __name__ == "__main__":
    main()