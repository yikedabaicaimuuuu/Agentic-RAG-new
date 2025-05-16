from langchain_community.vectorstores import FAISS
from typing import Optional

class RetrievalAgent:
    def __init__(self, vectorstore, evaluation_agent, top_k=5, max_retries=2):
        """
        :param vectorstore: 已加载好 FAISS VectorStore 的实例
        :param evaluation_agent: 用于评估的 EvaluationAgent
        :param top_k: 每次检索返回的文档数量
        :param max_retries: 最大检索重试次数（当检测到检索质量不足时）
        """
        self.vectorstore = vectorstore
        # 这里通过 as_retriever() + k 设置检索
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        self.max_retries = max_retries
        self.evaluation_agent = evaluation_agent

    def retrieve(self, query: str, reference: Optional[str] = None):
        """
        新版检索逻辑:
          1) 调用 FAISS 进行向量检索
          2) 使用 evaluation_agent.evaluate_retrieval(...) 评估检索结果的 Recall / Precision
          3) 若不满足阈值, 重试(最多 self.max_retries 次)
          4) 返回最终 docs
        """
        attempts = 0
        docs = []

        while attempts < self.max_retries:
            # 1) 使用 FAISS Vectorstore 进行 Embedding 语义检索
            docs = self.retriever.invoke(query)

            if not docs:
                print("⚠️ 未检索到任何文档，直接退出。")
                return []

            # 2) 使用 EvaluationAgent 专门的检索评估（只算 Recall / Precision）
            eval_result = self.evaluation_agent.evaluate_retrieval(
                user_query=query,
                retrieved_docs=docs,
                reference=reference  # ✅ 传入引用内容以计算 Recall

                # 假设我们可以传入一个 reference_contexts 或 None
                # 若没有“标准文档”可比对, 可以视情况使用 LLMContextPrecisionWithoutReference 等
            )

            # 3) 从评估结果中解析 context_recall & context_precision
            context_precision = float(eval_result.get("context_precision", 0))
            context_recall = float(eval_result.get("context_recall", 0))

            print(f"🔎 [Retrieval Attempt #{attempts+1}] Precision={context_precision:.2f}, Recall={context_recall:.2f}")

            # 4) 判断是否达标
            if context_precision < 0.5 or context_recall < 0.5:
                print("🔍 Retrieval insufficient. Retrying...")
                attempts += 1
            else:
                print("✅ Retrieval is sufficient.")
                break  # 检索质量足够时停止重试

        return docs
