from langchain_community.vectorstores import FAISS
from typing import Optional, Dict, Any


class RetrievalAgent:
    def __init__(self, vectorstore, evaluation_agent, top_k=5):
        self.vectorstore = vectorstore
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        self.evaluation_agent = evaluation_agent

    def retrieve(self, query: str, reference: Optional[str] = None) -> Dict[str, Any]:
        """
        Single-round retrieval logic, controlled by the upper layer router whether to retry:
        Returns:
          {
            "docs": [...],
            "context_precision": float,
            "context_recall": float
          }
        """
        # 1) Vector retrieval
        docs = self.retriever.invoke(query)

        if not docs:
            print("⚠️ No documents retrieved")
            return {
                "docs": [],
                "context_precision": 0.0,
                "context_recall": 0.0
            }

        # 2) Evaluate retrieval effectiveness
        eval_result = self.evaluation_agent.evaluate_retrieval(
            user_query=query,
            retrieved_docs=docs,
            reference=reference
        )

        context_precision = float(eval_result.get("context_precision", 0.0))
        context_recall = float(eval_result.get("context_recall", 0.0))

        print(f"🔎 [Retrieval] Precision={context_precision:.2f}, Recall={context_recall:.2f}")

        return {
            "docs": docs,
            "context_precision": context_precision,
            "context_recall": context_recall
        }
