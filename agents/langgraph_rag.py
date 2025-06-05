from typing import Dict, List, Any, TypedDict, Optional
from langchain_core.documents import Document
import langgraph.graph as lg
from langgraph.graph import END, StateGraph
from functools import lru_cache
import time

from agents.reasoning_agent import ReasoningAgent
from agents.retrieval_agent import RetrievalAgent
from agents.evaluation_agent import EvaluationAgent
from agents.generation_agent import GenerationAgent

import numpy as np  # 🔧 add this for extract_scalar
from decimal import Decimal

# 🔧 used to extract numbers from various results (list/np/float)
def extract_scalar(val):
    if isinstance(val, list) and val:
        return float(val[0])
    elif isinstance(val, (int, float, np.floating, np.generic, Decimal)):
        return float(val)
    else:
        return float(val) if val is not None else 0.0


# 1) State type can keep or add more fields for multiple indicators
class AgentState(TypedDict):
    question: str
    refined_query: str
    docs: List[Document]
    answer: str
    faithfulness_score: float
    response_relevancy: float
    noise_sensitivity: float
    semantic_f1_score: float  # ✅ add this missing field
    context_recall: float
    context_precision: float
    attempts: int
    next_step: str
    messages: List[Dict[str, Any]]
    error: Optional[str]
    start_time: float
    metrics: Dict[str, Any]
    requery_count: int
    regenerate_count: int
    max_attempts: int
    max_regenerates: int
    max_requeries: int
    reference: Optional[str]  # ✅ add field: for passing Ground Truth


def create_rag_graph(
    retrieval_agent: RetrievalAgent,
    reasoning_agent: ReasoningAgent,
    generation_agent: GenerationAgent,
    evaluation_agent: EvaluationAgent
):
    # Modification1: Ensure that the file resource opened after the query is closed
    def cached_retrieve(query: str, reference: str = None):
        try:
            result = retrieval_agent.retrieve(query, reference=reference)
            return result
        except Exception as e:
            print(f"检索错误: {e}")
            return []

    # Use a limited-size LRU cache to prevent cache from becoming too large and consuming resources
    retrieve_cache = {}
    max_cache_size = 20  # Reduce cache size, reduce memory usage

    # Modification2: Implement custom cache to ensure resource management
    def cached_retrieve_with_resource_mgmt(query: str, reference: str = None):
        cache_key = (query, reference)

        if cache_key in retrieve_cache:
            return retrieve_cache[cache_key]

        result = cached_retrieve(query, reference=reference)

        # If the cache is full, remove the oldest item
        if len(retrieve_cache) >= max_cache_size:
            # Remove the first key
            oldest_key = next(iter(retrieve_cache))
            del retrieve_cache[oldest_key]

        retrieve_cache[cache_key] = result
        return result

    # -----------------------------
    # (1) Query optimization node
    # -----------------------------
    def query_optimizer(state: AgentState) -> AgentState:
        """Use ReasoningAgent to optimize user queries (only return refined_query)"""
        try:
            print(f"\n🧠 优化查询: {state['question']}")
            start = time.time()

            reasoning_result = reasoning_agent.plan(
                user_question=state["question"],
                retrieved_docs=[] # Here empty; if you need to pass docs, you can also do so
            )
            # The original was reasoning_result["response"], now changed to reasoning_result["refined_query"]
            refined_query = reasoning_result["refined_query"]

            duration = time.time() - start
            return {
                **state,
                "refined_query": refined_query,
                "metrics": {**state["metrics"], "query_optimization_time": duration},
                "messages": state["messages"] + [{
                    "role": "system",
                    "content": f"Optimized query: {refined_query}"
                }]
            }
        except Exception as e:
            print(f"⚠️ Query optimization error: {e}")
            return {
                **state,
                "refined_query": state["question"],
                "error": f"Query optimization failed: {str(e)}",
                "messages": state["messages"] + [{
                    "role": "system",
                    "content": f"Query optimization failed: {str(e)}"
                }]
            }

    # -----------------------------
    # (2) Retrieval node
    # -----------------------------
    def retriever(state: AgentState) -> AgentState:
        """Retrieve based on refined_query, optionally perform evaluate_retrieval"""
        try:
            query = state["refined_query"]
            reference = state.get("reference", None)  # ✅ 新增
            print(f"\n📚 Retrieving based on optimized query: {query}")

            start = time.time()
            # Modification3: Use resource-safe caching retrieval function
            ret_result = cached_retrieve_with_resource_mgmt(query, reference=reference)
            docs = ret_result["docs"]
            context_precision = extract_scalar(ret_result.get("context_precision"))
            context_recall = extract_scalar(ret_result.get("context_recall"))
            duration = time.time() - start

            if not docs:
                print("⚠️ No relevant documents found")
                return {
                    **state,
                    "docs": [],
                    "answer": "Sorry, I couldn't find any information related to your question.",
                    "faithfulness_score": 0.0,
                    "next_step": "end",
                    "metrics": {
                        **state["metrics"],
                        "retrieval_time": duration,
                        "doc_count": 0
                    },
                    "messages": state["messages"] + [{
                        "role": "system",
                        "content": "No relevant documents found"
                    }]
                }
            # ✅ Limit context length (to prevent downstream token explosion)
            def trim_doc_text(doc):
                return doc.page_content[:3000] if len(doc.page_content) > 3000 else doc.page_content

            docs = [
                Document(page_content=trim_doc_text(doc), metadata=doc.metadata)
                for doc in docs
            ]


            # If you want to record retrieval quality (Recall/Precision), you can call:
            ret_eval = evaluation_agent.evaluate_retrieval(query, docs, reference=reference)
            context_precision = extract_scalar(ret_eval.get("context_precision", 0.0))
            context_recall = extract_scalar(ret_eval.get("context_recall", 0.0))

            print(f"🎯 Retrieval Metrics: Precision={context_precision:.2f}, Recall={context_recall:.2f}")


            return {
                **state,
                "docs": docs,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "metrics": {
                    **state["metrics"],
                    "retrieval_time": duration,
                    "doc_count": len(docs),
                    "context_precision": context_precision,
                    "context_recall": context_recall
                },
                "messages": state["messages"] + [{
                    "role": "system",
                    "content": f"Retrieved {len(docs)} documents"
                }]
            }
        except Exception as e:
            print(f"⚠️ Retrieval error: {e}")
            return {
                **state,
                "docs": [],
                "error": f"Retrieval failed: {str(e)}",
                "next_step": "end",
                "messages": state["messages"] + [{
                    "role": "system",
                    "content": f"Retrieval failed: {str(e)}"
                }]
            }

    # -----------------------------
    # (3) Generate answer node
    # -----------------------------
    def generator(state: AgentState) -> AgentState:
        """Call GenerationAgent to generate an answer, do not evaluate here, let the evaluator node do it"""
        try:
            query = state["refined_query"]
            docs = state["docs"]
            reference = state.get("reference", None)  # ✅ ground truth
            print(f"\n✍️ Generate answer...")
            print(f"🧪 Reference in generator: {reference}")  # ✅ Debug information

            start = time.time()
            # Generate answer (internal will do multiple retries/Prompt optimization)
            answer_result = generation_agent.answer(
                question=query,
                docs=docs,
                evaluation_agent=evaluation_agent,
                ground_truth=reference  # ✅ Key addition
            )
            duration = time.time() - start

            relevancy = (
                answer_result.get("response_relevancy") or
                answer_result.get("answer_relevancy") or
                0.0
            )
            # ✅✅✅ Insert debug output: check if semantic_f1 exists & passed correctly
            print("🧪 Final answer_result:", answer_result)
            print("🧪 semantic_f1_score in answer_result:", answer_result.get("semantic_f1_score"))
            print("🧪 semantic_f1 (alt key):", answer_result.get("semantic_f1"))

            return {
                **state,
                "answer": answer_result["answer"],
                "faithfulness_score": answer_result.get("faithfulness_score", 0.0),
                "response_relevancy": extract_scalar(relevancy),
                "noise_sensitivity": answer_result.get("noise_sensitivity", 1.0),
                # ✅ If there are two possible keys, make a compatible处理：
                "semantic_f1_score": (
                answer_result.get("semantic_f1_score", 0.0)
                if answer_result.get("semantic_f1_score") is not None
                else answer_result.get("semantic_f1", 0.0)
            ),
                "eval_result": answer_result.get("cached_eval_result", None),  # ✅ Pass evaluation details
                "metrics": {
                    **state["metrics"],
                    "generation_time": duration,
                    "cached_eval_result": answer_result.get("cached_eval_result", None)  # ✅ Add cached evaluation result

                },
                "messages": state["messages"] + [{
                    "role": "assistant",
                    "content": answer_result["answer"]
                }]

            }
        except Exception as e:
            print(f"⚠️ Generating answers incorrectly: {e}")
            return {
                **state,
                "answer": "Sorry, I encountered an issue while generating an answer.",
                "error": f"Failed to generate an answer: {str(e)}",
                "next_step": "end",
                "messages": state["messages"] + [{
                    "role": "system",
                    "content": f"Failed to generate an answer: {str(e)}"
                }]
            }

    # -----------------------------
    # (4) Evaluator node
    # -----------------------------
    def evaluator(state: AgentState) -> AgentState:
        print(f"⚡ Evaluator skipped (由 Generator 已评估)")
        return state

    # -----------------------------
    # (5) Router node
    # -----------------------------
    def router(state: AgentState) -> AgentState:
        """
        Based on multiple indicators, determine the next step:
        1) If retrieval metrics are low, prioritize re-query
        2) If retrieval is acceptable but answer quality is insufficient, re-generate
        3) If answer quality is good or attempts limit reached, end
        """

        attempts = state["attempts"]
        requery_count = state.get("requery_count", 0)
        regenerate_count = state.get("regenerate_count", 0)
        max_attempts = state.get("max_attempts", 5)
        max_requeries = state.get("max_requeries", 2)
        max_regenerates = state.get("max_regenerates", 2)



        # ---- Unpack各指标 ----
        # Retrieval metrics
        context_recall = extract_scalar(state.get("context_recall", 0.0))
        context_precision = extract_scalar(state.get("context_precision", 0.0))
        # Generation metrics
        faithfulness = extract_scalar(state.get("faithfulness_score", 0.0))
        relevancy = extract_scalar(state.get("response_relevancy", 0.0))
        noise = extract_scalar(state.get("noise_sensitivity", 1.0))
        semantic_f1_score = extract_scalar(state.get("semantic_f1_score", 0.0))
        # Other states

        error = state.get("error", None)


        print(f"\n🔄 路由决策: attempts={attempts}, requery={requery_count}, regenerate={regenerate_count}")
        print(f"    → recall={context_recall:.2f}, precision={context_precision:.2f}")
        print(f"    → faith={faithfulness:.2f}, relevancy={relevancy:.2f}, noise={noise:.2f}, semantic_f1_score={semantic_f1_score:.2f}")


        # If an error occurs, end directly
        if error:
            return {**state, "next_step": "end", "attempts": attempts + 1}

        # --- Step 0: Generation priority override ---
        if faithfulness >= 0.7 and relevancy >= 0.7 and noise <= 0.4 and semantic_f1_score >= 0.7:
            print("✅ Generation quality is high, skipping retrieval quality check. Proceed to end.")
            return {**state, "next_step": "end", "attempts": attempts + 1}

        # --- Step 1: Check if retrieval quality is significantly insufficient (Recall/Precision too low)
        #     If retrieval is significantly insufficient, it is more likely to need re-query
        if context_recall < 0.5 or context_precision < 0.3:
            if requery_count < max_requeries:
                return {
                    **state,
                    "next_step": "requery",
                    "requery_count": requery_count + 1,
                    "attempts": attempts + 1
                }
            else:
                # When the requery limit is reached, do not force regenerate, end directly
                print("⚠️ Retrieval attempts exhausted. Ending.")
                return {**state, "next_step": "end", "attempts": attempts + 1}

        # --- Step 2: If semantic correctness is high and retrieval is good, even if faithfulness is low, allow early termination
        if semantic_f1_score >= 0.8 and context_recall >= 0.7:
            print("🎯 High semanticF1 and good retrieval, accept the answer.")
            return {**state, "next_step": "end", "attempts": attempts + 1}

        # --- Answer quality is poor ---
        if faithfulness < 0.6 or relevancy < 0.5 or noise > 0.4 or semantic_f1_score < 0.7:
            if regenerate_count < max_regenerates:
                return {
                    **state,
                    "next_step": "regenerate",
                    "regenerate_count": regenerate_count + 1,
                    "attempts": attempts + 1
                }
            else:
                print("⚠️ Generation attempts exhausted. Ending.")
                return {**state, "next_step": "end", "attempts": attempts + 1}

        # --- Answer is sufficient ---
        return {**state, "next_step": "end", "attempts": attempts + 1}


    # -----------------------------
    # (6) requery_optimizer
    # -----------------------------
    def requery_optimizer(state: AgentState) -> AgentState:
        """Call ReasoningAgent.plan again based on retrieved documents"""
        try:
            print(f"\n🔄 Re-optimizing query...")
            start = time.time()

            # This time pass docs to plan() so the Agent can optimize the query
            reasoning_result = reasoning_agent.plan(
                user_question=state["question"],
                retrieved_docs=state["docs"]
            )
            refined_query = reasoning_result["refined_query"]
            duration = time.time() - start

            return {
                **state,
                "refined_query": refined_query,
                "metrics": {
                    **state["metrics"],
                    "requery_optimization_time": duration
                },
                "messages": state["messages"] + [{
                    "role": "system",
                    "content": f"Re-optimized query: {refined_query}"
                }]
            }
        except Exception as e:
            print(f"⚠️ Re-optimizing query error: {e}")
            return {
                **state,
                "error": f"Re-optimizing query failed: {str(e)}",
                "next_step": "end",
                "messages": state["messages"] + [{
                    "role": "system",
                    "content": f"Re-optimizing query failed: {str(e)}"
                }]
            }

    # -----------------------------
    # (7) finalizer
    # -----------------------------
    def finalizer(state: AgentState) -> AgentState:
        """Complete the process and calculate the total time."""
        total_time = time.time() - state["start_time"]
        print(f"\n⏱️ Total processing time: {total_time:.2f} seconds")

        # Modification4: Clear cache, release resources
        retrieve_cache.clear()

        return {
            **state,
            "metrics": {**state["metrics"], "total_time": total_time}
        }

    # Create StateGraph
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("query_optimizer", query_optimizer)
    workflow.add_node("retriever", retriever)
    workflow.add_node("generator", generator)

    workflow.add_node("router", router)
    workflow.add_node("requery_optimizer", requery_optimizer)
    workflow.add_node("finalizer", finalizer)

    # Set node order
    workflow.add_edge("query_optimizer", "retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", "router")
    workflow.add_conditional_edges(
        "router",
        lambda st: st["next_step"],
        {
            "end": "finalizer",
            "regenerate": "generator",
            "requery": "requery_optimizer"
        }
    )
    workflow.add_edge("requery_optimizer", "retriever")
    workflow.add_edge("finalizer", END)

    # Set entry point
    workflow.set_entry_point("query_optimizer")
    return workflow.compile()

# -----------------------------
# Run RAG process
# -----------------------------

# ✅ Initialize at the beginning of run_rag_pipeline
#evaluation_agent = EvaluationAgent(debug_mode=False)  #  debug mode

def run_rag_pipeline(
    question: str,
    retrieval_agent: RetrievalAgent,
    reasoning_agent: ReasoningAgent,
    generation_agent: GenerationAgent,
    evaluation_agent: EvaluationAgent,


    **kwargs
) -> Dict[str, Any]:

    # Optional parameter: reference as ground truth, for evaluation phase
    reference = kwargs.get("reference", None)

    graph = create_rag_graph(
        retrieval_agent,
        reasoning_agent,
        generation_agent,
        evaluation_agent
    )

    if kwargs.get("visualize", False):
        try:
            from IPython.display import display
            display(graph.get_graph().draw_mermaid_png())
        except Exception as e:
            print(f"无法生成可视化: {e}")

    initial_state = {
        "question": question,
        "refined_query": "",
        "docs": [],
        "answer": "",
        "faithfulness_score": 0.0,
        "response_relevancy": 0.0,
        "noise_sensitivity": 0.0,
        "semantic_f1_score": 0.0,
        "context_recall": 0.0,
        "context_precision": 0.0,
        "attempts": 0,
        "requery_count": 0,
        "regenerate_count": 0,
        "max_attempts": 5,
        "max_regenerates": 2,
        "max_requeries": 2,
        "next_step": "",
        "error": None,
        "start_time": time.time(),
        "metrics": {},
        "messages": [{"role": "user", "content": question}],
        "reference": reference
    }
    print("📎 Ground Truth in initial_state:", initial_state["reference"])

    result = graph.invoke(initial_state)

    print(f"\n✅ 最终答案: {result['answer']}")
    print(f"📊 Faithfulness: {result['faithfulness_score']:.2f}, "
          f"Relevancy: {result['response_relevancy']:.2f}, "
          f"Noise: {result['noise_sensitivity']:.2f}")

    # ✅ If generation stage produces semantic_f1, print it
    if "semantic_f1_score" in result:
        print(f"🎯 Semantic F1: {result['semantic_f1_score']:.2f}")

    # Output performance metrics
    print("\n📈 Performance metrics:")
    for metric, value in result["metrics"].items():
        if isinstance(value, float):
            print(f"  - {metric}: {value:.2f}")
        else:
            print(f"  - {metric}: {value}")

    return result
