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

import numpy as np  # 🔧 添加此项用于 extract_scalar

# 🔧 用于提取各类结果中的数值（列表/np/float）
def extract_scalar(val):
    if isinstance(val, list) and val:
        return float(val[0])
    elif isinstance(val, (int, float, np.floating, np.generic)):
        return float(val)
    else:
        return float(val) if val is not None else 0.0


# 1) State 类型里可以保留或增加更多字段用于多指标
class AgentState(TypedDict):
    question: str
    refined_query: str
    docs: List[Document]
    answer: str
    faithfulness_score: float
    response_relevancy: float
    noise_sensitivity: float
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


def create_rag_graph(
    retrieval_agent: RetrievalAgent,
    reasoning_agent: ReasoningAgent,
    generation_agent: GenerationAgent,
    evaluation_agent: EvaluationAgent
):
    # 修改1: 确保在查询完成后关闭可能打开的文件资源
    def cached_retrieve(query: str, reference: str = None):
        try:
            result = retrieval_agent.retrieve(query, reference=reference)
            return result
        except Exception as e:
            print(f"检索错误: {e}")
            return []

    # 使用有限大小的LRU缓存，防止缓存过大导致资源耗尽
    retrieve_cache = {}
    max_cache_size = 20  # 减小缓存大小，降低内存占用

    # 修改2: 实现自定义缓存，确保资源管理
    def cached_retrieve_with_resource_mgmt(query: str, reference: str = None):
        cache_key = (query, reference)

        if cache_key in retrieve_cache:
            return retrieve_cache[cache_key]

        result = cached_retrieve(query, reference=reference)

        # 如果缓存已满，移除最早的条目
        if len(retrieve_cache) >= max_cache_size:
            # 移除第一个键
            oldest_key = next(iter(retrieve_cache))
            del retrieve_cache[oldest_key]

        retrieve_cache[cache_key] = result
        return result

    # -----------------------------
    # (1) 查询优化节点
    # -----------------------------
    def query_optimizer(state: AgentState) -> AgentState:
        """利用 ReasoningAgent 优化用户查询 (仅返回 refined_query)"""
        try:
            print(f"\n🧠 优化查询: {state['question']}")
            start = time.time()

            reasoning_result = reasoning_agent.plan(
                user_question=state["question"],
                retrieved_docs=[] # 这里空; 如果需要传 docs 也可
            )
            # 原先是 reasoning_result["response"], 现改为 reasoning_result["refined_query"]
            refined_query = reasoning_result["refined_query"]

            duration = time.time() - start
            return {
                **state,
                "refined_query": refined_query,
                "metrics": {**state["metrics"], "query_optimization_time": duration},
                "messages": state["messages"] + [{
                    "role": "system",
                    "content": f"优化后的查询: {refined_query}"
                }]
            }
        except Exception as e:
            print(f"⚠️ 查询优化出错: {e}")
            return {
                **state,
                "refined_query": state["question"],
                "error": f"查询优化失败: {str(e)}",
                "messages": state["messages"] + [{
                    "role": "system",
                    "content": f"查询优化失败: {str(e)}"
                }]
            }

    # -----------------------------
    # (2) 检索节点
    # -----------------------------
    def retriever(state: AgentState) -> AgentState:
        """基于 refined_query 进行检索，可选做 evaluate_retrieval"""
        try:
            query = state["refined_query"]
            reference = state.get("reference", None)  # ✅ 新增
            print(f"\n📚 基于优化后的查询进行检索: {query}")

            start = time.time()
            # 修改3: 使用资源安全的缓存检索函数
            docs = cached_retrieve_with_resource_mgmt(query, reference=reference)
            duration = time.time() - start

            if not docs:
                print("⚠️ 未检索到相关文档")
                return {
                    **state,
                    "docs": [],
                    "answer": "抱歉，我无法找到与您问题相关的信息。",
                    "faithfulness_score": 0.0,
                    "next_step": "end",
                    "metrics": {
                        **state["metrics"],
                        "retrieval_time": duration,
                        "doc_count": 0
                    },
                    "messages": state["messages"] + [{
                        "role": "system",
                        "content": "未检索到相关文档"
                    }]
                }
            # ✅ 限制 context 长度（防止 downstream 爆 token）
            def trim_doc_text(doc):
                return doc.page_content[:3000] if len(doc.page_content) > 3000 else doc.page_content

            docs = [
                Document(page_content=trim_doc_text(doc), metadata=doc.metadata)
                for doc in docs
            ]


            # 如果想记录检索质量(Recall/Precision)，可调用:
            ret_eval = evaluation_agent.evaluate_retrieval(query, docs, reference=reference)
            context_precision = ret_eval.get("context_precision", 0.0)
            context_recall = ret_eval.get("context_recall", 0.0)

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
                    "content": f"检索到 {len(docs)} 个文档"
                }]
            }
        except Exception as e:
            print(f"⚠️ 检索出错: {e}")
            return {
                **state,
                "docs": [],
                "error": f"检索失败: {str(e)}",
                "next_step": "end",
                "messages": state["messages"] + [{
                    "role": "system",
                    "content": f"检索失败: {str(e)}"
                }]
            }

    # -----------------------------
    # (3) 生成答案节点
    # -----------------------------
    def generator(state: AgentState) -> AgentState:
        """调用 GenerationAgent 生成回答，不在此做评估，交给 evaluator 节点做"""
        try:
            query = state["refined_query"]
            docs = state["docs"]
            print(f"\n✍️ 生成答案...")

            start = time.time()
            # 生成回答 (内部会做多次重试/Prompt优化)
            answer_result = generation_agent.answer(query, docs, evaluation_agent)
            duration = time.time() - start

            return {
                **state,
                "answer": answer_result["answer"],
                "faithfulness_score": answer_result.get("faithfulness_score", 0.0),
                "response_relevancy": answer_result.get("response_relevancy", 0.0),
                "noise_sensitivity": answer_result.get("noise_sensitivity", 1.0),
                "metrics": {
                    **state["metrics"],
                    "generation_time": duration,
                    "cached_eval_result": answer_result.get("eval_result", None)  # ✅ 添加缓存评估结果
                },
                "messages": state["messages"] + [{
                    "role": "assistant",
                    "content": answer_result["answer"]
                }]

            }
        except Exception as e:
            print(f"⚠️ 生成答案出错: {e}")
            return {
                **state,
                "answer": "抱歉，在生成答案时遇到了问题。",
                "error": f"生成答案失败: {str(e)}",
                "next_step": "end",
                "messages": state["messages"] + [{
                    "role": "system",
                    "content": f"生成答案失败: {str(e)}"
                }]
            }

    # -----------------------------
    # (4) 评估节点
    # -----------------------------
    def evaluator(state: AgentState) -> AgentState:
        print(f"⚡ Evaluator skipped (由 Generator 已评估)")
        return state

    # -----------------------------
    # (5) 路由器节点
    # -----------------------------
    def router(state: AgentState) -> AgentState:
        """
        根据多个指标综合判断下一步:
        1) 如果检索指标低, 优先 re-query
        2) 若检索合格但回答质量不够, re-generate
        3) 若回答质量好或尝试次数达上限, end
        """

        attempts = state["attempts"]
        requery_count = state.get("requery_count", 0)
        regenerate_count = state.get("regenerate_count", 0)
        max_attempts = state.get("max_attempts", 5)
        max_requeries = state.get("max_requeries", 2)
        max_regenerates = state.get("max_regenerates", 2)



        # ---- 解包各指标 ----
        # Retrieval metrics
        context_recall = state["context_recall"]    # e.g. 0.0 ~ 1.0
        context_precision = state["context_precision"]
        # Generation metrics
        faithfulness = state["faithfulness_score"]
        relevancy = state.get("response_relevancy", 0.0)
        noise = state.get("noise_sensitivity", 1.0)
        # 其他状态

        error = state.get("error", None)


        print(f"\n🔄 路由决策: attempts={attempts}, requery={requery_count}, regenerate={regenerate_count}")
        print(f"    → recall={context_recall:.2f}, precision={context_precision:.2f}")
        print(f"    → faith={faithfulness:.2f}, relevancy={relevancy:.2f}, noise={noise:.2f}")


        # 若已出错, 直接结束
        if error:
            return {**state, "next_step": "end", "attempts": attempts + 1}

        # --- Step 0: Generation 优先 override ---
        if faithfulness >= 0.7 and relevancy >= 0.7 and noise <= 0.4:
            print("✅ Generation quality is high, skipping retrieval quality check. Proceed to end.")
            return {**state, "next_step": "end", "attempts": attempts + 1}

        # --- Step 1: 检索质量是否明显不足(Recall/Precision过低) ---
        #    如果确实检索不理想, 更可能需要 re-query
            # --- 检索不足 ---
        if context_recall < 0.5 or context_precision < 0.3:
            if requery_count < max_requeries:
                return {
                    **state,
                    "next_step": "requery",
                    "requery_count": requery_count + 1,
                    "attempts": attempts + 1
                }
            else:
                # 达到 requery 上限时不强制 regenerate，直接 end
                print("⚠️ Retrieval attempts exhausted. Ending.")
                return {**state, "next_step": "end", "attempts": attempts + 1}

        # --- 回答质量差 ---
        if faithfulness < 0.6 or relevancy < 0.5 or noise > 0.4:
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

        # --- 回答已足够 ---
        return {**state, "next_step": "end", "attempts": attempts + 1}


    # -----------------------------
    # (6) requery_optimizer
    # -----------------------------
    def requery_optimizer(state: AgentState) -> AgentState:
        """基于已检索文档再次调用 ReasoningAgent.plan"""
        try:
            print(f"\n🔄 重新优化查询...")
            start = time.time()

            # 这次给 plan() 传入 docs，以便 Agent 优化 query
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
                    "content": f"重新优化的查询: {refined_query}"
                }]
            }
        except Exception as e:
            print(f"⚠️ 重新优化查询出错: {e}")
            return {
                **state,
                "error": f"重新优化查询失败: {str(e)}",
                "next_step": "end",
                "messages": state["messages"] + [{
                    "role": "system",
                    "content": f"重新优化查询失败: {str(e)}"
                }]
            }

    # -----------------------------
    # (7) finalizer
    # -----------------------------
    def finalizer(state: AgentState) -> AgentState:
        """完成流程, 统计总时间"""
        total_time = time.time() - state["start_time"]
        print(f"\n⏱️ 总处理时间: {total_time:.2f}秒")

        # 修改4: 清空缓存，释放资源
        retrieve_cache.clear()

        return {
            **state,
            "metrics": {**state["metrics"], "total_time": total_time}
        }

    # 建立 StateGraph
    workflow = StateGraph(AgentState)

    # 注册节点
    workflow.add_node("query_optimizer", query_optimizer)
    workflow.add_node("retriever", retriever)
    workflow.add_node("generator", generator)

    workflow.add_node("router", router)
    workflow.add_node("requery_optimizer", requery_optimizer)
    workflow.add_node("finalizer", finalizer)

    # 设置节点顺序
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

    # 设置入口
    workflow.set_entry_point("query_optimizer")
    return workflow.compile()

# -----------------------------
# 运行 RAG 流程
# -----------------------------

# ✅ 在 run_rag_pipeline 开头初始化
#evaluation_agent = EvaluationAgent(debug_mode=False)  #  debug mode

def run_rag_pipeline(
    question: str,
    retrieval_agent: RetrievalAgent,
    reasoning_agent: ReasoningAgent,
    generation_agent: GenerationAgent,
    evaluation_agent: EvaluationAgent,

    **kwargs
) -> Dict[str, Any]:

    # 可选参数：reference 作为 ground truth，方便评估阶段使用
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
        "context_recall": 0.0,
        "context_precision": 0.0,
        "attempts": 0,
        "requery_count": 0,
        "regenerate_count": 0,
        "max_attempts": 5,
        "max_regenerates": 2,
        "max_requeries": 2,
        "next_step": "",
        "next_step": "",
        "error": None,
        "start_time": time.time(),
        "metrics": {},
        "messages": [{"role": "user", "content": question}],
        "reference": reference  # 新增：支持 reference 评估
    }

    result = graph.invoke(initial_state)

    print(f"\n✅ 最终答案: {result['answer']}")
    print(f"📊 Faithfulness: {result['faithfulness_score']:.2f}, "
          f"Relevancy: {result['response_relevancy']:.2f}, "
          f"Noise: {result['noise_sensitivity']:.2f}")

    # 输出性能指标
    print("\n📈 性能指标:")
    for metric, value in result["metrics"].items():
        if isinstance(value, float):
            print(f"  - {metric}: {value:.2f}")
        else:
            print(f"  - {metric}: {value}")

    return result
