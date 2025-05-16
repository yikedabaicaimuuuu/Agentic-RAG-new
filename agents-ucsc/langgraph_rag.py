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

def create_rag_graph(
    retrieval_agent: RetrievalAgent,
    reasoning_agent: ReasoningAgent,
    generation_agent: GenerationAgent,
    evaluation_agent: EvaluationAgent
):
    @lru_cache(maxsize=100)
    def cached_retrieve(query: str):
        return retrieval_agent.retrieve(query)

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
            print(f"\n📚 基于优化后的查询进行检索: {query}")

            start = time.time()
            docs = cached_retrieve(query)
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

            # 如果想记录检索质量(Recall/Precision)，可调用:
            ret_eval = evaluation_agent.evaluate_retrieval(query, docs)
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
                "metrics": {**state["metrics"], "generation_time": duration},
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
        """
        评估生成的答案质量:
        - 不再用 full_evaluate, 改用 evaluate_generation
        - 取 faithfulness, response_relevancy, noise_sensitivity
        """
        try:
            query = state["refined_query"]
            docs = state["docs"]
            answer = state["answer"]

            print(f"\n📊 评估答案质量...")
            start = time.time()
            # 用“回答评估”
            eval_result = evaluation_agent.evaluate_generation(query, docs, answer)
            duration = time.time() - start

            faithfulness = float(eval_result.get("faithfulness", 0))
            relevancy = float(eval_result.get("response_relevancy", 0))
            noise_keys = [k for k in eval_result.keys() if k.startswith("noise_sensitivity")]
            noise = evaluation_agent._get_numeric_value(eval_result[noise_keys[0]]) if noise_keys else 0.0


            return {
                **state,
                "faithfulness_score": faithfulness,
                "response_relevancy": relevancy,
                "noise_sensitivity": noise,
                "metrics": {**state["metrics"], "evaluation_time": duration},
                "messages": state["messages"] + [{
                    "role": "system",
                    "content": (
                        f"评估结果: 忠实度={faithfulness:.2f}, "
                        f"相关度={relevancy:.2f}, 噪音敏感度={noise:.2f}"
                    )
                }]
            }
        except Exception as e:
            print(f"⚠️ 评估出错: {e}")
            return {
                **state,
                "faithfulness_score": 0.0,
                "response_relevancy": 0.0,
                "noise_sensitivity": 1.0,
                "error": f"评估失败: {str(e)}",
                "next_step": "end",
                "messages": state["messages"] + [{
                    "role": "system",
                    "content": f"评估失败: {str(e)}"
                }]
            }

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

        # ---- 解包各指标 ----
        # Retrieval metrics
        context_recall = state["context_recall"]    # e.g. 0.0 ~ 1.0
        context_precision = state["context_precision"]
        # Generation metrics
        faithfulness = state["faithfulness_score"]
        relevancy = state.get("response_relevancy", 0.0)
        noise = state.get("noise_sensitivity", 1.0)
        # 其他状态
        attempts = state["attempts"]
        error = state.get("error", None)

        print(f"\n🔄 路由决策: attempt={attempts}, "
            f"recall={context_recall:.2f}, prec={context_precision:.2f}, "
            f"faith={faithfulness:.2f}, relevancy={relevancy:.2f}, noise={noise:.2f}")

        # 若已出错, 直接结束
        if error:
            return {**state, "next_step": "end", "attempts": attempts + 1}

        # --- Step 1: 检索质量是否明显不足(Recall/Precision过低) ---
        #    如果确实检索不理想, 更可能需要 re-query
        if context_recall < 0.5 or context_precision < 0.3:
            if attempts >= 3:
                # 多次尝试后仍不行 => 放弃
                next_step = "end"
            else:
                next_step = "requery"
            return {**state, "attempts": attempts + 1, "next_step": next_step}

        # --- Step 2: 回答质量检查 (Faithfulness, Relevancy, Noise等) ---
        #   2.1 Faithfulness (回答是否忠实上下文)
        #       若低于阈值(0.6), 再次生成(或决定改写 Query?)
        if faithfulness < 0.6:
            # 可以判断 attempts 次数, 决定是 re-generate 还是 re-query
            if attempts < 2:
                next_step = "regenerate"
            else:
                next_step = "requery"
            return {**state, "attempts": attempts + 1, "next_step": next_step}

        #   2.2 Relevancy (回答与问题对齐程度)
        #       若相关度过低(<0.5), 可能只需 re-generate
        if relevancy < 0.5:
            if attempts < 3:
                next_step = "regenerate"
            else:
                next_step = "end"
            return {**state, "attempts": attempts + 1, "next_step": next_step}

        #   2.3 NoiseSensitivity (回答是否被错误信息干扰)
        #       若噪音敏感度过高(>0.4), 也可 re-generate
        if noise > 0.4:
            if attempts < 2:
                next_step = "regenerate"
            else:
                # 如果多次都改不掉干扰 => requery
                next_step = "requery"
            return {**state, "attempts": attempts + 1, "next_step": next_step}

        # --- Step 3: 全部指标尚可, or 已达尝试上限 => end ---
        if attempts >= 3:
            next_step = "end"
        else:
            next_step = "end"  # 回答已达标, 所以结束

        return {**state, "attempts": attempts + 1, "next_step": next_step}


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
    workflow.add_node("evaluator", evaluator)
    workflow.add_node("router", router)
    workflow.add_node("requery_optimizer", requery_optimizer)
    workflow.add_node("finalizer", finalizer)

    # 设置节点顺序
    workflow.add_edge("query_optimizer", "retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", "evaluator")
    workflow.add_edge("evaluator", "router")
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
def run_rag_pipeline(
    question: str,
    retrieval_agent: RetrievalAgent,
    reasoning_agent: ReasoningAgent,
    generation_agent: GenerationAgent,
    evaluation_agent: EvaluationAgent,
    **kwargs
) -> Dict[str, Any]:

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
        "next_step": "",
        "error": None,
        "start_time": time.time(),
        "metrics": {},
        "messages": [{"role": "user", "content": question}]
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
