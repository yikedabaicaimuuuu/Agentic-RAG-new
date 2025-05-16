from agents.reasoning_agent import ReasoningAgent
from agents.retrieval_agent import RetrievalAgent
from agents.evaluation_agent import EvaluationAgent
from agents.generation_agent import GenerationAgent
from agents.langgraph_rag import run_rag_pipeline

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import dspy
from dspy.evaluate import SemanticF1
from tqdm import tqdm
import httpx
from langchain_openai import ChatOpenAI
import atexit

def main():

    # ------------------------
    # (1) 初始化 DSPy LLM
    # ------------------------
    lm = dspy.LM('openai/gpt-3.5-turbo')
    dspy.configure(lm=lm)

    # ------------------------
    # (2) 初始化 Embeddings & FAISS
    # ------------------------
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore_path = "vectorstore-hotpot/hotpotqa_faiss"
    vectorstore = FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # ------------------------
    # (3) 初始化各 Agent
    # ------------------------
    # 由于 ReasoningAgent 不再需要evaluation_agent来评估最终回答, 去掉其构造参数
    reasoning_agent = ReasoningAgent(model_name="gpt-3.5-turbo")
    # RetrieverAgent 可依旧保留evaluation_agent用来内部评估(可选)
    evaluation_agent = EvaluationAgent(model_name="gpt-3.5-turbo")
    retrieval_agent = RetrievalAgent(
        vectorstore=vectorstore,
        evaluation_agent=evaluation_agent,  # 若不需要也可传 None
        top_k=3
    )
    generation_agent = GenerationAgent(model_name="gpt-3.5-turbo")

    # ------------------------
    # (4) 测试集循环
    # ------------------------
    # 如果 ReasoningAgent 内部用 DSPy 有 trainset/valset/testset,
    # 可以取 reason_agent.testset做简单测试
    # 假设 reason_agent 仍加载了 dataset, 取 5 个测试样本
    TESTSET_SIZE = 2
    total = min(TESTSET_SIZE, len(reasoning_agent.testset))
    subset_testset = reasoning_agent.testset[:total]

    print("\n🧪 运行测试集评估...")
    correct = 0
    faithfulness_scores = []
    failed_cases = []

    # 用 DSPy 的 SemanticF1 计算回答与groundtruth相似度
    semantic_f1_metric = SemanticF1(decompositional=True)

    for i, example in enumerate(tqdm(subset_testset, desc="评估测试集")):
        question = example.question
        ground_truth = example.response
        print(f"\n🔍 运行测试 {i+1}/{total} - 问题: {question}")

        # (a) 调用新的 RAG Pipeline
        result = run_rag_pipeline(
            question=question,
            retrieval_agent=retrieval_agent,
            reasoning_agent=reasoning_agent,
            generation_agent=generation_agent,
            evaluation_agent=evaluation_agent,
            reference=example.context  # ✅ 传入 ground truth 用于 Recall 评估
        )

        # (b) 获取回答和指标（加上默认值保护）
        predicted_answer = result.get("answer", "")
        faithfulness_score = result.get("faithfulness_score", 0.0)
        relevancy_score = result.get("answer_relevancy", 0.0)
        noise_score = result.get("noise_sensitivity", 1.0)
        # 也可拿 response_relevancy, noise_sensitivity if needed

        # (c) 计算语义相似度
        similarity_score = semantic_f1_metric(example, dspy.Example(response=predicted_answer))
        faithfulness_scores.append(faithfulness_score)

        # 判断是否正确 (arbitrary threshold)
        if similarity_score > 0.7:
            correct += 1
        else:
            failed_cases.append({
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer,
                "faithfulness_score": faithfulness_score,
                "relevancy_score": relevancy_score,
                "noise_score": noise_score,
                "similarity_score": similarity_score
            })

    # 计算测试准确率 & 平均忠实度
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\n✅ 测试准确率: {correct}/{total} ({accuracy:.2f}%)")
    print(f"📊 平均忠实度评分: {avg_faithfulness:.2f}")

    # 输出失败案例
    if failed_cases:
        print("\n⚠️ 失败案例 (低忠实度或低相似度):")
        for case in failed_cases[:5]:  # 仅打印前5个
            print("\n🔍 问题:", case["question"])
            print("📖 标准答案:", case["ground_truth"])
            print("📝 预测答案:", case["predicted_answer"])
            print(f"📊 忠实度: {case['faithfulness_score']:.2f}")
            print(f"🎯 相关度: {case['relevancy_score']:.2f}")
            print(f"🔊 噪音敏感度: {case['noise_score']:.2f}")
            print(f"✅ 语义相似度: {case['similarity_score']:.2f}")

if __name__ == "__main__":
    main()

