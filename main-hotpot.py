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
# ✅ import area (if there are other imports, put them above)
from tqdm import tqdm
import numpy as np
from decimal import Decimal
# 🔧 used to extract numbers from various results (list/np/float)
def extract_scalar(val):
    if isinstance(val, list) and val:
        return float(val[0])
    elif isinstance(val, (int, float, np.floating, np.generic, Decimal)):
        return float(val)
    else:
        return float(val) if val is not None else 0.0


# ✅ mode switch parameter
EVAL_MODE = "hybrid"  # optional: "strict", "lenient", "hybrid"

# ✅ new function: unified judgment of whether a sample is successful
def is_success(result: dict) -> bool:
    """Determine whether the sample is successful based on multiple indicators"""
    faith = extract_scalar(result.get("faithfulness_score", 0.0))
    rel = extract_scalar(result.get("response_relevancy", 0.0))
    noise = extract_scalar(result.get("noise_sensitivity", 1.0))
    sem_f1 = extract_scalar(result.get("semantic_f1_score", 0.0))
    recall = extract_scalar(result.get("context_recall", 0.0))

    if EVAL_MODE == "strict":
        return faith >= 0.7 and rel >= 0.7 and noise <= 0.4 and sem_f1 >= 0.7
    elif EVAL_MODE == "lenient":
        return sem_f1 >= 0.75 and recall >= 0.7
    else:  # hybrid mode
        return (
            (faith >= 0.7 and rel >= 0.7 and noise <= 0.4 and sem_f1 >= 0.7)
            or
            (sem_f1 >= 0.85 and recall >= 0.7)
        )




def main():

    # ------------------------
    # (1) Initialize DSPy LLM
    # ------------------------
    lm = dspy.LM('openai/gpt-3.5-turbo')
    dspy.configure(lm=lm)

    #  DSPy's SemanticF1 calculates the similarity between the answer and the ground truth
    semantic_f1_metric = SemanticF1(decompositional=True)

    # ------------------------
    # (2) Initialize Embeddings & FAISS
    # ------------------------
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore_path = "vectorstore-hotpot/hotpotqa_faiss"
    vectorstore = FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # ------------------------
    # (3) Initialize each Agent
    # ------------------------
    # Since ReasoningAgent no longer needs evaluation_agent to evaluate the final answer, remove its constructor parameter
    reasoning_agent = ReasoningAgent(model_name="gpt-3.5-turbo")
    # RetrieverAgent can still retain evaluation_agent for internal evaluation (optional)
    evaluation_agent = EvaluationAgent(model_name="gpt-3.5-turbo")
    retrieval_agent = RetrievalAgent(
        vectorstore=vectorstore,
        evaluation_agent=evaluation_agent,  # If not needed, pass None
        top_k=3
    )
    generation_agent = GenerationAgent(model_name="gpt-3.5-turbo", semantic_f1_metric=semantic_f1_metric)


    # ------------------------
    # (4) Test set loop
    # ------------------------
    # If ReasoningAgent uses DSPy for trainset/valset/testset,
    # you can take reason_agent.testset for simple testing
    # Assuming reason_agent still loads the dataset, take 5 test samples
    TESTSET_SIZE = 10
    total = min(TESTSET_SIZE, len(reasoning_agent.testset))
    subset_testset = reasoning_agent.testset[:total]

    print("\n🧪 Running test set evaluation...")
    correct = 0
    faithfulness_scores = []
    semantic_f1_scores = []  # ✅ 新增列表来收集每一项

    failed_cases = []



    for i, example in enumerate(tqdm(subset_testset, desc="Evaluating test set")):
        question = example.question
        ground_truth = example.response
        print(f"\n🔍 Running test {i+1}/{total} - Question: {question}")

        # (a) Call the new RAG Pipeline
        result = run_rag_pipeline(
            question=question,
            retrieval_agent=retrieval_agent,
            reasoning_agent=reasoning_agent,
            generation_agent=generation_agent,
            evaluation_agent=evaluation_agent,
            reference=ground_truth  # ✅ pass ground truth for Recall evaluation
        )
        print("📎 Ground Truth (pre-pipeline):", ground_truth)


        # (b) Get the answer and metrics (with default value protection)
        predicted_answer = result.get("answer", "")
        faithfulness_score = result.get("faithfulness_score", 0.0)
        relevancy_score = result.get("response_relevancy", 0.0)
        noise_score = result.get("noise_sensitivity", 1.0)
        semantic_f1_score = result.get("semantic_f1_score", 0.0)
        # (c) Calculate semantic similarity
        #similarity_score = semantic_f1_metric(example, dspy.Example(response=predicted_answer))
        faithfulness_scores.append(faithfulness_score)
        semantic_f1_scores.append(semantic_f1_score)  # ✅ collect each score


        # ✅ Use the unified function to determine if it passes
        if is_success(result):
            correct += 1
        else:
            failed_cases.append({
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer,
                "faithfulness_score": faithfulness_score,
                "relevancy_score": relevancy_score,
                "noise_score": noise_score,
                "semantic_f1_score": semantic_f1_score
            })


    # Calculate test accuracy & average faithfulness
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
    avg_f1 = sum(semantic_f1_scores) / len(semantic_f1_scores) if semantic_f1_scores else 0


    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\n✅ Test accuracy: {correct}/{total} ({accuracy:.2f}%)")
    print(f"📊 Average faithfulness score: {avg_faithfulness:.2f}")
    print(f"🧮 Average Semantic F1 score: {avg_f1:.2f}")  # ✅ 输出
    # 输出失败案例
    if failed_cases:
        print("\n⚠️ Failed cases (low faithfulness or low similarity):")
        for case in failed_cases[:5]:  # 仅打印前5个
            print("\n🔍 Question:", case["question"])
            print("📖 Standard answer:", case["ground_truth"])
            print("📝 Predicted answer:", case["predicted_answer"])
            print(f"📊 Faithfulness: {case['faithfulness_score']:.2f}")
            print(f"🎯 Relevancy: {case['relevancy_score']:.2f}")
            print(f"🔊 Noise sensitivity: {case['noise_score']:.2f}")
            print(f"✅ Semantic F1 score: {case['semantic_f1_score']:.2f}")

if __name__ == "__main__":
    main()

