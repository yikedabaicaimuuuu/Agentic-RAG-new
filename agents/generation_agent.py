from langchain_openai import ChatOpenAI
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import safe_trim_prompt, trim_text_to_token_limit
import dspy
from dspy.evaluate import SemanticF1



def extract_scalar(val):
        if isinstance(val, list) and val:
            return float(val[0])
        elif isinstance(val, (int, float, np.floating, np.generic)):
            return float(val)
        else:
            return float(val) if val is not None else 0.0

class GenerationAgent:
    def __init__(self, model_name="gpt-3.5-turbo", llm=None, semantic_f1_metric=None):
        if llm is not None:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=1000)

        self.semantic_f1_metric = semantic_f1_metric

    def _compute_combined_score(self, faith, relevancy, noise):
        # adjustable weights
        return 0.5 * faith + 0.4 * relevancy - 0.3 * noise

    def _trim_context(self, docs, max_tokens=8000):
        combined = "\n".join([f"<Document {i}> {doc.page_content}" for i, doc in enumerate(docs, start=1)])
        return trim_text_to_token_limit(combined, max_tokens=max_tokens, model="gpt-3.5-turbo")


    def _build_prompt(self, question, context, attempt):
        instructions = """
        You are an AI assistant that answers questions based strictly on retrieved documents.
        Follow this structured reasoning process:

        Instructions:
        1. Extract the key facts from the retrieved context.
        2. Summarize the main points that directly answer the question.
        3. Write the final answer **explicitly and directly answering the question**, using the same key entities and keywords as the user question.
        4. Do not add general comments or indirect responses. Do not mention the retrieved context or your process.
        5. If the retrieved context does not contain enough information, explicitly state "The provided context does not contain enough information to answer this question."
        """

        if attempt > 0:
            instructions += f"\n(Note: This is attempt #{attempt + 1}. Try to be more concise and faithful than before.)"

        prompt = f"""
        {instructions}

        Question: {question}
        Retrieved Context:
        {context}

        Strictly based on the retrieved context above, provide the most accurate and faithful answer.
        """
        return safe_trim_prompt(prompt, model="gpt-3.5-turbo")


    def answer(self, question, docs, evaluation_agent, ground_truth=None, max_attempts=3):
        context = self._trim_context(docs)
        best_answer = ""
        best_combined_score = -1.0
        best_metrics = {
            "faithfulness_score": 0.0,
            "response_relevancy": 0.0,
            "noise_sensitivity": 1.0,
            "semantic_f1_score": 0.0
        }
        best_eval_result = None  # ✅ Add cache best evaluation results

        for attempt in range(max_attempts):
            prompt = self._build_prompt(question, context, attempt)
            print(f"✂️ Prompt token-trimmed to fit model limit.")
            print(f"🧪 Ground truth inside GenerationAgent: {ground_truth}")


            raw_answer = self.llm.invoke(prompt)
            answer_text = raw_answer.content if hasattr(raw_answer, 'content') else str(raw_answer)

            eval_result = evaluation_agent.evaluate_generation(
                user_query=question,
                retrieved_docs=docs,
                response=answer_text
            )

            faithfulness_score = extract_scalar(eval_result.get("faithfulness", 0.0))
            relevancy_score = extract_scalar(eval_result.get("response_relevancy", 0.0))
            noise_sensitivity = 1.0  # 默认最大
            for k, v in eval_result.items():
                if "noise_sensitivity" in k:
                    noise_sensitivity = extract_scalar(v)
                    break

            # ✅ Insert SemanticF1 calculation logic
            if self.semantic_f1_metric and ground_truth:
                try:
                    semantic_f1_score = self.semantic_f1_metric(
                        dspy.Example(question=question, response=ground_truth),
                        dspy.Example(response=answer_text)
                    )
                    print(f"   🧠 SemanticF1: {semantic_f1_score:.2f}")

                except Exception as e:
                    print(f"   ⚠️ Failed to compute SemanticF1: {e}")
                    semantic_f1_score = 0.0
            else:
                semantic_f1_score = 0.0

            eval_result["semantic_f1_score"] = semantic_f1_score


            print(f"🔍 Attempt #{attempt + 1}")
            print(f"   Faithfulness: {faithfulness_score:.2f}")
            print(f"   Relevancy: {relevancy_score:.2f}")
            print(f"   NoiseSensitivity: {noise_sensitivity:.2f}")
            print(f"   SemanticF1: {semantic_f1_score:.2f}")

            combined_score = self._compute_combined_score(faithfulness_score, relevancy_score, noise_sensitivity)

            if (
                faithfulness_score >= 0.6 and
                relevancy_score >= 0.7 and
                noise_sensitivity <= 0.4 and
                semantic_f1_score >= 0.7
            ):
                print("✅ Generation quality is sufficient. Returning answer.")
                return {
                    "answer": answer_text,
                    "faithfulness_score": faithfulness_score,
                    "response_relevancy": relevancy_score,
                    "noise_sensitivity": noise_sensitivity,
                    "semantic_f1_score": semantic_f1_score,
                    "cached_eval_result": eval_result,
                    "eval_result": eval_result  # ✅ ensure semanticF1 is not lost
                }

            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_answer = answer_text
                best_metrics = {
                    "faithfulness_score": faithfulness_score,
                    "response_relevancy": relevancy_score,
                    "noise_sensitivity": noise_sensitivity,
                    "semantic_f1_score": semantic_f1_score
                }
                best_eval_result = eval_result  # ✅ Synchronize best evaluation results

            print("⚠️ Generation quality insufficient. Refining prompt and retrying...\n")

        print("⚠️ Max attempts reached, returning best available answer.")
        return {
            "answer": best_answer,
            "faithfulness_score": best_metrics.get("faithfulness_score", 0.0),
            "response_relevancy": best_metrics.get("response_relevancy", 0.0),
            "noise_sensitivity": best_metrics.get("noise_sensitivity", 1.0),
            "semantic_f1_score": best_metrics.get("semantic_f1_score", 0.0),
            "cached_eval_result": best_eval_result,
            "eval_result": best_eval_result  # ✅ Add this line to ensure final result includes all metrics
        }




