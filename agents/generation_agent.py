from langchain_openai import ChatOpenAI
import numpy as np

def extract_scalar(val):
        if isinstance(val, list) and val:
            return float(val[0])
        elif isinstance(val, (int, float, np.floating, np.generic)):
            return float(val)
        else:
            return float(val) if val is not None else 0.0

class GenerationAgent:
    def __init__(self, model_name="gpt-3.5-turbo", llm=None):
        if llm is not None:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(model_name=model_name, temperature=0)

    def _compute_combined_score(self, faith, relevancy, noise):
        # 可调整的加权组合函数
        return 0.5 * faith + 0.4 * relevancy - 0.3 * noise

    def _trim_context(self, docs, max_chars=3000):
        combined = "\n".join([f"<Document {i}> {doc.page_content}" for i, doc in enumerate(docs, start=1)])
        if len(combined) > max_chars:
            print(f"⚠️ Retrieved context too long ({len(combined)} chars), trimming to {max_chars} chars")
            combined = combined[:max_chars]
        return combined


    def answer(self, question, docs, evaluation_agent, max_attempts=3):
        """
        生成答案，并结合"分指标评估" (Faithfulness, Response Relevancy, Noise Sensitivity等)。
        若质量不足，优化 Prompt 并重新生成。
        """
        # 将检索到的文档拼成文本
        context = "\n".join([f"<Document {i}> {doc.page_content}" for i, doc in enumerate(docs, start=1)])

        attempts = 0
        best_answer = ""
        best_combined_score = -1.0  # 初始化为负分确保任何得分都能更新

        context = self._trim_context(docs)

        while attempts < max_attempts:
            prompt = f"""
            You are an AI assistant that answers questions based strictly on retrieved documents.
            Follow this structured reasoning process:

            Instructions:
            1. Extract the key facts from the retrieved context.
            2. Summarize the main points that directly answer the question.
            3. Write the final answer **explicitly and directly answering the question**, using the same key entities and keywords as the user question.
            4. Do not add general comments or indirect responses. Do not mention the retrieved context or your process.
            5. If the retrieved context does not contain enough information, explicitly state "The provided context does not contain enough information to answer this question."

            Question: {question}
            Retrieved Context:

            {context}

            Strictly based on the retrieved context above, provide the most accurate and faithful answer.

            """

            if len(prompt.split()) * 1.5 > 12000:
                print("⚠️ Prompt too long even after context trimming. Aborting.")
                return {
                    "answer": "抱歉，内容过长无法生成，请检查输入。",
                    "faithfulness_score": 0.0,
                    "response_relevancy": 0.0,
                    "noise_sensitivity": 1.0,
                }
            # 1) 调用 LLM 生成回答
            raw_answer = self.llm.invoke(prompt)
            answer_text = raw_answer.content if hasattr(raw_answer, 'content') else str(raw_answer)


            # 2) 使用"分指标评估" evaluate_generation
            #    (如果你只想要faithfulness, 也可在EvaluationAgent里只评估faithfulness)
            eval_result = evaluation_agent.evaluate_generation(
                user_query=question,
                retrieved_docs=docs,
                response=answer_text
            )

            # 3) 解析分数
            faithfulness_score = eval_result.get("faithfulness", 0.0)
            relevancy_score = eval_result.get("response_relevancy", 0.0)

            # 修复噪声敏感度提取
            noise_keys = [k for k in eval_result.keys() if "noise_sensitivity" in k]
            if noise_keys:
                raw_noise = eval_result[noise_keys[0]]
                print(f"DEBUG extracted raw_noise: {raw_noise}")
                noise_sensitivity = extract_scalar(raw_noise)
            else:
                noise_sensitivity = 1.0  # fallback 只有在完全提取失败时使用

            #print(f"DEBUG: Parsed Noise Sensitivity: {noise_sensitivity} from raw: {raw_noise}")


            combined_score = self._compute_combined_score(faithfulness_score, relevancy_score, noise_sensitivity)

            print(f"🔍 Attempt #{attempts+1}")
            print(f"   Faithfulness: {faithfulness_score:.2f}")
            print(f"   Relevancy: {relevancy_score:.2f}")
            print(f"   NoiseSensitivity: {noise_sensitivity:.2f}")

            # 4) 设定多维阈值 (根据需求灵活调整)
            if faithfulness_score >= 0.6 and relevancy_score >= 0.5 and noise_sensitivity <= 0.4:
                print("✅ Generation quality is sufficient. Returning answer.")
                return {
                    "answer": answer_text,
                    "faithfulness_score": faithfulness_score,
                    "response_relevancy": relevancy_score,
                    "noise_sensitivity": noise_sensitivity,
                    "cached_eval_result": eval_result  # 🆕 加这一行
                }

            # 如果当前得分比之前最好更高，就更新best_answer
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_answer = answer_text
                best_metrics = {
                    "faithfulness_score": faithfulness_score,
                    "response_relevancy": relevancy_score,
                    "noise_sensitivity": noise_sensitivity,
                }

            # 不达标 → 可以进行一些 Prompt 改写 或 Retry
            print("⚠️ Generation quality insufficient. Refining prompt and retrying...\n")
            attempts += 1

        print("⚠️ Max attempts reached, returning best available answer.")
        return {
            "answer": best_answer,
            **best_metrics,
        }



