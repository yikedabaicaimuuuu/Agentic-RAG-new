from langchain_openai import ChatOpenAI

class GenerationAgent:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)

    def answer(self, question, docs, evaluation_agent, max_attempts=3):
        """
        生成答案，并结合“分指标评估” (Faithfulness, Response Relevancy, Noise Sensitivity等)。
        若质量不足，优化 Prompt 并重新生成。
        """
        # 将检索到的文档拼成文本
        context = "\n".join([f"<Document {i}> {doc.page_content}" for i, doc in enumerate(docs, start=1)])

        attempts = 0
        best_answer = ""
        best_score = 0.0  # 如果你想用 faithfulness 做衡量，也可使用综合分

        while attempts < max_attempts:
            prompt = f"""
            You are an AI assistant that answers questions based strictly on retrieved documents.
            Follow this structured reasoning process:

            Step 1: Extract the key facts from the retrieved context.
            Step 2: Summarize the main points that directly answer the question.
            Step 3: Generate the final answer strictly based on these extracted points.

            Question: {question}
            Retrieved Context:

            {context}

            Strictly based on the retrieved context above, provide the most accurate and faithful answer.
            If the context does not contain enough information, state that explicitly.
            """
            # 1) 调用 LLM 生成回答
            raw_answer = self.llm.invoke(prompt)
            answer_text = raw_answer.content if hasattr(raw_answer, 'content') else str(raw_answer)

            # 2) 使用“分指标评估” evaluate_generation
            #    (如果你只想要faithfulness, 也可在EvaluationAgent里只评估faithfulness)
            eval_result = evaluation_agent.evaluate_generation(
                user_query=question,
                retrieved_docs=docs,
                response=answer_text
            )

            # 3) 解析分数
            faithfulness_score = eval_result.get("faithfulness", 0.0)
            relevancy_score = eval_result.get("response_relevancy", 0.0)
            noise_sensitivity = eval_result.get("noise_sensitivity", 1.0)  # 假设默认=1 代表噪音高

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
                }

            # 如果当前得分比之前最好更高，就更新best_answer
            if faithfulness_score > best_score:
                best_answer = answer_text
                best_score = faithfulness_score

            # 不达标 → 可以进行一些 Prompt 改写 或 Retry
            print("⚠️ Generation quality insufficient. Refining prompt and retrying...\n")
            attempts += 1

        print("⚠️ Max attempts reached, returning best available answer.")
        return {
            "answer": best_answer if best_answer else answer_text,
            "faithfulness_score": faithfulness_score,
            "response_relevancy": relevancy_score,
            "noise_sensitivity": noise_sensitivity,
        }
