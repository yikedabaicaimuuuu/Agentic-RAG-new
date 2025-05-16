from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from langchain.embeddings import OpenAIEmbeddings

from ragas.metrics import (
    ContextPrecision,
    LLMContextRecall,
    Faithfulness,
    ResponseRelevancy,
    NoiseSensitivity
)
from ragas.llms import LangchainLLMWrapper
import traceback
import numpy as np




class EvaluationAgent:
    """
    多功能评估Agent:
      1) quick_evaluate(...) 仅用LLM简单判断检索是否足以回答问题
      2) evaluate_retrieval(...) 专门计算检索Precision/Recall
      3) evaluate_generation(...) 专门计算回答质量(Faithfulness等)
      4) full_evaluate(...) 兼容以前的综合评估逻辑(同时算 context_precision, context_recall, faithfulness)
    """

    def __init__(self, model_name="gpt-3.5-turbo", embeddings=None, llm=None):
        #self.debug_mode = debug_mode 如果未来需要使用，需要添加这个参数
        if llm is not None:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(model_name=model_name, temperature=0)

        if embeddings is None:
            try:
                # Try to import and initialize embeddings (OpenAI embeddings or other)
                print("📦 Initializing OpenAI embeddings for ResponseRelevancy")
                self.embeddings = OpenAIEmbeddings()
            except (ImportError, Exception) as e:
                print(f"⚠️ Could not initialize embeddings: {str(e)}")
                print("⚠️ ResponseRelevancy will have reduced effectiveness")
                self.embeddings = None
        else:
            self.embeddings = embeddings  # Use provided embeddings

        # 原先 full_evaluate 使用的通用 metrics
        # (ContextPrecision, LLMContextRecall, Faithfulness)
        self.metrics = [
            ContextPrecision(),
            LLMContextRecall(),
            Faithfulness()
        ]

        # 可选: 如果想专门区分retrieval vs generation，用不同列表:
        # self.retrieval_metrics = [...]
        # self.generation_metrics = [...]

    # ----------------------------------------------------------------------
    # 1) quick_evaluate(...) - 保留原逻辑
    # ----------------------------------------------------------------------
    def quick_evaluate(self, question, docs):
        """
        仅使用 LLM 评估：判断检索是否足够回答问题，并提供关键词建议。
        """
        if not docs:
            return {"sufficient": False, "suggested_keywords": "expand keywords"}

        context = " ".join([doc.page_content for doc in docs])
        eval_prompt = (
            f"Question: {question}\n"
            f"Retrieved content: {context[:1000]}...\n"
            "Is this information sufficient to answer the question? "
            "Please respond with 'sufficient' or 'insufficient' and provide additional keywords."
        )

        eval_response = self.llm.predict(eval_prompt)
        sufficient = "sufficient" in eval_response.lower()
        suggested_keywords = " ".join(eval_response.split()[-3:])

        return {"sufficient": sufficient, "suggested_keywords": suggested_keywords}

    # ----------------------------------------------------------------------
    # 2) 提取数字的工具函数 - 保留原逻辑
    # ----------------------------------------------------------------------
    def _get_numeric_value(self, value):
        """
        Extract a numeric value from various possible data types:
        - If it's a list, return the average
        - If it's a number, return it directly
        - Otherwise return 0
        """
        # Handle None case
        if value is None:
            return 0

        # ✅ Handle direct numeric or numpy float types
        if isinstance(value, (int, float, np.float32, np.float64)):
            return float(value)

        # ✅ If value is a list (e.g., [np.float64(0.0)]), average valid entries
        if isinstance(value, list):
            if not value:
                return 0
            numeric_values = [
                float(v) for v in value
                if isinstance(v, (int, float, np.float32, np.float64))
            ]
            if not numeric_values:
                return 0
            return sum(numeric_values) / len(numeric_values)


        # Try to convert to float
        try:
            return float(value)
        except (TypeError, ValueError):
            # Try to extract from object if it has a numeric attribute
            if hasattr(value, "value") and isinstance(getattr(value, "value"), (int, float)):
                return getattr(value, "value")
            return 0

    def _extract_score(self, result, metric_name):
        """
        Extract score from RAGAS result structure, handling different formats.
        """

        # 1. result 是 dict 且包含 metric_name
        if isinstance(result, dict) and metric_name in result:
            return self._get_numeric_value(result[metric_name])

        # 2. result 有 __getitem__ 方法（模拟 dict）且不是字符串
        if hasattr(result, "__getitem__") and not isinstance(result, str):
            try:
                return self._get_numeric_value(result[metric_name])
            except (KeyError, TypeError):
                pass

        # 3. result 有 .scores 属性（部分 ragas 版本结构）
        if hasattr(result, "scores"):
            scores = result.scores
            if isinstance(scores, dict) and metric_name in scores:
                return self._get_numeric_value(scores[metric_name])

        # 4. result 有 .data 属性（ragas 1.x 中的 EvaluationResult.data）
        if hasattr(result, "data") and isinstance(result.data, dict):
            if metric_name in result.data:
                return self._get_numeric_value(result.data[metric_name])

        # 5. result 是一个包含 .name 属性的 list-like 结构
        if hasattr(result, "__iter__") and not isinstance(result, (str, dict)):
            for item in result:
                if hasattr(item, "name") and item.name == metric_name:
                    return self._get_numeric_value(getattr(item, "score", 0))

        # ❌ 最后兜底: 打印调试信息
        result_type = type(result).__name__
        available_attrs = []

        if isinstance(result, dict):
            available_attrs = list(result.keys())
        elif hasattr(result, "__dict__"):
            available_attrs = list(result.__dict__.keys())
        elif hasattr(result, "scores") and isinstance(result.scores, dict):
            available_attrs = list(result.scores.keys())

        print(f"⚠️ Could not find metric '{metric_name}' in result of type '{result_type}'")
        print(f"⚠️ Available keys/attributes: {available_attrs}")

        return 0


    # ----------------------------------------------------------------------
    # 3) evaluate_retrieval(...) - 新增: 只评估检索质量(Precision/Recall)
    # ----------------------------------------------------------------------
    def evaluate_retrieval(self, user_query, retrieved_docs, reference=None):
        """
        Specialized function for RetrieverAgent:
        - Only evaluates context_precision, context_recall
        """
        #if self.debug_mode:
        #    return self._debug_evaluate_retrieval(user_query, retrieved_docs, reference)
        retrieved_texts = [doc.page_content for doc in retrieved_docs if doc.page_content.strip()]
        if not retrieved_texts:
            print("⚠️ Warning: No retrieved texts to evaluate in retrieval!")
            retrieved_texts = ["N/A"]

        # Build RAGAS data
        data = {
            "user_input": user_query,
            "retrieved_contexts": retrieved_texts,
            "response": "N/A",  # No response text at retrieval stage
            "reference": reference if reference else "N/A"
        }

        print(f"🔍 DEBUG: Query length: {len(user_query)}")
        print(f"🔍 DEBUG: Retrieved docs count: {len(retrieved_texts)}")

        dataset = EvaluationDataset.from_list([data])

        # Metrics for Precision/Recall
        metrics = [ContextPrecision(), LLMContextRecall()]

        try:
            result = evaluate(dataset=dataset, metrics=metrics, llm=LangchainLLMWrapper(self.llm))

            print("\n🔎 Debugging retrieval evaluation result type:", type(result))
            print("🔎 Debugging retrieval evaluation result:", result)

            # DIRECT ACCESS - this is the key fix based on the debug output you provided
            if isinstance(result, dict):
                context_precision = self._get_numeric_value(result.get('context_precision', 0))
                context_recall = self._get_numeric_value(result.get('context_recall', 0))
            else:
                # Fall back to the extraction method if not a dict
                context_precision = self._get_numeric_value(self._extract_score(result, "context_precision"))
                context_recall = self._get_numeric_value(self._extract_score(result, "context_recall"))

            print(f"🎯 Context Precision: {context_precision:.4f}")
            print(f"📈 Context Recall: {context_recall:.4f}")

        except Exception as e:
            print(f"❌ Error in evaluate_retrieval: {str(e)}")
            import traceback
            traceback.print_exc()
            context_precision = 0
            context_recall = 0

        return {
            "context_precision": context_precision,
            "context_recall": context_recall
        }
    # ----------------------------------------------------------------------
    # 4) evaluate_generation(...) - 新增: 只评估回答质量(Faithfulness等)
    # ----------------------------------------------------------------------
    def evaluate_generation(self, user_query, retrieved_docs, response, reference=None):
        """
        Evaluates response quality:
        - Faithfulness (whether faithful to retrieved_docs)
        - ResponseRelevancy (whether response aligns with user_query)
        - NoiseSensitivity (whether response is disturbed by incorrect context)
        """

        retrieved_texts = [doc.page_content for doc in retrieved_docs if doc.page_content.strip()]
        if not retrieved_texts:
            print("⚠️ Warning: No retrieved texts to evaluate in generation!")
            retrieved_texts = ["N/A"]

          # 🔧 自动裁剪 response 防止评估时超长
        if len(response) > 3000:
            print(f"⚠️ Response too long ({len(response)} chars), trimming to 3000 chars")
            response = response[:3000]

        if not response or response.strip() == "":
            print("⚠️ Warning: Empty response to evaluate!")
            response = "N/A"

        data = {
            "user_input": user_query,
            "retrieved_contexts": retrieved_texts,
            "response": response,
            "reference": reference if reference else "N/A"
        }

        print(f"🔍 DEBUG: Query length: {len(user_query)}")
        print(f"🔍 DEBUG: Retrieved docs: {len(retrieved_texts)}")
        print(f"🔍 DEBUG: Response length: {len(response)}")

        dataset = EvaluationDataset.from_list([data])

        try:
            # Check if embeddings are available for ResponseRelevancy
            if self.embeddings is None:
                print("⚠️ Warning: No embeddings provided for ResponseRelevancy - using simplified metrics")
                metrics = [Faithfulness(), NoiseSensitivity(llm=LangchainLLMWrapper(self.llm))]
            else:
                metrics = [
                    Faithfulness(),
                    ResponseRelevancy(embeddings=self.embeddings, llm=LangchainLLMWrapper(self.llm)),
                    NoiseSensitivity(llm=LangchainLLMWrapper(self.llm))
                ]

            result = evaluate(dataset=dataset, metrics=metrics, llm=LangchainLLMWrapper(self.llm))

            print(f"\n🔎 Debugging: Result type: {type(result).__name__}")
            print(f"🔎 Debugging generation evaluation result: {result}")

            # Extract the dictionary data from the result
            result_dict = {}

            # If it's already a dictionary
            if isinstance(result, dict):
                result_dict = result
            # Try to access through __dict__ attribute
            elif hasattr(result, '__dict__'):
                # Try to extract scores from internal attributes
                if '_scores_dict' in result.__dict__:
                    result_dict = result._scores_dict
                elif 'scores' in result.__dict__ and isinstance(result.scores, dict):
                    result_dict = result.scores
            # If it has a string representation that looks like a dict, try to parse it
            else:
                try:
                    import ast
                    str_result = str(result)
                    if str_result.startswith('{') and str_result.endswith('}'):
                        result_dict = ast.literal_eval(str_result)
                except:
                    pass

            # Print the extracted dictionary
            print(f"🔍 DEBUG: Extracted result dict: {result_dict}")
            print(f"🔍 DEBUG: Result dict keys: {list(result_dict.keys())}")

            # Get metrics from the extracted dictionary
            faithfulness = self._get_numeric_value(result_dict.get('faithfulness', 0))

            # Check for either 'answer_relevancy' or 'response_relevancy'
            relevancy = 0
            if 'answer_relevancy' in result_dict:
                relevancy = self._get_numeric_value(result_dict.get('answer_relevancy'))
            elif 'response_relevancy' in result_dict:
                relevancy = self._get_numeric_value(result_dict.get('response_relevancy'))

            # Check for noise_sensitivity with any additional suffix
            noise_sensitivity = 0
            for k in result_dict.keys():
                if "noise_sensitivity" in k:
                    noise_sensitivity = self._get_numeric_value(result_dict[k])
                    print(f"🔧 Parsed Noise Sensitivity: {noise_sensitivity:.4f} from key: {k}")

                break

            print(f"📊 Faithfulness: {faithfulness:.4f}")
            print(f"🎯 Response Relevancy: {relevancy:.4f}")
            print(f"🔊 Noise Sensitivity: {noise_sensitivity:.4f}")

        except Exception as e:
            print(f"❌ Error in evaluate_generation: {str(e)}")
            import traceback
            traceback.print_exc()
            faithfulness = 0
            relevancy = 0
            noise_sensitivity = 0

        return {
            "faithfulness": faithfulness,
            "response_relevancy": relevancy,
            "noise_sensitivity": noise_sensitivity
        }

    # def _debug_evaluate_retrieval(self, user_query, retrieved_docs, reference=None):
    #     """
    #     Enhanced retrieval evaluation with claims + entailment debug (for recall drift diagnosis)
    #     """
    #     from ragas import evaluate as ragas_evaluate
    #     from ragas.metrics import context_precision, context_recall

    #     retrieved_texts = [doc.page_content for doc in retrieved_docs if doc.page_content.strip()]
    #     if not retrieved_texts:
    #         print("⚠️ No retrieved texts to evaluate in debug mode!")
    #         retrieved_texts = ["N/A"]

    #     testset = Testset.from_dict({
    #         "question": [user_query],
    #         "contexts": [retrieved_texts],
    #         "ground_truth": [reference or ""]
    #     })

    #     result = ragas_evaluate(
    #         testset,
    #         metrics=[context_precision, context_recall]
    #     )

    #     precision = float(result["context_precision"][0])
    #     recall = float(result["context_recall"][0])

    #     # Debug claims and entailment
    #     if hasattr(result, 'claims') and hasattr(result, 'entailment'):
    #         claims_list = result.claims[0]
    #         entailment_scores = result.entailment[0]
    #         print(f"📝 Claims extracted: {claims_list}")
    #         print(f"✅ Entailment scores: {entailment_scores}")
    #         for claim, score in zip(claims_list, entailment_scores):
    #             print(f"→ Claim: {claim}\n   Entailment Score: {score:.2f}\n")

    #     return {
    #         "context_precision": precision,
    #         "context_recall": recall
    #     }


    # ----------------------------------------------------------------------
    # 5) full_evaluate(...) - 保留兼容旧逻辑
    # ----------------------------------------------------------------------
    def full_evaluate(self, query, retrieved_docs, response=None, reference=None):
        """
        使用 RAGAS 进行完整评估：
        - 检索质量（context precision, recall）
        - 生成质量（faithfulness）
        """
        retrieved_texts = [doc.page_content for doc in retrieved_docs if doc.page_content.strip()]
        if not retrieved_texts:
            print("⚠️ Warning: No retrieved texts to evaluate!")
            retrieved_texts = ["N/A"]

        # 提取 ground truth reference

        data = {
            "user_input": query,
            "retrieved_contexts": retrieved_texts,
            "response": response if response else "N/A",
            "reference": reference if reference else "N/A"
        }
        # Print debug info
        print(f"\n🔍 DEBUG: Query length: {len(query)}")
        print(f"🔍 DEBUG: Retrieved docs: {len(retrieved_texts)}")
        print(f"🔍 DEBUG: Response length: {len(response) if response else 0}")

        dataset = EvaluationDataset.from_list([data])
        metrics = [ContextPrecision(), LLMContextRecall(), Faithfulness()]
        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=LangchainLLMWrapper(self.llm)
            )

            print("\n🔎 Debugging: Raw Evaluation Result ➝", result)

            # 兼容不同返回结构：ragas 1.x 使用 result.scores，否则直接用 result 本身
            scores = getattr(result, "scores", result)

            if not scores:
                print("❌ Scores object is empty or None")
                return {"faithfulness": 0, "context_recall": 0, "context_precision": 0}

            # Print all available keys for debugging
            print(f"🔑 Available score keys: {list(scores.keys()) if hasattr(scores, 'keys') else 'No keys'}")

            faithfulness_score = self._get_numeric_value(scores.get("faithfulness", 0))
            context_recall = self._get_numeric_value(scores.get("context_recall", 0))
            context_precision = self._get_numeric_value(scores.get("context_precision", 0))

            print("✅ Metrics extracted successfully")

        except Exception as e:
            print(f"❌ Detailed error in evaluation: {str(e)}")

            traceback.print_exc()
            faithfulness_score = 0
            context_recall = 0
            context_precision = 0

        print(f"📊 Faithfulness Score: {faithfulness_score:.4f}")
        print(f"📈 Context Recall: {context_recall:.4f}")
        print(f"🎯 Context Precision: {context_precision:.4f}")

        return {
            "faithfulness": faithfulness_score,
            "context_recall": context_recall,
            "context_precision": context_precision
        }




