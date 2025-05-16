import dspy
import json
import random
from langchain_openai import ChatOpenAI
import httpx


def load_dataset(json_path="data-hotpot/hotpot_mini_corpus.json", train_ratio=0.7, val_ratio=0.2):
    """加载 Hotpot mini corpus (json) 并拆分 trainset, valset, testset"""

    # 加载 JSON 文件
    with open(json_path, 'r') as f:
        corpus = json.load(f)

    # **确保数据足够大**
    if len(corpus) < 10:
        raise ValueError("⚠️ 数据量太少，无法训练 MIPROv2！请增加数据！")

    # **生成pseudo questions and answers**
    examples = []
    for item in corpus:
        # 🔥 现在 context 是已经清洗好的长文本
        context_text = item['context']
        question = item['question']
        answer = item['answer']

        examples.append(
            dspy.Example(context=context_text, question=question, response=answer).with_inputs("context", "question")
        )

    # **随机打乱数据**
    random.shuffle(examples)

    # **计算数据集大小**
    total_size = len(examples)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # **拆分数据**
    trainset, valset, testset = (
        examples[:train_size],
        examples[train_size:train_size+val_size],
        examples[train_size+val_size:]
    )

    # **确保trainset/valset不为空**
    if not trainset or not valset:
        raise ValueError("⚠️ trainset 或 valset 为空，无法运行 MIPROv2，请检查数据！")

    return trainset, valset, testset


class ReasoningAgent:
    """
    仅执行 Query/Prompt 优化，不直接评估最终回答质量，也不产出最终回答；
    而是返回 refined_query，供 RetrievalAgent 或后续环节调用。
    """

    def __init__(self, model_name="gpt-3.5-turbo", llm=None):
        if llm is not None:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(model_name=model_name, temperature=0)

        # ❶ ChainOfThought签名必须是简单格式
        self.chain = dspy.ChainOfThought("context, question -> response")

        # ❷ MIPROv2 优化器
        self.optimizer = dspy.MIPROv2(
            metric=dspy.evaluate.SemanticF1(),  # 用SemanticF1评估prompt优化效果
            auto="medium",
            num_threads=5
        )

        # ❸ 加载数据集
        self.trainset, self.valset, self.testset = load_dataset(val_ratio=0.1)

         # ❹ 编译 optimized agent (必须在有了 trainset/valset 后)
        self.optimized_agent = self.optimizer.compile(
            self.chain,
            trainset=self.trainset,
            valset=self.valset,
            requires_permission_to_run=False
        )

    def _should_fallback(self, retrieved_context: str) -> bool:
        """判断retrieved_context是否太差，需要fallback"""
        if len(retrieved_context.split()) < 50:
            return True
        keywords = ["overview", "history", "general", "unrelated", "background", "miscellaneous"]
        return any(kw.lower() in retrieved_context.lower() for kw in keywords)

    def _fewshot_examples(self) -> str:
        """提供 few-shot 示例"""
        return (
            "Example 1:\n"
            "Question: What are the main types of cloud computing?\n"
            "Retrieved docs: Discusses types of databases, benefits of cloud, and a small mention of SaaS.\n"
            "Refined query: Main categories of cloud computing services: IaaS, PaaS, and SaaS.\n\n"
            "Example 2:\n"
            "Question: How does photosynthesis work in plants?\n"
            "Retrieved docs: Talks about plant biology in general, mentions light and chlorophyll briefly.\n"
            "Refined query: Explain the stages of photosynthesis in plants, focusing on light-dependent and light-independent reactions.\n\n"
        )

    def _instruction_prefix(self) -> str:
        """生成优化 refined_query 时需要的 instruction"""
        return (
            "Given the retrieved context and the user question:\n"
            "1. Analyze whether the retrieved context is sufficient, overly broad, noisy, or off-topic.\n"
            "2. If the context contains irrelevant information or lacks key details, refine the query to be more specific, targeted, and avoid irrelevant topics.\n"
            "3. Your goal is to produce a new query that maximizes precision without sacrificing necessary recall.\n"
            "4. If the context is empty or irrelevant, generate a completely new, highly focused query based on the question itself.\n\n"
        )

    def plan(self, user_question, retrieved_docs=None):
        """核心方法：生成 refined_query，同时支持动态fallback和few-shot引导"""

        retrieved_context = ""
        fallback = False

        if retrieved_docs:
            retrieved_context = "\n".join(doc.page_content for doc in retrieved_docs)

            if self._should_fallback(retrieved_context):
                print("⚠️ 检测到retrieved context质量差，直接fallback到 user_question")
                fallback = True
            else:
                print("✅ Retrieved context质量良好，正常优化 refined_query")
                # 加 few-shot + 指令引导
                few_shot_context = self._fewshot_examples()
                instruction = self._instruction_prefix()
                retrieved_context = instruction + few_shot_context + "====\n" + retrieved_context
        else:
            print("⚠️ 没有retrieved docs，直接使用 user_question")
            fallback = True

        if fallback:
            # fallback时只用instruction + user_question
            dspy_response = self.optimized_agent(context=self._instruction_prefix(), question=user_question)
        else:
            dspy_response = self.optimized_agent(context=retrieved_context, question=user_question)

        if hasattr(dspy_response, "response"):
            refined_query = dspy_response.response
        else:
            refined_query = str(dspy_response)

        return {
            "refined_query": refined_query
        }


