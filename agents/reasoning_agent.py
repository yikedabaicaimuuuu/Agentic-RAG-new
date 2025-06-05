import dspy
import json
import random
from langchain_openai import ChatOpenAI
import httpx
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import trim_text_to_tokens




def load_dataset(json_path="data-hotpot/hotpot_mini_corpus.json", train_ratio=0.7, val_ratio=0.2):
    """加载 Hotpot mini corpus (json) 并拆分 trainset, valset, testset"""

    # Load JSON file
    with open(json_path, 'r') as f:
        corpus = json.load(f)

    # **Ensure data is large enough**
    if len(corpus) < 10:
        raise ValueError("⚠️ Data is too small, cannot train MIPROv2! Please increase data!")

    # **Generate pseudo questions and answers**
    examples = []
    for item in corpus:
        # 🔥 Now context is already cleaned long text
        context_text = item['context']
        question = item['question']
        answer = item['answer']

        examples.append(
            dspy.Example(context=context_text, question=question, response=answer).with_inputs("context", "question")
        )

    # **Shuffle data randomly**
    random.shuffle(examples)

    # **Calculate data set size**
    total_size = len(examples)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # **Split data**
    trainset, valset, testset = (
        examples[:train_size],
        examples[train_size:train_size+val_size],
        examples[train_size+val_size:]
    )

    # **Ensure trainset/valset is not empty**
    if not trainset or not valset:
        raise ValueError("⚠️ trainset or valset is empty, cannot run MIPROv2, please check data!")

    return trainset, valset, testset


class ReasoningAgent:
    """
    Only perform Query/Prompt optimization, do not directly evaluate final answer quality, nor produce final answer;
    but return refined_query, RetrievalAgent can use it.
    """

    def __init__(self, model_name="gpt-3.5-turbo", llm=None):
        if llm is not None:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=1000)

        # ❶ ChainOfThought signature must be a simple format
        self.chain = dspy.ChainOfThought("context, question -> response")

        # ❷ MIPROv2 optimizer
        self.optimizer = dspy.MIPROv2(
            metric=dspy.evaluate.SemanticF1(),  # Use SemanticF1 to evaluate prompt optimization effect
            auto="medium",
            num_threads=5
        )

        # ❸ Load dataset
        self.trainset, self.valset, self.testset = load_dataset(val_ratio=0.1)

         # ❹ Compile optimized agent (must after trainset/valset is loaded)
        self.optimized_agent = self.optimizer.compile(
            self.chain,
            trainset=self.trainset,
            valset=self.valset,
            requires_permission_to_run=False
        )

    def _should_fallback(self, retrieved_context: str) -> bool:
        """Determine if retrieved_context is too poor, need fallback"""
        if len(retrieved_context.split()) < 50:
            return True
        keywords = ["overview", "history", "general", "unrelated", "background", "miscellaneous"]
        return any(kw.lower() in retrieved_context.lower() for kw in keywords)

    def _fewshot_examples(self) -> str:
        """Provide few-shot examples"""
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
        """Generate instruction needed for optimizing refined_query"""
        return (
            "Given the retrieved context and the user question:\n"
            "1. Analyze whether the retrieved context is sufficient, overly broad, noisy, or off-topic.\n"
            "2. If the context contains irrelevant information or lacks key details, refine the query to be more specific, targeted, and avoid irrelevant topics.\n"
            "3. Your goal is to produce a new query that maximizes precision without sacrificing necessary recall.\n"
            "4. If the context is empty or irrelevant, generate a completely new, highly focused query based on the question itself.\n\n"
        )

    def plan(self, user_question, retrieved_docs=None):
        """Core method: generate refined_query, support dynamic fallback and few-shot guidance"""

        retrieved_context = ""
        fallback = False

        if retrieved_docs:
            retrieved_context = "\n".join(doc.page_content for doc in retrieved_docs)

            if self._should_fallback(retrieved_context):
                print("⚠️ Detected retrieved context is poor, fallback directly to user_question")
                fallback = True
            else:
                print("✅ Retrieved context is good, normal optimization of refined_query")
                # Add few-shot + instruction guidance
                few_shot_context = self._fewshot_examples()
                instruction = self._instruction_prefix()
                full_prompt = instruction + few_shot_context + "====\n" + retrieved_context

                # ✅ Trim the concatenated prompt to prevent exceeding context window
                trimmed_prompt = trim_text_to_tokens(full_prompt, max_tokens=8000)
                retrieved_context = trimmed_prompt
        else:
            print("⚠️ No retrieved docs, use user_question directly")
            fallback = True

        if fallback:
            # When fallback, only use instruction + user_question
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


