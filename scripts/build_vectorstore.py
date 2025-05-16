import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

def split_text_by_sentence(text, chunk_size=300, chunk_overlap=50):
    """基于句子边界进行文本切分"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["。", "？", "！", "\n", ".", "?", "!"]  # 支持中英文句号
    )
    return text_splitter.split_text(text)

def load_hotpot_mini(json_path="data-hotpot/hotpot_mini_corpus.json"):
    """加载Hotpot mini corpus (json)"""
    with open(json_path, 'r') as f:
        corpus = json.load(f)
    return corpus

def create_vectorstore(json_path="data-hotpot/hotpot_mini_corpus.json", persist_path="vectorstore-hotpot/hotpotqa_faiss"):
    """从Hotpot mini corpus创建FAISS向量库"""

    # 1. 加载小型corpus
    print("🚀 Loading mini corpus...")
    corpus = load_hotpot_mini(json_path)
    print(f"✅ Loaded {len(corpus)} items.")

    # 2. 做chunking
    print("✂️ Splitting into chunks...")
    docs = []
    for item in corpus:
        # 使用正确的字段名称: context 而不是 content
        if "context" not in item:
            print(f"⚠️ Warning: Missing 'context' in item with question: {item.get('question', 'Unknown')}")
            continue

        context = item['context']
        question = item.get('question', '')
        answer = item.get('answer', '')

        # 为每个chunk添加问题和答案作为元数据，便于后续检索
        chunks = split_text_by_sentence(context)
        for chunk in chunks:
            if chunk.strip():
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "question": question,
                        "answer": answer,
                        # 我们可以从context中提取标题 (如果格式是"Title: content")
                        "title": chunk.split(':', 1)[0] if ':' in chunk else ""
                    }
                ))

    print(f"✅ Prepared {len(docs)} chunks from {len(corpus)} items.")

    # 打印一些示例chunks以便验证
    if docs:
        print("\n📝 Sample chunks:")
        for i in range(min(3, len(docs))):
            print(f"Chunk {i+1}:")
            print(f"  Content: {docs[i].page_content[:100]}...")
            print(f"  Metadata: {docs[i].metadata}")
        print()

    # 3. 建立Embedding和Vectorstore
    print("🧠 Embedding documents...")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # 跟你main里保持一致

    vectorstore = FAISS.from_documents(docs, embeddings)

    # 4. 保存
    print(f"💾 Saving FAISS index to {persist_path}...")
    os.makedirs(persist_path, exist_ok=True)
    vectorstore.save_local(persist_path)

    print("✅ Vectorstore created successfully.")
    print(f"  - Contains {len(docs)} vectors")
    print(f"  - Saved to: {persist_path}")

if __name__ == "__main__":
    create_vectorstore()
