import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


def split_text_by_sentence(text, chunk_size=300, chunk_overlap=50):
    """基于句子边界进行文本切分"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["。", "？", "！", "\n"]  # 句子分隔符
    )
    return text_splitter.split_text(text)

def preprocess_data(csv_path="zip/data/ucsc_data.csv"):
    """加载 CSV 并对 context 进行句子级 chunking"""
    df = pd.read_csv(csv_path)
    processed_docs = []

    for _, row in df.iterrows():
        question, answer, context = row["question"], row["answer"], row["context"]
        chunks = split_text_by_sentence(context)

        for chunk in chunks:
            processed_docs.append({"question": question, "answer": answer, "context": chunk})

    processed_df = pd.DataFrame(processed_docs)
    processed_df.to_csv("data/processed_ucsc_data.csv", index=False)
    return processed_df

def create_vectorstore(csv_path="zip/data/processed_ucsc_data.csv", persist_path="zip/vectorstore/faiss_index"):
    """Load processed data and create a FAISS vector database"""
    df = pd.read_csv(csv_path)
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-small-v2",
        model_kwargs={"device": "cpu"}
    )

    docs = [
        Document(
            page_content=f"passage: {row['context']}",
            metadata={"question": row["question"], "answer": row["answer"]}
        )
        for _, row in df.iterrows()
    ]

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(persist_path)
    return vectorstore

if __name__ == "__main__":
    preprocess_data()
    create_vectorstore()
