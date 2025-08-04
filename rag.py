
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# === TEXT EMBEDDING SETUP ===

text_embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text):
    return text_embedder.encode(text, normalize_embeddings=True)

# === PDF LOADING AND EMBEDDING ===

pdf_path = "/content/policy.pdf"
doc = fitz.open(pdf_path)

all_docs = []
all_embeddings = []

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

for i, page in enumerate(doc):
    text = page.get_text()
    if text.strip():
        doc_obj = Document(page_content=text, metadata={'page': i})
        chunks = splitter.split_documents([doc_obj])
        for chunk in chunks:
            emb = embed_text(chunk.page_content)
            all_docs.append(chunk)
            all_embeddings.append(emb)

doc.close()

# === BUILD FAISS VECTOR STORE ===

text_vector_store = FAISS.from_embeddings(
    text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, all_embeddings)],
    embedding=None
)

# === RETRIEVAL ===

def retrieve(query, k=5):
    query_embedding = embed_text(query)
    return text_vector_store.similarity_search_by_vector(query_embedding, k=k)

# === GEMINI PROMPT ===

def create_prompt(query, retrieved_docs):
    context = "\n\n".join([
        f"[Page {doc.metadata.get('page', '?')}]: {doc.page_content.strip()}"
        for doc in retrieved_docs
    ])

    return HumanMessage(content=f"""You are given the following excerpts from an insurance document:

{context}

Question: {query}

Please answer the question using only the given content. Be specific and concise.""")

# === LLM SETUP ===

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3
)

# === MAIN PIPELINE ===

def rag_pipeline(query):
    context_docs = retrieve(query, k=5)
    message = create_prompt(query, context_docs)
    response = llm.invoke([message])

    print(f"\nRetrieved {len(context_docs)} documents:")
    for doc in context_docs:
        preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        print(f"  - Page {doc.metadata.get('page', '?')}: {preview}")
    print()

    return response.content

# === RUN EXAMPLE ===

if __name__ == "__main__":
    query = "What is the waiting period for pre-existing diseases?"
    print(f"Query: {query}")
    print("-" * 50)
    answer = rag_pipeline(query)
    print(f"\nAnswer:\n{answer}")
    print("=" * 70)

