from langchain_community.vectorstores import FAISS
from utils.embedding import get_embeddings

def create_vector_store(chunks):

    print("Chunks received:", len(chunks))

    embeddings = get_embeddings()

    print("Embeddings loaded")

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    print("FAISS created")

    vector_store.save_local("vectorstore")

    print("Saved Successfully")

    return vector_store