from langchain_community.vectorstores import FAISS
from utils.embedding import get_embeddings


def get_retriever():

    embeddings = get_embeddings()

    vector_store = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever