from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):

    print("Documents:", len(documents))

    for i, doc in enumerate(documents):
        print(f"Page {i+1} length:", len(doc.page_content))
        print(doc.page_content[:100])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print("Chunks:", len(chunks))

    return chunks