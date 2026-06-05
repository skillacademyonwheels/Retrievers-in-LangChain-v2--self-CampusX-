from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv
import os
import streamlit as st
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

st.title("Vector Store Example with Chroma and HuggingFace Embeddings")

# Create some sample documents
loader = CSVLoader(file_path="imdb_top_1000.csv", encoding="utf-8")
docs = loader.load()

# Create an embedding model
# langchain-huggingface v1 does not accept `huggingfacehub_api_token` directly.
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"token": HF_API_KEY} if HF_API_KEY else {},
)

# Create a Chroma vector store
vector_store = Chroma(collection_name="imdb_top_1000", persist_directory="chroma_db", embedding_function=embedding_model)

# Add documents only the first time so reruns do not duplicate the same rows.
if vector_store._collection.count() == 0:
    vector_store.add_documents(docs)

# Persistence is automatic with langchain_chroma when persist_directory is set.


# Example of querying the vector store
query = st.text_input("Enter a query to search the IMDB top 1000 movies:")
results = vector_store.similarity_search(query, k=5)

st.write("### Search Results:",results)

def _extract_field(result, field_name):
    """Read a field from metadata first, then from CSVLoader page_content."""
    if field_name in result.metadata and result.metadata[field_name]:
        return result.metadata[field_name]

    prefix = f"{field_name}: "
    for line in result.page_content.splitlines():
        if line.startswith(prefix):
            value = line[len(prefix):].strip()
            if value:
                return value
    return "N/A"


for result in results:
    print(result.page_content)
    print(result.metadata)
    # title = _extract_field(result, "Series_Title")
    # score = _extract_field(result, "IMDB_Rating")
    # print(f"Title: {title}, Score: {score}")