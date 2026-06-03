import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
import faiss

from langchain.vectorstores import FAISS
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from a parquet file into a pandas DataFrame.

    Args:
        file_path (str): The path to the parquet file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    try:
        df = pd.read_parquet(file_path, engine='pyarrow')
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

def create_vectorstore(documents: List[Dict[str,Any]])-> FAISS:
    """
    Create a vector store from a list of documents.

    Args:
        documents (List[Dict[str, Any]]): A list of documents where each document is a dictionary 
        containing 'pid' (int) and 'text' (Document).

    Returns:
        FAISS: A FAISS vector store containing the document embeddings.
    """
    # Initialize the vector store
    batch_encode_size = 256
    len_embed = 384

    nlp_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                model_kwargs={'device':'cpu'},
                                encode_kwargs = {'normalize_embeddings': True, 'batch_size': batch_encode_size})

    index = faiss.IndexFlatIP(len_embed)

    vector_store = FAISS(
        embedding_function=nlp_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        distance_strategy = DistanceStrategy.COSINE,
    )
    # Add documents to the vector store
    tmp_doc = []
    tmp_pid = []
    for doc in tqdm(documents,total=len(documents)):
        tmp_doc.append(doc['text'])
        tmp_pid.append(doc['pid'])

        if len(tmp_pid) >= batch_encode_size:
            vector_store.add_documents(documents=tmp_doc, ids=tmp_pid)
            tmp_doc = []
            tmp_pid = []

    vector_store.add_documents(documents=tmp_doc, ids=tmp_pid)

    return vector_store

def vectorize_dataset(file_path: str, save_path:str) -> FAISS:
    """
    Vectorize a dataset and create a FAISS vector store.

    Args:
        file_path (str): The path to the parquet file containing the dataset.
        save_path (str): The path where the vector store will be saved.
    Returns:
        FAISS: A FAISS vector store containing the document embeddings.
    """
    # Load the dataset
    df = load_dataset(file_path)
    print(df.head())
    if df.empty:
        print("No data to vectorize.")
        return None
    # Edit the DataFrame to ensure it has the required columns
    documents = []
    for index, row in df.iterrows():
        doc = Document(page_content=row['passage'], metadata={'pid': index})
        documents.append({'pid': index, 'text': doc})
    # Create the vector store
    vector_store = create_vectorstore(documents)
    # Check if the vector store is created successfully
    doc_count = vector_store.index.ntotal
    if doc_count == 0:
        print("Vector store creation failed, no documents indexed.")
        return None
    print(f"Vector store created with {doc_count} documents.")
    # Save the vector store to disk
    vector_store.save_local(save_path)

    return vector_store

if __name__ == "__main__":
        # Example usage
        file_path = "src\model\dataset\passages.parquet"
        save_path = "src\model\dataset\\faiss"
        vector_store = vectorize_dataset(file_path, save_path)
        if vector_store:
            print(f"Vector store saved successfully at {save_path}.")