import bs4
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import CharacterTextSplitter
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4
import os
import argparse

# Set your OpenAI API key, Pinecone API key, Unstructured API key here
os.environ["OPENAI_API_KEY"] = 'your-openai-api-key'
os.environ["PINECONE_API_KEY"] = 'your-pinecone-api-key'
os.environ["UNSTRUCTURED_API_KEY"] = 'your-unstructured-api-key'

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
pc = Pinecone(api = pinecone_api_key)


def split(file_path, chunk_size, chunk_overlap):
    '''
    Split the file into chunks
    '''
    loader = UnstructuredLoader(
        file_path=file_path,
        api_key=os.getenv("UNSTRUCTURED_API_KEY"),
        partition_via_api=True,)
    documents = loader.load()
    # Initialize the text splitter
    text_splitter = CharacterTextSplitter(chunk_size, chunk_overlap)
    # split the loaded document
    split_docs = text_splitter.split_documents(documents)
    return split_docs

#   
def add_doc(split_docs: list, vector_store: PineconeVectorStore):
    '''
    add the list of splitted documents into the vector store
    '''
    uuids = [str(uuid4()) for _ in range(len(split_docs))]
    vector_store.add_documents(documents=split_docs, ids=uuids)

def connect_vector_store(index_name):
    '''
    Initialize an object connecting to the vector store linked to given index
    '''
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        raise Exception("The given index name does not exist")
    index = pc.Index(index_name)
    embeddings = OpenAIEmbeddings()
    return PineconeVectorStore(index=index, embedding=embeddings)

def retrieval_augmented_generation(model, query, vector_store, k, score_threshold):
    '''
    Generate the answer for the query by retrieving from the vector store
    '''
    system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    retriever = create_retriever(vector_store, k, score_threshold)
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input":query})
    return response["answer"]

def create_retriever(vector_store, k, score_threshold):
    retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": k, "score_threshold": score_threshold},
    )
    return retriever

def load_file_into_vectorbase(file_path, index_name, chunk_size=5000, chunk_overlap=0):
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    #index = pc.Index(index_name)
    split_docs = split(file_path, chunk_size, chunk_overlap)
    vector_store = connect_vector_store(index_name)
    add_doc(split_docs, vector_store)
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True, type=str)
    parser.add_argument("--index_name", required=True, type=str)
    parser.add_argument("--model_name", default="gpt-4o-mini", required=False, type=str)
    parser.add_argument("--temperature", default=1, required=False, type=int)
    parser.add_argument("--k", default=4, required=False, type=int)
    parser.add_argument("--score_threshold", default=0.5, required=False, type=float)
    parser.add_argument("--query_doc",  required=True, type=str)
    parser.add_argument("--answer_doc", required=True, type=str)
    return parser.parse_args()

def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

if __name__ == "main":
    args = get_args()
    files = get_all_file_paths(args.directory)

    if not files: # if the file_name is passed rather than the firectory
        files = [args.directory]

    # Add all the documents in the data directory to the vector store
    for file in files:
        load_file_into_vectorbase(file, args.index_name)

    vectorstore = connect_vector_store(args.index_name)

    model = ChatOpenAI(
        model=args.model_name,
        temperature=args.temperature,
        api_key=openai_api_key
    )

    with open(args.query_doc, 'r') as f:
        query_list = f.readlines()
    
    response_list = []
    for query in query_list:
        response = retrieval_augmented_generation(model, query, vectorstore, args.k, args.score_threshold)
        response_list.append(response)
    
    with open(args.answer_doc, "a") as f:
        for response in response_list:
            f.write(response_list)