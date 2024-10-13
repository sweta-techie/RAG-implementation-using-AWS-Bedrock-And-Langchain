import boto3
import streamlit as st
from langchain.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but summarize with 
at least 250 words with detailed explanations. If you don't know the answer, 
just say that you don't know; don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

## Bedrock Client
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Get embedding model
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", client=bedrock
)

def get_documents():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500
    )
    docs = text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_llm():
    # Use the correct text generation model
    llm = Bedrock(
        model_id="amazon.titan-text-express-v1",  # Corrected model_id
        client=bedrock,
        model_kwargs={
            'maxTokenCount': 512,
            'temperature': 0.5,
            'topP': 0.9
        }
    )
    return llm

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config("RAG Demo")
    st.header("End-to-End RAG Application")
    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update or Create Vector Store:")
        if st.button("Store Vector"):
            with st.spinner("Processing..."):
                docs = get_documents()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Send"):
        if not user_question:
            st.error("Please enter a question.")
            return
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local(
                "faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True
            )
            llm = get_llm()
            try:
                response = get_response_llm(llm, faiss_index, user_question)
                st.write(response)
            except ValueError as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
