import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import JSONLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
import time

# Loading the API Keys
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.environ["GROQ_API_KEY"]

# Initializing the LLM
llm = ChatGroq(api_key=groq_api_key,
               model_name = "mixtral-8x7b-32768")

# Loading the cleaned data
article = "/Users/dhruvyadav/Desktop/Assesments/ML-Question/cleaned_data.json"

# Defining a basic prompt
prompt = ChatPromptTemplate.from_template(
"""
You are a question answer chatbot. Your task is to give accurate answers based on the context given. If nothing matches the context do not answer anything.
<context>
{context}
</context>
Question: {input}
"""
)


# UI and QnA App Implementation
st.title("Question/Answering App")


def vector_embeddings():
    if "vectors" not in st.session_state:
        st.write("Please wait vectors are being generated")
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = JSONLoader(article, jq_schema=".[]", text_content=False)
        st.session_state.documents = st.session_state.loader.load()
        st.write("JSON file loaded")
        time.sleep(1)
        st.write("Now generating vectors")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents)
        # Due to lack of computational resources I didn't use all the document
        st.session_state.vectors = ObjectBox.from_documents(st.session_state.final_documents[:1000], st.session_state.embeddings, embedding_dimensions=768)
        time.sleep(1)
        st.write("Vectors generated!")
        st.session_state.initialized = True

placeholder = st.empty()
output_parser = StrOutputParser()

if "initialized" not in st.session_state:
    st.session_state.initialized = False
with placeholder:
    vector_embeddings()
placeholder.empty()

if st.session_state.initialized:
    input_prompt = st.text_input("Enter your question here")
    submit = st.button("Ask")
    if input_prompt and submit:
        documents_chain = create_stuff_documents_chain(llm, prompt, output_parser=output_parser)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, documents_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({"input":input_prompt})
        st.write("Response time : ", time.process_time()-start)
        st.write(response["answer"])
        with st.expander("Context for the response"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("----------------------------------")