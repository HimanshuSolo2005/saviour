import os
import gradio as gr
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

loader = PyPDFLoader("Chapter_1.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

from transformers import pipeline
llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
llm = HuggingFacePipeline(pipeline=llm_pipeline)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def chatbot(query: str) -> str:
    result = qa_chain.invoke(query)
    if isinstance(result, dict) and "result" in result:
        return result["result"]
    return str(result)

iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(lines=2, placeholder="Ask me about Operating Systems..."),
    outputs="text",
    title="OS Knowledge Chatbot",
    description="Chat with your OS subject."
)

if __name__ == "__main__":
    iface.launch()
