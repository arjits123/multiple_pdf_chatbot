from PyPDF2 import PdfReader
import streamlit as st
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv 


load_dotenv()

genai.configure(api_key = os.getenv('GOOGLE_API_KEY'))

# read all the PDFS
def get_pdf_text(pdf_docs):
    text = ""
    for p in pdf_docs:
        pdf_reader = PdfReader(p)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

# Convert the text into chunks
def get_text_chunks(text):
    text_splitters = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
    chunks = text_splitters.split_text(text)
    return chunks

# Convert chunks into embeddings
def get_vector_store(chunk):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(chunk,embeddings)
    vector_store.save_local('faiss_index')

#Creating prompt template
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provive all the details, if the
    answer is not in the provided context, just say "answer is not available in the context", don't provide the wrong answer\n\n
    Context: \n {context}? \n
    Question: \n {question} \n
    
    Answer: 
"""
    model = ChatGoogleGenerativeAI(model = "gemini-pro", temperature = 0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ['context', 'question'])
    chain = load_qa_chain(model, chain_type = "stuff", prompt = prompt)
    return chain

# Get the user input
def user_input(user_question):
    """
    When you save a FAISS index, it contains vectors that were embedded using a specific embedding model. 
    To accurately perform searches or operations on this index after loading it, you need to use the same 
    embedding model. This ensures that the new queries are embedded in the 
    same vector space and can be compared correctly with the vectors in the index.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {'input_documents': docs, "question": user_question},
        return_only_outputs = True
    )

    print(response)
    st.write('Reply: ', response['output_text'])


# Create app
def main():
    st.set_page_config("Chat Multiple PDF")
    st.header("Chat with Multiple PDF using Gemini ")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()
