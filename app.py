import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os

# Set up the Streamlit page configuration
st.set_page_config(page_title="PDF RAG App", page_icon="📚", layout="centered")

st.title("📚 Course Material Q&A App")
st.write("Upload a PDF of your course materials and ask questions about the content.")

# Sidebar for API Key and File Upload
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Google Gemini API Key:", type="password")
    
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

# Ensure the API key is provided before proceeding
if api_key:
    # Set the API key as an environment variable for LangChain to pick up
    os.environ["GOOGLE_API_KEY"] = api_key

    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Step 1: Read the PDF
        # We process the PDF only if it hasn't been processed in the current session
        if "vector_store" not in st.session_state:
            with st.spinner("Reading and processing the PDF..."):
                pdf_reader = PdfReader(uploaded_file)
                raw_text = ""
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        raw_text += text
                
                # Step 2: Split the text into chunks
                # We use overlap to ensure context isn't lost between chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(raw_text)
                
                # Step 3: Create embeddings and store in FAISS
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_store = FAISS.from_texts(chunks, embeddings)
                
                # Save the vector store in session state so it persists during interaction
                st.session_state.vector_store = vector_store
                st.success("PDF processed successfully! You can now ask questions.")

        # Step 4: Setup the QA Interface
        st.subheader("Ask a Question")
        user_question = st.text_input("What would you like to know about the course material?")
        
        if user_question:
            with st.spinner("Generating answer..."):
                # Initialize the Gemini model
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
                
                # Create a retrieval chain using the vector store from session state
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff", # "stuff" stuffs all retrieved chunks into the prompt
                    retriever=retriever
                )
                
                # Run the chain and display the answer
                response = qa_chain.invoke(user_question)
                st.write("**Answer:**")
                st.write(response["result"])
else:
    st.info("Please enter your Google Gemini API Key in the sidebar to get started.")
