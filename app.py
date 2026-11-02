import os
import streamlit as st
from google import genai
from google.genai import types

# Page Config
st.set_page_config(page_title="Gemini PDF Assistant", layout="centered")
st.title("📄 Gemini PDF RAG Assistant")

# API Key handling
if "GEMINI_API_KEY" in st.secrets:
  os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
else:
  api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
  if api_key:
    os.environ["GEMINI_API_KEY"] = api_key

# File Uploader (Max 10MB constraint)
uploaded_file = st.file_uploader("Upload a PDF file (max 10MB)", type=["pdf"])

if uploaded_file is not None:
  if uploaded_file.size > 10 * 1024 * 1024:
    st.error("File size exceeds 10MB limit. Please upload a smaller file.")
  else:
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
      f.write(uploaded_file.getbuffer())

    if not os.environ.get("GEMINI_API_KEY"):
      st.warning("Please enter your Gemini API key in the sidebar to proceed.")
    else:
      # Initialize client and persistent chat session safely
      if "file_name" not in st.session_state or st.session_state.file_name != uploaded_file.name:
        with st.spinner("Uploading and processing PDF with Gemini..."):
          # Create client instance
          client = genai.Client()
          
          # Upload file
          gemini_file = client.files.upload(file=temp_file_path)
          
          # Initialize chat session
          chat = client.chats.create(
              model="gemini-2.5-flash",
              config=types.GenerateContentConfig(
                  system_instruction=(
                      "You are a helpful assistant specialized in analyzing "
                      "the provided PDF document. Answer questions accurately "
                      "based strictly on the document's content."
                  )
              ),
          )
          
          # Store state persistently
          st.session_state.uploaded_gemini_file = gemini_file
          st.session_state.chat = chat
          st.session_state.file_name = uploaded_file.name
          st.success("PDF ready! You can now ask questions below.")

      # Chat Interface State
      if "messages" not in st.session_state:
        st.session_state.messages = []

      for message in st.session_state.messages:
        with st.chat_message(message["role"]):
          st.markdown(message["content"])

      if user_query := st.chat_input("Ask something about the PDF..."):
        st.session_state.messages.append(
            {"role": "user", "content": user_query}
        )
        with st.chat_message("user"):
          st.markdown(user_query)

        with st.chat_message("assistant"):
          with st.spinner("Thinking..."):
            try:
              # Send message using the saved chat session
              response = st.session_state.chat.send_message([
                  st.session_state.uploaded_gemini_file,
                  user_query,
              ])
              answer = response.text
              st.markdown(answer)
              st.session_state.messages.append(
                  {"role": "assistant", "content": answer}
              )
            except Exception as e:
              st.error(f"Error communicating with Gemini: {e}")

    if os.path.exists(temp_file_path):
      os.remove(temp_file_path)
