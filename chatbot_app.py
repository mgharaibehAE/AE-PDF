import streamlit as st
import tempfile
import os
import random
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Load and display logo
logo_url = "logo.png"  # Replace with the URL of your logo or local file path
st.image(logo_url, width=200)

st.title("Ask your PDF")

# Input for OpenAI API Key
openai_api_key = st.sidebar.text_input("Enter OpenAI API Key:", value="", type="password")

st.sidebar.markdown("""
**Disclaimer:**

This product is developed and distributed by AnalytiXplore. It is provided "as is" without warranty of any kind, either express or implied, including but not limited to the implied warranties of merchantability or fitness for a particular purpose.

AnalytiXplore does not collect, store, process, or manage any data accessed or generated through the use of this product. All data handling, storage, and management are the sole responsibility of the user or the user's organization. AnalytiXplore shall have no responsibility or liability with respect to any data privacy laws, regulations, or standards that may be applicable.

Users are strongly advised to review their data privacy practices and ensure compliance with all relevant laws and regulations. It is the user’s responsibility to secure any necessary consents, permissions, or authorizations required for the collection, processing, and storage of data.

Use of this product signifies acceptance of these terms and an understanding that AnalytiXplore bears no responsibility for any data privacy concerns arising from the use of this product.
""", unsafe_allow_html=True)

if st.sidebar.button('Contact Us'):
    st.sidebar.write("[Send Email](mailto:info@analytixplore.com)")

# Initialize or get the session state
if 'chain' not in st.session_state:
    st.session_state['chain'] = None
    st.session_state['chat_history'] = []

# File Uploader
uploaded_file = st.file_uploader("Please select a PDF file to upload (Max size: 10MB):", type=["pdf"])

if uploaded_file:
    file_size = len(uploaded_file.read())
    uploaded_file.seek(0)  # Reset file pointer to the beginning
    
    if file_size > 10 * 1024 * 1024:  # Check if file size is greater than 10MB
        st.error("The file size exceeds 10MB. Please upload a smaller file.")
    elif st.session_state['chain'] is None and openai_api_key:
        with st.spinner("Processing..."):
            # Create a temporary file to store the uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            # Process the PDF file
            loader = UnstructuredFileLoader(tmp_file_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_overlap=0)
            texts = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            doc_search = Chroma.from_documents(texts, embeddings)
            st.session_state['chain'] = VectorDBQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type="stuff", vectorstore=doc_search)
            
            # Delete the temporary file
            os.remove(tmp_file_path)
        
        st.success("Processing complete!")

# User Input for Questions
query = st.text_input("Ask a question about the file:")

if query:
    if st.button("Submit"):
        if st.session_state['chain'] is not None:
            with st.spinner("Searching..."):
                response = st.session_state['chain'].run(query)
            st.session_state['chat_history'].append({'question': query, 'answer': response})
        else:
            st.error("Please upload and process a file first.")

# Display Chat History
st.subheader("Chat History")
for chat in reversed(st.session_state['chat_history']):
    st.markdown(f"**Q:** {chat['question']}")
    st.markdown(f"**A:** {chat['answer']}")
