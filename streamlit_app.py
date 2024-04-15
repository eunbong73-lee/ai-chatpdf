__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import streamlit as st
###제목
st.title("ChatPDF")
st.write("---")

###파일업로드 ( https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader )
uploaded_file = st.file_uploader("PDF 파일을 올려주세요", type=['pdf']))
st.write("---")

###파일 업로드되면 동작하는 함수
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os
def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

### 업로드되면 동작하는 코드 ( Ref -- https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/chat_with_documents.py )
if uploaded_file is not None:
    
    pages = pdf_to_document(uploaded_file)

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)

    # Embedding
    from langchain_openai import OpenAIEmbeddings
    embeddings_model = OpenAIEmbeddings()

    # load it int Chroma
    from langchain_chroma import Chroma
    vectordb = Chroma.from_documents(texts, embeddings_model)


    # Question
    st.header("PDF에게 질문해 보세요 !!!")
    question = st.text_input('질문을 입력하세요')
    
    if st.button('질문하기'):
        with st.spinner('답변 작성 중'):
            from langchain.chains import RetrievalQA
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())
            result = qa_chain ({"query":question})
            st.write(result)
