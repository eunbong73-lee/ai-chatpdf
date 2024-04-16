__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import streamlit as st
from streamlit_extras.buy_me_a_coffee import button

button(username="eunbong" , floating=True, width=221)

###제목
st.title("ChatPDF")
st.write("---")


###파일업로드 ( https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader )
uploaded_file = st.file_uploader("PDF 파일을 올려주세요", type="pdf")
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


    # Stream 받아 줄 Handler 만들기
    from langchain.callbacks.base import BaseCallbackHandler
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text = initial_text
        def on_llm_new_token(self, token:str, **kwargs) -> None:
            self.text+=token
            self.container.markdown(self.text)

    
    if st.button('질문하기') or question:
        with st.spinner('답변 작성 중'):
            from langchain.chains import RetrievalQA
            from langchain_openai import ChatOpenAI
            #from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
            #llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
            chat_box = st.empty()
            stream_handler = StreamHandler(chat_box)
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1, streaming=True, callbacks=[stream_handler])
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())
            qa_chain ({"query":question})
