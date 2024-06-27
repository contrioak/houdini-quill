import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain import hub
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

###### MULTI SEARCH RAG AGENT START 

# os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# # WIKIPEDIA TOOL

# wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max = 200)
# wikipedia_tool = WikipediaQueryRun(api_wrapper = wiki_api_wrapper)


# # ARXIV TOOL 

# arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
# arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

# # LANGSMITH TOOL

# web_loader = WebBaseLoader("https://docs.smith.langchain.com/")
# web_docs = web_loader.load()
# web_documents = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200).split_documents(web_docs)
# vectordb = FAISS.from_documents(web_documents,OpenAIEmbeddings())
# retriever = vectordb.as_retriever()
# langsmith_retriever_tool = create_retriever_tool(retriever,"langsmith_search", 
#                     "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!")


# # COMBINING TOOLS 

# tools = [wikipedia_tool, arxiv_tool, langsmith_retriever_tool]

# # CREATING AGENT 

# # LLM
# llm = ChatOpenAI(model = "gpt-3.5-turbo-0125", temperature = 0)

# # PROMPT
# prompt = hub.pull("hwchase17/openai-functions-agent")
# prompt.messages

# # AGENT AND AGENT EXECUTOR
# agent = create_openai_tools_agent(llm, tools, prompt)
# agent_executor = AgentExecutor(agent = agent, tools = tools, verbose = True)

###### MULTI SEARCH RAG AGENT END  

############################### PDF AGENT ####################################
def get_documents(pdf_documents):
    text = ""
    for doc in pdf_documents:
        pdf = PdfReader(doc)
        for page in pdf.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100000, chunk_overlap = 700)
    chunk = text_splitter.split_text(text)
    return chunk

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)
    vector_store.save_local("faiss_vector_index")

def get_conversation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    llm  = ChatGoogleGenerativeAI(model = "gemini-pro", temperature = 0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(llm, chain_type = "stuff", prompt = prompt)
    return chain

def user_input(user_query):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    tmp_db = FAISS.load_local("faiss_vector_index", embeddings, allow_dangerous_deserialization=True)
    docs = tmp_db.similarity_search(user_query)
    chain = get_conversation_chain()
    response = chain(
        {"input_documents": docs, "question": user_query}, return_only_outputs = True
    )
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(
        page_title="Interact with Content",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    with st.sidebar:
        st.title("🪐Houdini🪐")

        # Use radio buttons for clear user selection
        interaction_mode = st.radio(
            "",  # No label, descriptive text within options
            ("PDF Genie", "Multisearch Rag Agent"),
            horizontal=True  # Arrange options horizontally
        )

    # Clear heading based on user selection
    if interaction_mode == "PDF Genie":
        st.header("Your own personal genie!!🤖")

    if interaction_mode == "PDF Genie":
        # User input with a descriptive placeholder
        user_question = st.text_input("Ask Genie any question realted to the PDFs...", placeholder="Ask your quesetion here")

        if user_question:
            user_input(user_question)

        # Enhanced file upload and processing (assuming `get_documents`, etc. are defined elsewhere)
        with st.sidebar:
            st.title("File Upload & Processing")

            pdf_docs = st.file_uploader("Upload your PDF files:", type="pdf", accept_multiple_files=True)

            if pdf_docs:
                # Visually engaging button with an icon
                if st.button("Upload & Process ()", key="process_button"):
                    with st.spinner("Gemini is processing your PDFs..."):
                        raw_text = get_documents(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done! Ask Gemini your questions now.")
    else:
        st.title("Multi-Search RAG Agent : Arxiv, Wikipedia, and Langsmith Search🌐\nOOPS...exausted the OPENAI CREDITS!!")

        # Text box for user query
        user_query = st.text_input("Enter your query:")
        st.write("OOPS...exausted the OPENAI CREDITS!!")
        # Button to trigger agent execution
        if st.button("Ask the Agent"):
          if user_query:
            # Execute agent with user query  
            # response = agent_executor.invoke({"input": user_query})

            # Display response on screen
            st.write(f"**Agent Response:** "{response}")
          else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
