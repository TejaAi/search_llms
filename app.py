import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve API keys from Streamlit secrets or fallback to user input
groq_api_key = st.secrets.get("GROQ_API_KEY", "")
hf_token = st.secrets.get("HF_TOKEN", "")
langchain_key = st.secrets.get("LANGCHAIN_API_KEY", "")

st.sidebar.title("Settings")
if not groq_api_key:
    groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
if not groq_api_key:
    st.error("Groq API Key is required to run the chatbot.")

# Initialize Search Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# Display App Header
st.title("🔎 LangChain - AI Search Chatbot")
st.write("Chat with an AI-powered search engine that can retrieve information from Wikipedia, Arxiv papers, and the web!")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm a search bot powered by AI. What can I find for you today?"}
    ]

# Display Chat Messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Capture User Input
prompt = st.chat_input("Ask me anything...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize Chat Model
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", streaming=True)

    # Initialize LangChain Agent
    tools = [search, arxiv, wiki]
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    # Use Streamlit Callback Handler
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st)  # FIX: Correct initialization
        try:
            response = search_agent.run(prompt, callbacks=[st_cb])  # FIX: Ensure `prompt` is passed correctly
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.messages.append({"role": "assistant", "content": "Oops! Something went wrong."})
