import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

# ---------------- TOOLS ---------------- #

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# ---------------- UI ---------------- #

st.title("🔎 Smart Search Chatbot ")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# ---------------- SESSION ---------------- #

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I can search the web, arxiv, and wikipedia. Ask me anything!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------------- CHAT ---------------- #

if user_input := st.chat_input("Ask anything..."):

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
    )

    # -------- TOOL ROUTING (SMART) -------- #

    query = user_input.lower()

    if "arxiv" in query or "research paper" in query:
        tool_result = arxiv.run(user_input)

    elif "wiki" in query or "wikipedia" in query:
        tool_result = wiki.run(user_input)

    else:
        tool_result = search.run(user_input)

    # -------- LLM FINAL RESPONSE -------- #

    final_response = llm.invoke([
        HumanMessage(content=f"""
        Answer the question using the tool result below.

        Question: {user_input}
        Tool Result: {tool_result}

        Give a clear and helpful answer.
        """)
    ])

    output = final_response.content

    st.session_state.messages.append({
        "role": "assistant",
        "content": output
    })

    st.chat_message("assistant").write(output)
