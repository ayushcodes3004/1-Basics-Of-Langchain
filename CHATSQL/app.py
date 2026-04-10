import streamlit as st
from pathlib import Path

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_groq import ChatGroq

# -------------------- UI --------------------
st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 Chat with your SQL Database")

# -------------------- CONSTANTS --------------------
LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["Use SQLite (student.db)", "Connect to MySQL"]
selected_opt = st.sidebar.radio("Choose Database", options=radio_opt)

# -------------------- MYSQL INPUT --------------------
mysql_host = None
mysql_user = None
mysql_password = None
mysql_db = None

if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("MySQL Host", value="127.0.0.1")
    mysql_user = st.sidebar.text_input("MySQL User", value="root")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database")
else:
    db_uri = LOCALDB

# -------------------- API KEY --------------------
api_key = st.sidebar.text_input("Groq API Key", type="password")

if not api_key:
    st.warning("Please enter your Groq API Key")
    st.stop()

# -------------------- MODEL --------------------
model_option = st.sidebar.selectbox(
    "Choose Model",
    ["meta-llama/llama-4-scout-17b-16e-instruct"]
)

llm = ChatGroq(
    groq_api_key=api_key,
    model_name=model_option,
    temperature=0
)

# -------------------- DB CONFIG --------------------
@st.cache_resource(ttl=7200)
def configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db):
    if db_uri == LOCALDB:
        db_path = (Path(__file__).parent / "student.db").absolute()
        return SQLDatabase.from_uri(f"sqlite:///{db_path}")

    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            raise ValueError("Missing MySQL credentials")

        return SQLDatabase.from_uri(
            f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
        )

# -------------------- INIT DB --------------------
try:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)

    st.sidebar.markdown("### 🔌 Connection Status")
    db.run("SELECT 1;")
    st.sidebar.success("✅ Database Connected")

    tables = db.get_usable_table_names()
    st.sidebar.write("📊 Tables:", tables if tables else "No tables found")

except Exception as e:
    st.sidebar.markdown("### 🔌 Connection Status")
    st.sidebar.error("❌ Connection Failed")
    st.sidebar.write(str(e))
    st.stop()

# -------------------- AGENT --------------------
agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="tool-calling",
    verbose=True
)

# -------------------- CHAT MEMORY --------------------
if "messages" not in st.session_state or st.sidebar.button("Clear Chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about your database"}
    ]

# -------------------- DISPLAY CHAT --------------------
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -------------------- USER INPUT --------------------
user_query = st.chat_input("Ask your database...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        try:
            callback = StreamlitCallbackHandler(st.container())

            result = agent.invoke(
                {"input": user_query},
                {"callbacks": [callback]}
            )

            output = result.get("output", "No response generated")

        except Exception as e:
            output = f"Error: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": output})
        st.write(output)