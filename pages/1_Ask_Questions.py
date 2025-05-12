import streamlit as st
import pandas as pd
import google.generativeai as genai
from sqlalchemy import create_engine
import plotly.express as px
from urllib.parse import quote_plus

# ---- Streamlit Config ----
st.set_page_config(layout="wide")

# ---- Gemini Setup ----
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash-lite")

# ---- Supabase Connection ----
DB = st.secrets["SUPABASE_DB"]
USER = st.secrets["SUPABASE_USER"]
PASSWORD = quote_plus(st.secrets["SUPABASE_PASSWORD"])
HOST = st.secrets["SUPABASE_HOST"]
PORT = st.secrets["SUPABASE_PORT"]
engine = create_engine(f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}")

# ---- Table Schemas ----
TABLE_SCHEMAS = {
    "master_watch_data": [
        "Brand", "Product Name", "Model Number", "Model Year", "Price", "Rating",
        "Discount", "Band Colour", "Band Material", "Band Width", "Case Diameter",
        "Case Material", "Case Thickness", "Dial Colour", "Crystal Material",
        "Case Shape", "Movement", "Gender"
    ],
    "best_sellers_men": [
        "Brand", "Product Name", "Price", "Rating", "Rank", "Gender"
    ],
    "best_sellers_women": [
        "Brand", "Product Name", "Price", "Rating", "Rank", "Gender"
    ]
}

# ---- Helper: Schema String for Prompt ----
def generate_schema_prompt():
    return "\n".join([f"- {table}: [{', '.join(cols)}]" for table, cols in TABLE_SCHEMAS.items()])

# ---- Helper: Generate SQL using chat context ----
def generate_sql_with_context(chat_history):
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    prompt = f"""
You are a SQL expert for a PostgreSQL watch marketplace database.

Here are the available tables and their columns:
{generate_schema_prompt()}

Use this ongoing chat to interpret follow-up questions too.

{context}

Now generate only the correct SQL query. Do not explain or describe.
"""

    response = model.generate_content(prompt)
    return response.text.strip().split("SQL Query:")[-1].strip("```sql").strip("```").strip()

# ---- Session State Init ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_result" not in st.session_state:
    st.session_state.query_result = None

# ---- Layout: Overview + Chart ----
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Overview of Result")
    if st.session_state.query_result is not None:
        st.dataframe(st.session_state.query_result.head())

with col2:
    st.subheader("📊 Chart Plot")
    if st.session_state.query_result is not None:
        numeric_cols = st.session_state.query_result.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) >= 2:
            fig = px.scatter(
                st.session_state.query_result,
                x=numeric_cols[0],
                y=numeric_cols[1],
                title=f"{numeric_cols[0]} vs {numeric_cols[1]}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric columns to display a chart.")

# ---- Chat Interface ----
st.markdown("---")
st.subheader("💬 Chat with Marketplace Analyzer")

user_input = st.text_input("Ask a question about your data")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("Generating SQL..."):
        try:
            sql_query = generate_sql_with_context(st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "assistant", "content": sql_query})
            df_result = pd.read_sql_query(sql_query, engine)
            st.session_state.query_result = df_result
        except Exception as e:
            st.error(f"Error executing query: {e}")

# ---- Chat History ----
with st.expander("📝 Chat History"):
    for msg in st.session_state.chat_history:
        st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")
