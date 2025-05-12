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
    "scraped_data_cleaned": [
        "file", "url", "brand", "product_name", "model_number", "model_year", "price",
        "rating(out_of_5)", "discount_(%)", "band_colour", "band_material", "band_width",
        "case_diameter", "case_material", "case_thickness", "dial_colour", "crystal_material",
        "case_shape", "movement", "water_resistance_de...", "special_features", "image",
        "imageurl", "price_band", "gender", "as_of_date"
    ],
    "final_watch_dataset_women_output_rows": [
        "url", "brand", "product_name", "model_number", "price", "ratings", "discount",
        "band_colour", "band_material", "band_width", "case_diameter", "case_material",
        "case_thickness", "dial_colour", "crystal_material", "case_shape", "movement",
        "water_resistance_depth", "special_features", "imageurl", "image"
    ],
    "final_watch_dataset_men_output_rows": [
        "url", "brand", "product_name", "model_number", "price", "ratings", "discount",
        "band_colour", "band_material", "band_width", "case_diameter", "case_material",
        "case_thickness", "dial_colour", "crystal_material", "case_shape", "movement",
        "water_resistance_depth", "special_features", "imageurl", "image"
    ]
}

# ---- Helper Functions ----
def generate_schema_prompt():
    return "\n".join([f"- {table}: [{', '.join(cols)}]" for table, cols in TABLE_SCHEMAS.items()])

def generate_sql_with_context(chat_history):
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    table_guidance = """
Use these rules to choose the correct table:
- Use `scraped_data_cleaned` for all queries unless best sellers are mentioned explicitly.
- Use `final_watch_dataset_men_output_rows` 
- Use `final_watch_dataset_women_output_rows` for women-specific ones.
- Reuse previous context if question is a follow-up.
"""

    prompt = f"""
You are a PostgreSQL SQL expert working with watch data.

Available table schemas:
{generate_schema_prompt()}

{table_guidance}

Conversation so far:
{context}

Now generate a valid SQL query for the user's most recent question.
Only return the SQL. Do not explain.
"""

    try:
        response = model.generate_content(prompt)
        sql = response.text.strip().split("```sql")[-1].strip("```").strip()
        return sql
    except Exception as e:
        return f"Gemini failed: {e}"

# ---- Session State Init ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_result" not in st.session_state:
    st.session_state.query_result = None
if "last_sql" not in st.session_state:
    st.session_state.last_sql = ""
if "last_table" not in st.session_state:
    st.session_state.last_table = ""

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
        sql_query = generate_sql_with_context(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": sql_query})
        st.session_state.last_sql = sql_query

        try:
            df_result = pd.read_sql_query(sql_query, engine)
            st.session_state.query_result = df_result

            # Try to extract table name from SQL (best effort)
            for table in TABLE_SCHEMAS.keys():
                if table.lower() in sql_query.lower():
                    st.session_state.last_table = table
                    break
        except Exception as e:
            st.error(f"Error executing query: {e}")

# ---- Chat History + Debug ----
with st.expander("📝 Chat History"):
    for msg in st.session_state.chat_history:
        st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")

if st.session_state.last_table:
    st.caption(f"📌 Last table used: `{st.session_state.last_table}`")
