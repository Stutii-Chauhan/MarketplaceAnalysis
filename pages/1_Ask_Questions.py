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
COMMON_WATCH_SCHEMA = {
    "url": "text",
    "brand": "text",
    "product_name": "text",
    "model_number": "text",
    "model_year": "int",
    "price": "float",
    "rating(out_of_5)": "float",
    "discount_(%)": "float",
    "band_colour": "text",
    "band_material": "text",
    "band_width": "text",
    "case_diameter": "text",
    "case_material": "text",
    "case_thickness": "text",
    "dial_colour": "text",
    "crystal_material": "text",
    "case_shape": "text",
    "movement": "text",
    "water_resistance_depth": "text",
    "special_features": "text",
    "image": "text",
    "imageurl": "text",
    "price_band": "text"
}

TABLE_SCHEMAS = {
    "scraped_data_cleaned": {
        "file": "int",
        **COMMON_WATCH_SCHEMA,
        "gender": "text",
        "as_of_date": "date"
    },
    "scraped_data_cleaned_men": {
        "file": "int",
        **COMMON_WATCH_SCHEMA,
        "as_of_date": "date"
    },
    "scraped_data_cleaned_women": {
        "file": "int",
        **COMMON_WATCH_SCHEMA,
        "as_of_date": "date"
    },
    "final_watch_dataset_men_output_rows": {
        k: v for k, v in COMMON_WATCH_SCHEMA.items()
        if k not in ["file", "gender", "as_of_date", "price_band", "model_year"]
    },
    "final_watch_dataset_women_output_rows": {
        k: v for k, v in COMMON_WATCH_SCHEMA.items()
        if k not in ["file", "gender", "as_of_date", "price_band", "model_year"]
    }
}

def detect_table_name(sql_query):
    match = re.search(r"FROM\s+([a-zA-Z0-9_]+)", sql_query, re.IGNORECASE)
    return match.group(1).strip() if match else "scraped_data_cleaned"

def enforce_case_insensitivity(sql_query, table_name):
    sql_query = sql_query.replace("= \"", "= '").replace("\"", "'")

    if table_name not in TABLE_SCHEMAS:
        return sql_query  # fail-safe

    text_columns = [col for col, dtype in TABLE_SCHEMAS[table_name].items() if dtype == "text"]

    for col in text_columns:
        # Equality match: column = 'value'
        pattern = rf"\b{col}\s*=\s*'([^']+)'"
        matches = re.findall(pattern, sql_query, flags=re.IGNORECASE)
        for match in matches:
            fixed = f"LOWER({col}) = '{match.lower()}'"
            sql_query = re.sub(rf"\b{col}\s*=\s*'[^']+'", fixed, sql_query, flags=re.IGNORECASE)

        # IN clause: column IN ('A', 'B')
        pattern_in = rf"\b{col}\s+IN\s*\(([^)]+)\)"
        matches_in = re.findall(pattern_in, sql_query, flags=re.IGNORECASE)
        for match in matches_in:
            values = [v.strip().strip("'").strip('"').lower() for v in match.split(",")]
            fixed = f"LOWER({col}) IN ({', '.join([f'\'{v}\'' for v in values])})"
            sql_query = re.sub(rf"\b{col}\s+IN\s*\([^)]+\)", fixed, sql_query, flags=re.IGNORECASE)

    return sql_query


# ---- Helper Functions ----
def generate_schema_prompt():
    return "\n".join([
        f"- {table}: [{', '.join([f'{col} ({dtype})' for col, dtype in cols.items()])}]"
        for table, cols in TABLE_SCHEMAS.items()
    ])

def generate_sql_with_context(chat_history):
def generate_sql_with_context(chat_history):
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    table_guidance = """
Use these rules to choose the correct table and columns:

scraped_data_cleaned Table Overview:
This is the master table with full product listings. Column descriptions:
- "brand" — the brand of the watch
- "product_name" — product's unique name
- "model_number" — model ID
- "model_year" — year the watch was launched on Amazon
- "price" — selling price
- "rating(out_of_5)" — customer rating out of 5
- "discount_(%)" — discount percentage
- "band_colour" — color of the strap/band
- "band_material" — material of the strap (e.g., Leather, Stainless Steel, Rubber)
- "band_width" — width of the band in mm
- "case_diameter" — diameter of the watch case in mm
- "case_material" — material of the case (e.g., Stainless Steel, Brass)
- "case_thickness" — thickness of the case in mm
- "dial_colour" — dial color
- "crystal_material" — material of the crystal (e.g., Mineral, Sapphire)
- "case_shape" — shape of the case
- "movement" — watch movement type (e.g., Quartz, Automatic)
- "water_resistance_depth" — water resistance in meters
- "special_features" — extra features
- "image", "imageurl" — product images
- "price_band" — price bucket (e.g., 10K–15K, 25K–40K, etc.)
- "gender" — target audience (Men, Women, Unisex, Couple)
- "as_of_date" — when the data was last loaded

Price Range logic:
- If the user's question includes price bands like "10k–15k", "15k–25k", "25k–40k", or "40k+", refer to the `price_band` column instead of using numeric price ranges.
- Match `price_band` = '10K–15K' for 10000–15000
- Match `price_band` = '15K–25K' for 15000–25000
- Match `price_band` = '25K–40K' for 25000–40000
- Match `price_band` = '40K+' for 40000 and above
- Do not use numeric BETWEEN for price when price_band exists
- "-" in  "10k–15k", "15k–25k", "25k–40k" is hyphen and not dash

Table Selection Rules:
- Use `scraped_data_cleaned` for all general queries unless best sellers are explicitly mentioned.
- Use `final_watch_dataset_men_output_rows` if the question refers to best sellers for men.
- Use `final_watch_dataset_women_output_rows` if the question refers to best sellers for women.
- Both best seller tables have the **same structure and column definitions** as `scraped_data_cleaned`, so use the same schema description when writing queries for them.

Material-related Column Disambiguation:
If the user's query contains materials (e.g., "stainless steel", "leather", "rubber"), choose the appropriate column:
- Use `band_material` if the question includes terms like "strap", "band", or "bracelet"
- Use `case_material` if it includes "case", "body", or "watch material"
- Use `crystal_material` if it includes "crystal", "glass", "sapphire", or "mineral"
- If no body part is specified, default to `case_material`

Text based filters:
- The text columns are stored in sentence case always. Follow this while writing queries.

Follow-Up Handling:
- For follow-up questions, retain previously used filters or table if the user does not explicitly change them.
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

        #Fix smart quotes and EN DASH
        sql = (
            sql
            .replace("–", "-")  # EN DASH → hyphen
            .replace("‘", "'").replace("’", "'")
            .replace("“", '"').replace("”", '"')
        )

        # Detect table and apply case-insensitive fix
        table_name = detect_table_name(sql)
        sql = enforce_case_insensitivity(sql, table_name)

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
    
    with st.spinner("Buzz is thinking..."):
        try:
            sql_query = generate_sql_with_context(st.session_state.chat_history)
            sql_query = sql_query.replace("–", "-").replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
            st.session_state.chat_history.append({"role": "assistant", "content": sql_query})
            st.session_state.last_sql = sql_query

            # ✅ Run the query immediately
            df_result = pd.read_sql_query(sql_query, engine)
            st.session_state.query_result = df_result



            # ✅ Show preview immediately
            st.markdown("### 📋 Query Output")
            if len(df_result) == 0:
                st.info("No results found.")
            elif df_result.shape[1] == 1:
                st.success(f"✅ Result: `{df_result.iloc[0,0]}`")
            else:
                st.dataframe(df_result.head())

        except Exception as e:
            st.error(f"❌ Failed to execute query: {e}")


# st.markdown("### 🧠 Chat History")
chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f" **You:** {msg['content']}")
        else:
            st.markdown(f" **Buzz:** `{msg['content']}`")

if st.session_state.last_table:
    st.caption(f"📌 Last table used: `{st.session_state.last_table}`")
