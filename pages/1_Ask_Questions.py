import streamlit as st
import pandas as pd
import google.generativeai as genai
from sqlalchemy import create_engine
import plotly.express as px
from urllib.parse import quote_plus
import re

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

# ---- Table Schemas ---
COMMON_WATCH_SCHEMA = {
    "url": "text",
    "brand": "text",
    "product_name": "text",
    "model_number": "text",
    "model_year": "int",
    "price": "float",
    "ratings": "float",
    "discount_percentage": "float",
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
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    table_guidance = """
Use these rules to choose the correct table and columns:

scraped_data_cleaned Table Overview:
This is the master table with full product listings. Column descriptions:
- "brand" — the brand of the watch
- "product_name" — product's unique name
- "model_number" — model ID or SKU code
- "model_year" — year the watch was launched on Amazon
- "price" — selling price
- "ratings" — customer rating out of 5
- "discount_percentage" — discount percentage
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

SKUs is defined as the models/products and SKU code is the model code/ product code

Brand Matching Rules:
- The brand column contains full names like:
  - "titan", "titan raga", "titan edge", "titan xylys"
  - "tommy hilfiger", "armani exchange", "michael kors", "daniel wellington", etc.

- Use the following mappings when interpreting user queries:
  - "raga" → 'titan raga'
  - "edge" → 'titan edge'
  - "xylys" → 'titan xylys'
  - "titan" → 'titan'
  - "tommy" → 'tommy hilfiger'
  - "armani" → 'armani exchange'
  - "dw" → 'daniel wellington'
  - "kors" → 'michael kors'

- Always match brand using:
  ```sql
  LOWER(brand) = '<mapped_full_name>'

Price Range logic:
- There are two types of price references in user queries:
  1. **Predefined Price Bands** → Use the `price_band` column
     - Examples: "10K–15K", "15K–25K", "25K–40K", "40K+" (these are exact predefined bands)
     - Match using: LOWER(price_band) = '10k–15k' etc.
  2. **Custom Price Filters** → Use the numeric `price` column
     - Examples: "below 12000", "between 10k and 12k", "under 18k", "greater than 25k", "less than 9500"
     - Interpret "K" or "k" as 1000 (e.g., 10k = 10000)
     - Use SQL filters like: `price BETWEEN 10000 AND 12000`, `price < 18000`, etc.

- Important:
  - Only use `price_band` if the exact band like '10K–15K' is clearly mentioned.
  - If the price range is custom or approximate (like “under 10k” or “between 8k and 12k”), use the numeric `price` column.
  - Convert “10k”, “25K” etc. to thousands: 10k = 10000

Dominance and Table Selection Rules:

- Treat “dominant”, “top”, or “popular” brands as the top 5 brands by frequency in the `scraped_data_cleaned` table.
- If the user says **only "top"** (without words like "selling", "sellers", or "best"), use the `scraped_data_cleaned` table.
- If the query mentions **“best”, “top seller”, “top selling”, “best seller”, or “best selling”**, then:
  - Use `scraped_data_cleaned_men` if the query refers to men
  - Use `scraped_data_cleaned_women` if the query refers to women

Table selection:

- Use `scraped_data_cleaned` for general queries (including “top”, “dominant”, or “popular” products or brands).
- Use scraped_data_cleaned_men if the query includes phrases like “best sellers for men” or “top selling men’s watches”.
- Use scraped_data_cleaned_women if the query includes phrases like “best sellers for women” or “top selling women’s watches”.
- These gender-specific tables already reflect best-selling products — **do not apply additional gender filters** when using them.
- All three tables share the same schema.

Ranking logic:

- The `file` column represents product rank across all tables — lower `file` value = better ranking (e.g., `file = 1` is the top seller).
- For top/best-selling queries, sort using `ORDER BY file ASC`.
- For general “top” queries or preference-based sorts (e.g., ratings, price, etc.), use the appropriate `ORDER BY` logic based on context.

Material-related Column Disambiguation:
If the user's query contains materials (e.g., "stainless steel", "leather", "rubber"), choose the appropriate column:
- Use `band_material` if the question includes terms like "strap", "band", or "bracelet"
- Use `case_material` if it includes "case", "body", or "watch material"
- Use `crystal_material` if it includes "crystal", "glass", "sapphire", or "mineral"
- If no body part is specified, default to `case_material`

Text based filters:
- The text columns are stored in sentence case always. Follow this while writing queries.

Dominance selection:
- Treat "dominant" as the top 5 brands based on their frequency in the scraped_data_cleaned table.
- Do not switch to any best seller tables unless the user explicitly mentions "best sellers" or similar terms.
- Default to scraped_data_cleaned for general queries, including "top", "popular", or "dominant" brands.

Chart Generation Rules:

- If the user asks to "plot", "graph", "visualize", or "draw a chart", assume they want a visual output.
- Use only numeric columns from the query result for plotting (e.g., `price`, `ratings`, `discount_percentage`, etc.).
- When generating SQL for charts, ensure the query includes at least two numeric columns for plotting.
- Do not generate charts unless the user explicitly asks for a plot/visual.
- If the user specifies which columns to plot (e.g., “price vs ratings”), the SQL should include both columns.
- When generating a chart, always include a line at the top of the SQL query like `-- chart: bar`, `-- chart: scatter`, or `-- chart: line` to indicate the desired chart type.
- Determine chart type based on user language and selected columns:

  - If the user says “relationship”, “correlation”, “compare two numeric values”, or “scatter plot” → use **scatter plot**
  - If the user says “distribution”, “spread”, “histogram” → use **histogram**
  - If the user says “bar chart”, “compare brands”, or mentions **category vs. number** (e.g., brand vs price) → use **bar chart**
  - If the user says “trend”, “over time”, “timeline”, or includes a **date column** → use **line chart**
  - If the user says “share”, “proportion”, “percentage”, or mentions “parts of whole” → use **pie chart** (if categorical distribution)

- Include appropriate columns in the SQL query to match the chart type:
  - For scatter: 2 numeric columns
  - For bar: 1 categorical + 1 numeric (use `GROUP BY`)
  - For line: date/time column + metric
  - For pie: 1 categorical + `COUNT(*)` or `SUM(value)`


- If only one numeric column is relevant, the chart logic will not run — fall back to a table or summary.

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
Only return the SQL. Do not explain. Do not format it as a Python object or JSON.
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

# ---- Reset Button ----
if st.button("🔁 Reset Chat", type="primary"):
    st.session_state.chat_history = []
    st.session_state.query_result = None
    st.session_state.last_sql = ""
    st.session_state.last_table = ""
    st.rerun()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Overview of Result")
    if st.session_state.query_result is not None and not st.session_state.query_result.empty:
        st.dataframe(st.session_state.query_result)

def detect_chart_type(sql):
    match = re.search(r"--\s*chart:\s*(\w+)", sql, re.IGNORECASE)
    return match.group(1).strip().lower() if match else "scatter"

with col2:
    st.subheader("📊 Chart Plot")

    if st.session_state.query_result is not None and not st.session_state.query_result.empty:
        df = st.session_state.query_result.copy()

        # Get numeric columns first
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        # Filter out rows with 0 in any numeric column
        if numeric_cols:
            df = df[(df[numeric_cols] != 0).all(axis=1)]

        chart_type = detect_chart_type(st.session_state.last_sql)

        if len(numeric_cols) >= 2 or chart_type in ["bar", "line"]:
            try:
                if chart_type == "bar":
                    fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=f"{df.columns[0]} vs {df.columns[1]}")
                elif chart_type == "line":
                    fig = px.line(df, x=df.columns[0], y=df.columns[1], title=f"{df.columns[0]} vs {df.columns[1]}")
                else:
                    fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=f"{numeric_cols[0]} vs {numeric_cols[1]}")

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"Could not render chart: {e}")
        else:
            st.info("Not enough numeric columns to display a chart.")


# ---- Chat Interface ----
st.markdown("---")
st.markdown("""
<div style='background-color:#f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
    <h3 style='margin-bottom: 10px;'>💬 Chat with Buzz</h3>
</div>
""", unsafe_allow_html=True)

# ✅ Use a form to submit new input only once
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask your question here", key="chat_input_internal")
    submitted = st.form_submit_button("Send")

    if submitted and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("Buzz is thinking..."):
            try:
                sql_query = generate_sql_with_context(st.session_state.chat_history)
                sql_query = sql_query.replace("–", "-").replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
                st.session_state.last_sql = sql_query

                # ✅ Run the SQL query
                df_result = pd.read_sql_query(sql_query, engine)

                # ✅ Save assistant response
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": sql_query,
                    "result": df_result
                })

                # ✅ Update preview table result
                st.session_state.query_result = df_result.copy()

                if df_result.empty:
                    st.info("No results found.")

            except Exception as e:
                st.error(f"❌ Failed to execute query: {e}")



# ---- Chat History Display ----
chat_container = st.container()
with chat_container:
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div style='background-color:#e7f3ff; padding:10px; border-radius:8px; margin-bottom:5px;'>
                    <strong>User:</strong> {msg['content']}
                </div>
                """, unsafe_allow_html=True
            )

        elif msg["role"] == "assistant":
            st.markdown("**Buzz (SQL):**")
            st.code(msg["content"], language="sql")

            result = msg.get("result")

            # ✅ Case 1: 1x1 DataFrame (e.g., COUNT(*))
            if isinstance(result, pd.DataFrame) and result.shape == (1, 1):
                value = result.iloc[0, 0]
                st.markdown(
                    f"""
                    <div style='background-color:#e8f8e4; padding:10px; border-radius:8px; margin-top:-10px;'>
                        <strong>Buzz (Result):</strong> {value}
                    </div>
                    """, unsafe_allow_html=True
                )

            # ✅ Case 2: DataFrame with 1 column
            elif isinstance(result, pd.DataFrame) and result.shape[1] == 1:
                col = result.columns[0]
                values = result[col].dropna().astype(str).tolist()
                bullet_values = values[:10]
                bullets_html = "".join([f"<li>{val}</li>" for val in bullet_values])

                st.markdown(
                    f"""
                    <div style='background-color:#f0fdf4; padding:10px; border-radius:8px; margin-top:-10px;'>
                        <strong>Buzz ({col.title()}s):</strong>
                        <ul>{bullets_html}</ul>
                    </div>
                    """, unsafe_allow_html=True
                )

            # ✅ Case 3: DataFrame with >1 column
            elif isinstance(result, pd.DataFrame) and result.shape[1] > 1:
                st.markdown(
                    f"""
                    <div style='background-color:#fffaf0; padding:10px; border-radius:8px; margin-top:-10px; margin-bottom:10px;'>
                        <strong>Buzz (Preview):</strong>
                    </div>
                    """, unsafe_allow_html=True
                )
                st.dataframe(result, use_container_width=True,height=170)

if st.session_state.chat_history:
    last_msg = st.session_state.chat_history[-1]
    if last_msg["role"] == "assistant" and isinstance(last_msg.get("result"), pd.DataFrame):
        st.session_state.query_result = last_msg["result"]

# if st.session_state.last_table:
#     st.caption(f"📌 Last table used: `{st.session_state.last_table}`")
