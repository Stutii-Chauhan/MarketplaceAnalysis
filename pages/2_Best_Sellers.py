import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# ---- Supabase DB Connection ----
DB = st.secrets["SUPABASE_DB"]
USER = st.secrets["SUPABASE_USER"]
PASSWORD = quote_plus(st.secrets["SUPABASE_PASSWORD"])
HOST = st.secrets["SUPABASE_HOST"]
PORT = st.secrets["SUPABASE_PORT"]
# engine = create_engine(f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}")
engine = create_engine(
    f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}",
    connect_args={"client_encoding": "utf8"}
)


@st.cache_data(ttl=600)
def load_data(table_name):
    df = pd.read_sql_table(table_name, con=engine)

    if "price" in df.columns:
        df["price"] = (
            pd.to_numeric(
                df["price"].astype(str).str.replace(",", "").str.strip(),
                errors="coerce"
            ).fillna(0).astype("Int64")
        )
    return df

def clean_filter_options(series):
    return sorted([
        val for val in series.dropna().unique()
        if str(val).strip().upper() not in ["NA", "N/A", "NONE", ""]
    ])

def render_best_sellers(gender):
    st.title(f" Best Sellers for {gender}")
    table = "scraped_data_cleaned_men" if gender == "Men" else "scraped_data_cleaned_women"
    df = load_data(table)

    if "filtered_df" in st.session_state:
        render_results(st.session_state.filtered_df)

    st.sidebar.header("Filter Products")

    # 1. Price Band (Checkboxes)
    st.sidebar.markdown("**Price Band**")
    df["price_band"] = df["price_band"].str.strip().str.upper()
    price_band_options = clean_filter_options(df["price_band"])
    selected_priceband = []
    for band in price_band_options:
        if st.sidebar.checkbox(band, key=f"price_band_{band}"):
            selected_priceband.append(band)

    # 2. Price Range Slider
    price_min, price_max = int(df["price"].min()), int(df["price"].max())
    selected_price = st.sidebar.slider("Price Range", price_min, price_max, (price_min, price_max))

    # 3. Brand
    df["brand"] = df["brand"].str.strip().str.lower().str.title()
    valid_brands = clean_filter_options(df["brand"])
    selected_brands = st.sidebar.multiselect("Brand", valid_brands)

    # 4. Dial Colour
    df["dial_colour"] = df["dial_colour"].str.strip().str.lower().str.title()
    valid_dialcol = clean_filter_options(df["dial_colour"])
    selected_dialcol = st.sidebar.multiselect("Dial Colour", valid_dialcol)

    # 5. Dial Shape
    df["case_shape"] = df["case_shape"].str.strip().str.lower().str.title()
    valid_dialshape = clean_filter_options(df["case_shape"])
    selected_dialshape = st.sidebar.multiselect("Dial Shape", valid_dialshape)

    #6. Case thickness
    df["case_thickness"] = df["case_thickness"].str.strip().str.lower().str.title()
    valid_thickness = clean_filter_options(df["case_thickness"])
    selected_thickness = st.sidebar.multiselect("Case Thickness", valid_thickness)

    #7. Case diameter
    df["case_diameter"] = df["case_diameter"].str.strip().str.lower().str.title()
    valid_diameter = clean_filter_options(df["case_diameter"])
    selected_diameter = st.sidebar.multiselect("Case Diameter", case_diameter)

    #8. Band Colour
    df["band_colour"] = df["band_colour"].str.strip().str.lower().str.title()
    valid_bandcol = clean_filter_options(df["band_colour"])
    selected_bandcol = st.sidebar.multiselect("Band Colour", valid_bandcol)

    #9. Band Material
    df["band_material"] = df["band_material"].str.strip().str.lower().str.title()
    valid_bandmaterial = clean_filter_options(df["band_material"])
    selected_bandmaterial = st.sidebar.multiselect("Band Material", valid_bandmaterial)

    #10. Movement
    df["movement"] = df["movement"].str.strip().str.lower().str.title()
    valid_movement = clean_filter_options(df["movement"])
    selected_movement = st.sidebar.multiselect("Movement", valid_movement)

    #11. Water Resistance
    df["water_resistance_depth"] = df["water_resistance_depth"].str.strip().str.lower().str.title()
    valid_waterresistance = clean_filter_options(df["water_resistance_depth"])
    selected_waterresistance = st.sidebar.multiselect("Water Resistance", valid_waterresistance)

    # Apply filters
    filtered_df = df.copy()
    if selected_priceband:
        filtered_df = filtered_df[filtered_df["price_band"].isin(selected_priceband)]
    filtered_df = filtered_df[
        (filtered_df["price"] >= selected_price[0]) & (filtered_df["price"] <= selected_price[1])
    ]
    if selected_brands:
        filtered_df = filtered_df[filtered_df["brand"].isin(selected_brands)]
    if selected_dialcol:
        filtered_df = filtered_df[filtered_df["dial_colour"].isin(selected_dialcol)]
    if selected_bandcol:
        filtered_df = filtered_df[filtered_df["band_colour"].isin(selected_bandcol)]
    if selected_dialshape:
        filtered_df = filtered_df[filtered_df["case_shape"].isin(selected_dialshape)]
    if selected_bandmaterial:
        filtered_df = filtered_df[filtered_df["band_material"].isin(selected_bandmaterial)]
    if selected_movement:
        filtered_df = filtered_df[filtered_df["movement"].isin(selected_movement)]
    if selected_waterresistance:
        filtered_df = filtered_df[filtered_df["water_resistance_depth"].isin(selected_waterresistance)]
    if selected_thickness:
        filtered_df = filtered_df[filtered_df["case_thickness"].isin(selected_thickness)]
    if selected_diameter:
        filtered_df = filtered_df[filtered_df["case_diameter"].isin(selected_diameter)]

    # --- Drop rows with missing critical fields before pagination ---
    filtered_df = filtered_df.dropna(subset=["product_name", "url", "imageurl", "price"], how="any")
    
    # --- Pagination Setup ---
    items_per_page = 12
    total_items = len(filtered_df)
    total_pages = (total_items - 1) // items_per_page + 1
    if "page_number" not in st.session_state:
        st.session_state.page_number = 1
    
    start_idx = (st.session_state.page_number - 1) * items_per_page
    end_idx = start_idx + items_per_page
    paged_df = filtered_df.iloc[start_idx:end_idx]
    
    if paged_df.empty:
        st.warning("No products found with selected filters.")
    else:
        st.markdown(f"**Showing {start_idx + 1}–{min(end_idx, total_items)} of {total_items} products**")
    
        rows = list(paged_df.iterrows())
        for i in range(0, len(rows), 3):
            with st.container():
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(rows):
                        _, row = rows[i + j]
    
                        # Skip any row with essential nulls (safety check)
                        if pd.isna(row["product_name"]) or pd.isna(row["url"]) or pd.isna(row["imageurl"]) or pd.isna(row["price"]):
                            continue
    
                        with cols[j]:
                            st.markdown(
                                f"""
                                <div style="border:1px solid #ddd; padding:20px; border-radius:10px;
                                            box-shadow:0 2px 10px rgba(0,0,0,0.05); height:540px;
                                            background-color:white; display:flex; flex-direction:column;
                                            justify-content:space-between; width:100%;">
                                    <div style='text-align:center'>
                                        <a href="{row['url']}" target="_blank">
                                            <img src="{row['imageurl']}" style="height:250px; width:250px; object-fit:contain; object-position:center;display:block; margin:auto; margin-bottom:15px;"/>
                                        </a>
                                    </div>
                                    <div style="font-weight:600; font-size:1rem; margin-bottom:10px;
                                                display: -webkit-box;
                                                -webkit-line-clamp: 2;
                                                -webkit-box-orient: vertical;
                                                overflow: hidden;
                                                text-align:center;
                                                height:3em;">
                                        {row['product_name']}
                                    </div>
                                    <div style="font-size:0.95rem; line-height:1.6; text-align:left;">
                                        <b>Brand:</b> {row.get('brand', 'N/A')}<br>
                                        <b>Model:</b> {row.get('model_number', 'N/A')}<br>
                                        <b>Price:</b> ₹{int(row['price'])}<br>
                                        <b>Discount:</b> {
                                            (
                                                f"{int(discount)}%" if float(discount).is_integer()
                                                else f"{round(float(discount), 1)}%"
                                            ) if pd.notna((discount := row.get("discount_percentage"))) and float(discount) != 0
                                            else "N/A"
                                        }<br>
                                        <b>Rating:</b> {
                                            (
                                                f"{int(rating)}/5" if float(rating).is_integer()
                                                else f"{round(float(rating), 1)}/5"
                                            ) if pd.notna((rating := row.get("ratings")))
                                                and str(rating).replace('.', '', 1).isdigit()
                                                and float(rating) != 0
                                            else "N/A"
                                        }<br>
                                        <b>Number of reviews:</b> {row.get('number_of_ratings', 'N/A')}<br>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
            st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

        # --- Pagination Controls ---
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.form("pagination_form", clear_on_submit=True):
            col1, col2, col3 = st.columns([1, 8, 1])
            clicked_page = None
        
            with col1:
                if st.session_state.page_number > 1:
                    if st.form_submit_button("⬅️ Prev"):
                        st.session_state.page_number -= 1
        
            with col2:
                current = st.session_state.page_number
                window = 2
                total_pages = (total_items - 1) // items_per_page + 1
        
                page_buttons = [1]
                if current - window > 2:
                    page_buttons.append("...")
        
                for p in range(max(2, current - window), min(current + window + 1, total_pages)):
                    page_buttons.append(p)
        
                if current + window < total_pages - 1:
                    page_buttons.append("...")
        
                if total_pages != 1:
                    page_buttons.append(total_pages)
        
                button_cols = st.columns(len(page_buttons))
                for idx, p in enumerate(page_buttons):
                    if p == "...":
                        button_cols[idx].markdown("**...**")
                    elif p == current:
                        button_cols[idx].form_submit_button(f"• {p} •", disabled=True)
                    else:
                        if button_cols[idx].form_submit_button(str(p)):
                            st.session_state.page_number = p
        
            with col3:
                if st.session_state.page_number < total_pages:
                    if st.form_submit_button("Next ➡️"):
                        st.session_state.page_number += 1


# ---- Main UI ----
st.set_page_config(page_title="Best Sellers", layout="wide", page_icon="📦")
# st.title(f" Best Sellers for {gender}")

if "selected_gender" not in st.session_state:
    st.session_state.selected_gender = "Men"

st.sidebar.markdown("### Gender Category")
st.sidebar.radio("Select the Gender", ["Men", "Women"], key="selected_gender")

render_best_sellers(st.session_state.selected_gender)
