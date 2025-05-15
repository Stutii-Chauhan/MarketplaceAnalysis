import streamlit as st

# --- Page Config ---
st.set_page_config(page_title="MarketBuzz", layout="wide")

# --- Main Heading ---
st.markdown("<h1 style='text-align: center;'> MarketBuzz </h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Description Section ---
st.markdown("""
Welcome to the **MarketBuzz** — your one-stop dashboard to explore:
- **Top-selling analog watches** by brand, price, and specs
- **Competitor benchmarking** using product listings, discounts, and reviews
- **Gender-based product distribution** and **best-performing SKUs**
- AI-powered insights to **ask questions** about the watch data

Use the sidebar to navigate between:
- 📊 **Ask Questions**: Interact with your data using Gemini LLM
- 🏆 **Best Sellers**: Explore top watches with filters and images
""")

st.markdown("---")
st.info("Use the options in the **sidebar** to get started.")
