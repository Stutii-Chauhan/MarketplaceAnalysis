import streamlit as st

# --- Page Config ---
st.set_page_config(page_title="MarketBuzz", layout="wide")

logo_url = "https://github.com/Stutii-Chauhan/MarketplaceAnalysis/blob/f5af9c1362284e88f145eb8b278b0537f10f1904/titan%20logo.png"

logo_html = f"""
<style>
#company-logo {{
    position: absolute;
    top: 1px;
    right: 20px;
    z-index: 9999;
}}
#company-logo img {{
    max-height: 150px;
    max-width: 200px;
    border-radius: 5px;
    object-fit: contain;
}}
</style>
<div id="company-logo">
    <img src="{logo_url}" alt="Company Logo">
</div>
"""

st.markdown(logo_html, unsafe_allow_html=True)

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
