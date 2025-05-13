import os
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import re
import numpy as np
from datetime import datetime

# --------------------------------------
# DB Setup
# --------------------------------------
db = os.environ["SUPABASE_DB"]
user = os.environ["SUPABASE_USER"]
raw_password = os.environ["SUPABASE_PASSWORD"]
host = os.environ["SUPABASE_HOST"]
port = os.environ["SUPABASE_PORT"]
password = quote_plus(raw_password)

engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db}")

#read the table
df = pd.read_sql_table("final_watch_data_cleaned", con=engine)
df.count()

#delete rows with NaN/blank in url column and also in the price Column.
df = df.dropna(subset=["url"])
df = df.dropna(subset=["price"])
df.count()

#Cleaning Model Number column

def clean_and_fix_model_number(row):
    def clean_model(val):
        if pd.isna(val):
            return None
        val = str(val).strip()

        blacklist = ["casual watch", "watches", "combo", "legacy", "pack", "premium", "design", "men", "women"]
        val_l = val.lower()
        if any(x in val_l for x in blacklist) or len(val) <= 3:
            return None

        # Try to extract the model pattern
        match = re.search(r'([A-Z]{1,3}[\dA-Z\-\.]{3,})$', val)
        return match.group(1) if match else None

    # Step 1: Clean existing Model Number
    model_num = clean_model(row.get("model_number"))
    if model_num:
        return model_num

    # Step 2: Fallback to Part Number
    part_num = row.get("part_number")
    if pd.notna(part_num) and str(part_num).strip():
        return str(part_num).strip().upper()

    # Step 3: Try extracting from Product Name
    product_name = row.get("product_name", "")
    if pd.notna(product_name):
        match = re.findall(r'([A-Z]{2,5}[\d]{3,}[A-Z]{0,5})', product_name)
        if match:
            return match[0]

    # Step 4: Fallback to "NA"
    return "NA"

df["model_number"] = df.apply(clean_and_fix_model_number, axis=1)

#delete duplicate values with product_name + model_number
df = df.drop_duplicates(subset=["product_name", "model_number"], keep="first")
df.count()

#price cleaning
df["price"] = (
    df["price"]
    .str.replace(",", "")                   # Remove commas
    .str.replace(r"\.$", "", regex=True)   # Remove trailing dot if exists
    .astype(float)
    .round(2)
)
df = df[df["price"] >= 10000] #removing products with price < 10000
df.count()

#cleaning ratings
df["rating(out_of_5)"] = (
    df["rating(out_of_5)"]
    .str.extract(r"(\d+\.?\d*)")        # extract only the numeric part
    .astype(float)
    .map(lambda x: int(x) if pd.notna(x) and x.is_integer() else round(x, 1) if pd.notna(x) else np.nan)
)

#adding price_band
df["price_band"] = pd.cut(
    df["price"],
    bins=[0, 10000, 15000, 25000, 40000, float("inf")],
    labels=["<10K", "10K-15K", "15K-25K", "25K-40K", "40K+"],
    right=False
)

df["discount_(%)"] = (
    df["discount_(%)"]
    .astype(str)
    .str.extract(r"(\d+\.?\d*)")[0]         # extract numeric part
    .replace("", np.nan)
    .astype(float)
    .map(lambda x: f"{int(x)}%" if x.is_integer() else f"{x}%" if pd.notna(x) else np.nan)
)


def normalize_to_mm(value):
    if pd.isna(value) or str(value).strip() == "":
        return np.nan

    text = str(value).strip().lower()
    match = re.search(r"(\d+\.?\d*)", text)
    if not match:
        return np.nan

    num = float(match.group(1))
    if "cm" in text:
        num *= 10  # convert to millimeters

    return f"{int(num)} Millimeters" if num.is_integer() else f"{num} Millimeters"

df["band_width"] = df["band_width"].apply(normalize_to_mm)
df["case_diameter"] = df["case_diameter"].apply(normalize_to_mm)
df["case_thickness"] = df["case_thickness"].apply(normalize_to_mm)

#removing the unwanted keywords
unwanted_keywords = ["pocket watch", "repair tool", "watch bezel", "watch band", "tool", "watch winder", "watch case"]
df = df[~df["product_name"].str.lower().str.contains('|'.join(unwanted_keywords))]
df.count()


#filling major brand names for the products where brand name is missing

brand_map = {
    "tommy hilfiger": "Tommy Hilfiger",
    "tommy": "Tommy Hilfiger",
    "armani exchange": "Armani Exchange",
    "diesel": "Diesel",
    "fossil": "Fossil",
    "titan": "Titan",
    "casio": "Casio",
    "michael kors": "Michael Kors",
    "maserati": "Maserati",
    "luminox": "Luminox",
    "zeppelin": "Zeppelin",
    "seiko": "Seiko",
    "ted baker": "Ted Baker",
    "invicta": "Invicta",
    "citizen": "Citizen",
    "emporio armani": "Emporio Armani",
    "guess": "Guess",
    "fiece": "Fiece",
    "just cavalli": "Just Cavalli",
    "earnshaw": "Earnshaw",
    "alba": "Alba",
    "daniel wellington": "Daniel Wellington",
    "police": "Police",
    "olevs": "Olevs",
    "ducati": "Ducati",
    "mathey-tissot": "Mathey-Tissot",
    "timex": "Timex",
    "swarovski": "Swarovski",
    "nautica": "Nautica",
    "swiss military hanowa": "Swiss Military Hanowa",
    "lacoste": "Lacoste",
    "boss": "Boss",
    "anne klein": "Anne Klein",
    "calvin klein": "Calvin Klein",
    "pierre cardin": "Pierre Cardin",
    "coach": "Coach",
    "p philip": "P Philip",
    "tag heuer": "Tag Heuer",
    "kenneth cole": "Kenneth Cole",
    "philipp plein": "Philipp Plein",
    "guy laroche": "Guy Laroche",
    "carlos philip": "Carlos Philip",
    "adidas": "Adidas",
    "movado": "Movado",
    "daniel klein": "Daniel Klein",
    "sonata": "Sonata",
    "d1 milano": "D1 Milano",
    "alexandre christie": "Alexandre Christie",
    "santa barbara": "Santa Barbara Polo & Racquet Club",
    "mini cooper": "MINI Cooper",
    "hanowa": "Hanowa",
    "charles-hubert": "Charles-Hubert",
    "gc": "GC"
}

# Lowercase version of product names for pattern matching
df["__product_lower__"] = df["product_name"].str.lower()

# Temporary brand match column (explicitly set as object type to store strings)
df["__brand_match__"] = pd.Series([np.nan] * len(df), dtype="object")

# Match brand keywords to update missing brand values
for keyword, clean_brand in brand_map.items():
    pattern = rf"\b{re.escape(keyword)}\b"
    mask = df["brand"].isna() & df["__product_lower__"].str.contains(pattern, regex=True, na=False)
    df.loc[mask, "__brand_match__"] = clean_brand

# Fill missing brand values with matched brands
df["brand"] = df["brand"].fillna(df["__brand_match__"])

# Still missing? Mark as "NA"
df["brand"] = df["brand"].fillna("NA")

# Drop helper columns
df.drop(columns=["__product_lower__", "__brand_match__"], inplace=True)


#Dividing Titan as Titan, Xylys, Edge and Raga
def categorize_titan(row):
    brand = str(row["brand"]).strip().title()
    product = str(row["product_name"]).strip().title()

    if brand == "Titan":
        if "Xylys" in product:
            return "Titan Xylys"
        elif "Edge" in product:
            return "Titan Edge"
        elif "Raga" in product:
            return "Titan Raga"
        else:
            return "Titan"
    return brand

df["brand"] = df.apply(categorize_titan, axis=1)

#Adding a gender column
def infer_gender(product_name):
    name = str(product_name).lower()

    if "couple" in name:
        return "Couple"
    elif any(word in name for word in ["female", "women", "woman", "girl", "ladies", "women's", "girl's"]):
        return "Women"
    elif any(word in name for word in ["male", "man", "men", "boy", "men's", "boy's"]):
        return "Men"
    else:
        return "Unisex"
df["gender"] = df["product_name"].apply(infer_gender)

#adding as of date
df["as_of_date"] = datetime.today().strftime("%Y-%m-%d")

# Ensure 'file' column is string
df['file'] = df['file'].astype(str)

# Extract numeric part from file name
df['file_num'] = df['file'].str.extract(r'(\d+)').astype(int)

# Sort by numeric value
df = df.sort_values(by='file_num').drop(columns=['file_num'])

# Then upload to Supabase
df.to_sql("scraped_data_cleaned", con=engine, if_exists="replace", index=False)
