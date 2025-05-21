import os
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import re
import numpy as np
from datetime import datetime

# ------------------------------------
# DB Setup
# ------------------------------------
db = os.environ["SUPABASE_DB"]
user = os.environ["SUPABASE_USER"]
raw_password = os.environ["SUPABASE_PASSWORD"]
host = os.environ["SUPABASE_HOST"]
port = os.environ["SUPABASE_PORT"]
password = quote_plus(raw_password)

engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db}")

#----------------------------------------------------------------
#read the table
df = pd.read_sql_table("final_watch_data_cleaned", con=engine)
df.count()

#----------------------------------------------------------------

#delete rows with NaN/blank in url column and also in the price Column.
df = df.dropna(subset=["url"])
df = df.dropna(subset=["price"])
df.count()

#----------------------------------------------------------------

#Cleaning Model Number 

MODEL_PATTERN = r'(?<!\w)([A-Z0-9]{3,24}(?:[\.\-_][A-Z0-9]{1,10})*)(?!\w)'

BLACKLIST = {
    "REGALIA", "STAINLESS", "AUTOMATICS", "COMBO", "PREMIUM", "CASUAL WATCH", "ANALOG",
    "", "NA", "NONE", "NAN"
}
BRAND_BLACKLIST = {
    "CASIO", "TITAN", "FOSSIL", "MATHEY", "TIMEX", "SEIKO", "CITIZEN", "RADO",
    "TISSOT", "MOVADO", "DIESEL", "GUESS", "ESPRIT", "ALBA", "INVICTA"
}

def extract_model_number_from_text(text):
    if not isinstance(text, str) or not text.strip():
        return pd.NA
    text = text.upper()

    # Split into possible tokens using "/", ",", or "or"
    tokens = re.split(r"[\/,]| or ", text)
    for token in tokens:
        token = token.strip()
        matches = re.findall(MODEL_PATTERN, token)
        for value in matches:
            if value in BRAND_BLACKLIST or value in BLACKLIST or not re.search(r'\d', value):
                continue
            return value
    return pd.NA

def clean_model_number(row):
    model = str(row.get("model_number", "")).strip().upper()
    part = str(row.get("part_number", "")).strip().upper()
    product_name = str(row.get("product_name", "")).strip()

    if model and model not in BLACKLIST and model not in BRAND_BLACKLIST and re.search(MODEL_PATTERN, model):
        return model

    if part and part not in BLACKLIST and part not in BRAND_BLACKLIST and re.search(MODEL_PATTERN, part):
        return part

    model_from_name = extract_model_number_from_text(product_name)
    if pd.notna(model_from_name):
        return model_from_name

    return "NA"

df["model_number"] = df.apply(clean_model_number, axis=1)

# #Post update check on model_number
# # Define common prefix patterns to strip
MODEL_PREFIXES = [
    "WATCH-", "WATCH_", "WATCH:", "MEN-", "WOMEN-", "FOSSIL ", "EMPORIO ARMANI ", "TSAR BOMBA-", "DIESEL ",
    "MICHAEL KORS ", "MICHAEL-KORS ", "TOMMY HILFIGER ", "TISSOT ", "MODEL: ", "MODEL:", "INVICTA-", "ARMANI EXCHANGE "
]

def strip_prefixes_from_model_number(text):
    if not isinstance(text, str):
        return text
    original = text.upper().strip()
    cleaned = original

    # Remove the prefix only if it matches the start
    for prefix in MODEL_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break  # Exit after the first match to avoid over-stripping

    # Only apply regex if prefix was removed
    if cleaned != original:
        match = re.search(MODEL_PATTERN, cleaned)
        return match.group(1) if match else cleaned

    # No prefix match → return original untouched
    return cleaned

df["model_number"] = df["model_number"].apply(strip_prefixes_from_model_number)

#-------------------------------------------------------------------------------

# import re
# import pandas as pd

# def clean_and_fix_model_number(row):
#     def clean_model(val):
#         if pd.isna(val):
#             return None
#         val = str(val).strip()

#         blacklist = ["casual watch", "watches", "combo", "legacy", "pack", "premium", "design", "men", "women"]
#         val_l = val.lower()
#         if any(x in val_l for x in blacklist) or len(val) <= 3:
#             return None

#         # Try to extract the model pattern
#         match = re.search(r'([A-Z]{1,3}[\dA-Z\-\.]{3,})$', val)
#         return match.group(1) if match else None

#     # Step 1: Clean existing Model Number
#     model_num = clean_model(row.get("Model Number"))
#     if model_num:
#         return model_num

#     # Step 2: Fallback to Part Number
#     part_num = row.get("Part Number")
#     if pd.notna(part_num) and str(part_num).strip():
#         return str(part_num).strip().upper()

#     # Step 3: Try extracting from Product Name
#     product_name = row.get("Product Name", "")
#     if pd.notna(product_name):
#         match = re.findall(r'([A-Z]{2,5}[\d]{3,}[A-Z]{0,5})', product_name)
#         if match:
#             return match[0]

#     # Step 4: Fallback to "NA"
#     return "NA"

# df["Model Number"] = df.apply(clean_and_fix_model_number, axis=1)

#-------------------------------------------------------------------------------------------------------------------

#delete duplicate values with product_name + model_number
df = df.drop_duplicates(subset=["product_name", "model_number"], keep="first")
df.count()
#----------------------------------------------------------------

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

#----------------------------------------------------------------

#cleaning ratings
def extract_custom_rating(text):
    if not isinstance(text, str):
        return np.nan

    matches = re.findall(r"\d+", text)  # Find all numbers
    if len(matches) >= 2:
        return float(f"{matches[0]}.{matches[1]}")
    elif len(matches) == 1:
        return float(matches[0])
    return np.nan

# Apply extraction
df["rating(out_of_5)"] = df["rating(out_of_5)"].apply(extract_custom_rating)

# ✅ Replace NaN with 0.0 instead of "NA"
df["rating(out_of_5)"] = df["rating(out_of_5)"].fillna(0.0)

df.rename(columns={"rating(out_of_5)": "ratings"}, inplace=True)
#----------------------------------------------------------------

#adding price_band
df["price_band"] = pd.cut(
    df["price"],
    bins=[0, 10000, 15000, 25000, 40000, float("inf")],
    labels=["<10K", "10K-15K", "15K-25K", "25K-40K", "40K+"],
    right=False
)

#----------------------------------------------------------------
#cleaning discount column

df["discount_(%)"] = (
    df["discount_(%)"]
    .astype(str)
    .str.extract(r"(\d+\.?\d*)")[0]  # Extract just the numeric part
    .replace("", np.nan)
    .astype(float)                  # Keep it float
)
df["discount_(%)"] = df["discount_(%)"].fillna(0.0)

df.rename(columns={"discount_(%)": "discount_percentage"}, inplace=True)

#----------------------------------------------------------------

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

#----------------------------------------------------------------
#clean model year
def clean_model_year(value):
    try:
        # Remove decimals like 2020.0 → 2020
        return int(float(value))
    except:
        return 0  # Use 0 instead of "NA" to keep it numeric
df["model_year"] = df["model_year"].apply(clean_model_year).astype(int)


#----------------------------------------------------------------

#removing the unwanted keywords
unwanted_keywords = ["pocket watch", "repair tool", "watch bezel", "watch band", "tool", "watch winder", "watch case"]
df = df[~df["product_name"].str.lower().str.contains('|'.join(unwanted_keywords))]
df.count()

#----------------------------------------------------------------

#filling major brand names for the products where brand name is missing

# Your brand mapping dictionary
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

# Fill missing brand using product_name
def infer_brand(row):
    brand = str(row.get("brand", "")).strip().lower()
    product_name = str(row.get("product_name", "")).lower()

    # If brand is already valid, return as-is
    if brand and brand != "nan" and brand != "na":
        return row["brand"].strip().title()

    # Infer from product name
    for keyword, mapped_brand in brand_map.items():
        if keyword in product_name:
            return mapped_brand

    return "NA"

# Apply to DataFrame
df["brand"] = df.apply(infer_brand, axis=1)

#----------------------------------------------------------------

#Dividing Titan as Titan, Xylys, Edge and Raga
def categorize_titan(row):
    brand = str(row["brand"]).strip().title()
    product = str(row["product_name"]).strip().title()

    # Direct rename if brand is "Xylys"
    if brand == "Xylys":
        return "Titan Xylys"

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

#----------------------------------------------------------------

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

#----------------------------------------------------------------

#Cleaning Special Features

def clean_special_features(text):
    if not isinstance(text, str):
        return text
    
    # Replace semicolons with commas
    text = text.replace(";", ",")

    # Capitalize each word (first letter after space)
    text = " ".join(word.capitalize() for word in text.split())
    
    return text

df["special_features"] = df["special_features"].apply(clean_special_features)

#----------------------------------------------------------------

#replace Nulls etc with "NA"

df = df.replace({np.nan: "NA"})
df = df.applymap(lambda x: "NA" if str(x).strip().lower() in ["", "na", "n/a", "none", "null", "nan", "n.a."] else x)

#----------------------------------------------------------------

#drop part_number
df.drop(columns=["part_number"], inplace=True)

#----------------------------------------------------------------

#as of date column
df["As of Date"] = datetime.today().strftime("%Y-%m-%d")

#----------------------------------------------------------------

#saving file

df['file'] = df['file'].astype(str).str.replace('.html', '', regex=False)
df['file'] = df['file'].astype(int)
df = df.sort_values(by='file')

#Upload to Supabase
df.to_sql("scraped_data_cleaned", con=engine, if_exists="replace", index=False)


