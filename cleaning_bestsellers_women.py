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
df = pd.read_sql_table("bestsellers_women", con=engine)
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

#----------------------------------------------------------------

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

df["rating(out_of_5)"] = (
    df["rating(out_of_5)"]
    .astype(str)
    .str.extract(r"(\d+\.?\d*)")[0]     # extract numeric part
    .astype(float)                      # ensure it's float
    .round(1)                           # optional: round to 1 decimal
)
#----------------------------------------------------------------

#adding price_band
df["price_band"] = pd.cut(
    df["price"],
    bins=[0, 10000, 15000, 25000, 40000, float("inf")],
    labels=["<10K", "10K-15K", "15K-25K", "25K-40K", "40K+"],
    right=False
)

#----------------------------------------------------------------

df["discount_(%)"] = (
    df["discount_(%)"]
    .astype(str)
    .str.extract(r"(\d+\.?\d*)")[0]         # extract numeric part
    .replace("", np.nan)
    .astype(float)
    .map(lambda x: f"{int(x)}%" if x.is_integer() else f"{x}%" if pd.notna(x) else np.nan)
)

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
# cleaning model year
def clean_model_year(value):
    if pd.isna(value):
        return "NA"
    try:
        # Convert to float first, then to int to remove .0, finally to string
        return str(int(float(value)))
    except:
        return "NA"

# Apply to your column
df["model_year"] = df["model_year"].apply(clean_model_year)

#----------------------------------------------------------------

#removing the unwanted keywords
unwanted_keywords = ["pocket watch", "repair tool", "watch bezel", "watch band", "tool", "watch winder", "watch case"]
df = df[~df["product_name"].str.lower().str.contains('|'.join(unwanted_keywords))]
df.count()

#----------------------------------------------------------------

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

#drop keywords from women watches (if any)

male_keywords = ["male", "men", "man", "boy", "gents", "men's", "boy's", "couple"]
pattern = r"\b(" + "|".join([re.escape(word) for word in male_keywords]) + r")\b"
df = df[~df["product_name"].str.contains(pattern, case=False, na=False, regex=True)]

#----------------------------------------------------------------

#have 100 products per price band and remove <10K products

df = df[df["price_band"] != "<10K"]
df = (
    df.groupby("price_band", group_keys=False)
    .apply(lambda x: x.sample(n=min(len(x), 100), random_state=42))
)

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

#-----------------------------------------------------------------

#saving file

df['file'] = df['file'].astype(str).str.replace('.html', '', regex=False)
df['file'] = df['file'].astype(int)
df = df.sort_values(by='file')

#Upload to Supabase
df.to_sql("check", con=engine, if_exists="replace", index=False)
