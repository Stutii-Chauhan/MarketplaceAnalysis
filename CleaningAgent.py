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

#delete rows with NaN/blank in URL column and also in the Price Column.
df = df.dropna(subset=["URL"])
df = df.dropna(subset=["Price"])
df.count()

#delete duplicate values with Product Name + Model Number
df = df.drop_duplicates(subset=["Product Name", "Model Number"], keep="first")
df.count()

#price cleaning
df["Price"] = (
    df["Price"]
    .str.replace(",", "")                   # Remove commas
    .str.replace(r"\.$", "", regex=True)   # Remove trailing dot if exists
    .astype(float)
    .round(2)
)
df = df[df["Price"] >= 10000] #removing products with price < 10000
df.count()

#cleaning ratings
df["Rating(out of 5)"] = (
    df["Rating(out of 5)"]
    .str.extract(r"(\d+\.?\d*)")        # extract only the numeric part
    .astype(float)
    .map(lambda x: int(x) if pd.notna(x) and x.is_integer() else round(x, 1) if pd.notna(x) else np.nan)
)

#adding price band
df["Price Band"] = pd.cut(
    df["Price"],
    bins=[0, 10000, 15000, 25000, 40000, float("inf")],
    labels=["<10K", "10K-15K", "15K-25K", "25K-40K", "40K+"],
    right=False
)

df["Discount (%)"] = (
    df["Discount (%)"]
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

df["Band Width"] = df["Band Width"].apply(normalize_to_mm)
df["Case Diameter"] = df["Case Diameter"].apply(normalize_to_mm)
df["Case Thickness"] = df["Case Thickness"].apply(normalize_to_mm)

#removing the unwanted keywords
unwanted_keywords = ["pocket watch", "repair tool", "watch bezel", "watch band", "tool", "watch winder", "watch case"]
df = df[~df["Product Name"].str.lower().str.contains('|'.join(unwanted_keywords))]
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

df["__product_lower__"] = df["Product Name"].str.lower()
df["__brand_match__"] = np.nan

for keyword, clean_brand in brand_map.items():
    pattern = rf"\b{re.escape(keyword)}\b"
    mask = df["Brand"].isna() & df["__product_lower__"].str.contains(pattern, regex=True)
    df.loc[mask, "__brand_match__"] = clean_brand

df["Brand"] = df["Brand"].fillna(df["__brand_match__"])

df["Brand"] = df["Brand"].fillna("NA")

df.drop(columns=["__product_lower__", "__brand_match__"], inplace=True)


#Dividing Titan as Titan, Xylys, Edge and Raga
def categorize_titan(row):
    brand = str(row["Brand"]).strip().title()
    product = str(row["Product Name"]).strip().title()

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

df["Brand"] = df.apply(categorize_titan, axis=1)

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
df["Gender"] = df["Product Name"].apply(infer_gender)

#adding as of date
df["As of Date"] = datetime.today().strftime("%Y-%m-%d")

df.to_sql("scraped_data_cleaned", con=engine, if_exists="replace", index=False)
