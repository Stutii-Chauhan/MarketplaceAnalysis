import os
import pandas as pd
from sqlalchemy import create_engine
import unicodedata

# Load the full connection string from environment variable
DATABASE_URL = os.environ["RAILWAY_DB_URL"]

# Debug
print("✅ Using Railway DB URL")
engine = create_engine(DATABASE_URL)

# Loop through all Excel/CSV files in the current directory
for file in sorted(os.listdir(".")):
    if file.endswith(".xlsx") or file.endswith(".csv"):
        print(f"📄 Processing: {file}")
        try:
            # Read data with encoding flexibility
            if file.endswith(".xlsx"):
                df = pd.read_excel(file)
            else:
                try:
                    df = pd.read_csv(file, encoding="utf-8-sig")
                except UnicodeDecodeError:
                    df = pd.read_csv(file, encoding="cp1252")

            # Normalize column names
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

            # Clean string columns
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].astype(str).apply(lambda x: unicodedata.normalize("NFKC", x).replace("’", "'").replace("�", "'"))

            # Generate table name
            table_name = file.lower().replace(".xlsx", "").replace(".csv", "").replace(" ", "_")

            # Upload to Railway DB
            with engine.connect() as conn:
                df.to_sql(table_name, conn, if_exists="replace", index=False)

            print(f"✅ Uploaded: {table_name}")
        except Exception as e:
            print(f"❌ Failed to upload {file}: {e}")
