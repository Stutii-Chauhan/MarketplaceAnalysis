import os
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# Load environment variables
db = os.environ["SUPABASE_DB"]
user = os.environ["SUPABASE_USER"]
raw_password = os.environ["SUPABASE_PASSWORD"]
host = os.environ["SUPABASE_HOST"]
port = os.environ["SUPABASE_PORT"]
password = quote_plus(raw_password)

assert all([db, user, raw_password, host, port]), "❌ One or more environment variables is missing!"

print("🔍 DEBUGGING ENVIRONMENT VARIABLES")
print("HOST:", host)
print("USER:", user)
print("DB:", db)

engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db}")

for file in sorted(os.listdir(".")):
    if file.endswith(".xlsx") or file.endswith(".csv"):
        print(f"📄 Processing: {file}")
        try:
            # Load data
            if file.endswith(".xlsx"):
                df = pd.read_excel(file)
            else:
                try:
                    df = pd.read_csv(file, encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(file, encoding="ISO-8859-1")

            # Clean column names
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

            # Generate table name
            table_name = file.lower().replace(".xlsx", "").replace(".csv", "").replace(" ", "_")

            # Upload using fresh connection
            with engine.connect() as conn:
                df.to_sql(table_name, conn, if_exists="replace", index=False)

            print(f"✅ Uploaded: {table_name}")
        except Exception as e:
            print(f"❌ Failed to upload {file}: {e}")
