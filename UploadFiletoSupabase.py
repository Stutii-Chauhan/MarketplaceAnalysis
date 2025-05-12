import os
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus  # ✅ Needed for password encoding

db = os.environ["SUPABASE_DB"]
user = os.environ["SUPABASE_USER"]
raw_password = os.environ["SUPABASE_PASSWORD"]
host = os.environ["SUPABASE_HOST"]
port = os.environ["SUPABASE_PORT"]

# ✅ Encode password for special characters like @, !, etc.
password = quote_plus(raw_password)

print("🔍 DEBUGGING ENVIRONMENT VARIABLES")
print("HOST:", host)
print("USER:", user)
print("DB:", db)

engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db}")

for file in os.listdir("."):
    if file.endswith(".xlsx") or file.endswith(".csv"):
        print(f"📄 Processing: {file}")
        try:
            if file.endswith(".xlsx"):
                df = pd.read_excel(file)
            else:
                df = pd.read_csv(file, encoding="ISO-8859-1")  # ✅ safer for non-UTF-8 content

            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
            table_name = file.lower().replace(".xlsx", "").replace(".csv", "").replace(" ", "_")
            df.to_sql(table_name, engine, if_exists="replace", index=False)
            print(f"✅ Uploaded: {table_name}")
        except Exception as e:
            print(f"❌ Failed to upload {file}: {e}")


#IPv4 from IPv6
