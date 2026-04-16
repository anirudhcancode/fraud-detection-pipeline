import pandas as pd
from sqlalchemy import create_engine, text

# Database connection
DB_USER = "fraud_user"
DB_PASSWORD = "fraud_pass"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "fraud_db"

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

def load_data(filepath: str) -> pd.DataFrame:
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def create_table(engine):
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS transactions (
                id SERIAL PRIMARY KEY,
                time FLOAT,
                v1 FLOAT, v2 FLOAT, v3 FLOAT, v4 FLOAT, v5 FLOAT,
                v6 FLOAT, v7 FLOAT, v8 FLOAT, v9 FLOAT, v10 FLOAT,
                v11 FLOAT, v12 FLOAT, v13 FLOAT, v14 FLOAT, v15 FLOAT,
                v16 FLOAT, v17 FLOAT, v18 FLOAT, v19 FLOAT, v20 FLOAT,
                v21 FLOAT, v22 FLOAT, v23 FLOAT, v24 FLOAT, v25 FLOAT,
                v26 FLOAT, v27 FLOAT, v28 FLOAT,
                amount FLOAT,
                class INTEGER
            );
        """))
        conn.commit()
    print("Table created successfully")

def ingest_data(df: pd.DataFrame, engine):
    df.columns = [col.lower() for col in df.columns]
    df.to_sql("transactions", engine, if_exists="replace", index=False)
    print(f"Ingested {len(df)} rows into PostgreSQL")

if __name__ == "__main__":
    df = load_data("data/creditcard.csv")
    create_table(engine)
    ingest_data(df, engine)
    print("Ingestion complete!")