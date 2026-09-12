# This reads your CSV and creates the Chroma vector database.

import os
import pandas as pd

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# ==================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()


# ==================================================
# CONFIGURATION
# ==================================================

CSV_PATH = "data/customer_support_tickets.csv"

CHROMA_PATH = "chroma_db"

COLLECTION_NAME = "customer_support_tickets"


# ==================================================
# LOAD DATA
# ==================================================

print("\nLoading customer support dataset...")

df = pd.read_csv(CSV_PATH)

print(
    f"Total tickets loaded: {len(df):,}"
)


# ==================================================
# CLEAN DATA
# ==================================================

required_columns = [
    "Ticket ID",
    "Product Purchased",
    "Ticket Type",
    "Ticket Subject",
    "Ticket Description",
    "Ticket Status",
    "Resolution",
    "Ticket Priority",
    "Ticket Channel",
    "Customer Satisfaction Rating"
]

# Check required columns
missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns in CSV: {missing_columns}"
    )


# Remove rows where important RAG fields are missing

df = df.dropna(
    subset=[
        "Product Purchased",
        "Ticket Subject",
        "Ticket Description",
        "Resolution"
    ]
)


print(
    f"Tickets after cleaning: {len(df):,}"
)


# ==================================================
# CREATE LANGCHAIN DOCUMENTS
# ==================================================

documents = []


for _, row in df.iterrows():

    product = str(
        row["Product Purchased"]
    ).strip()

    ticket_type = str(
        row["Ticket Type"]
    ).strip()

    subject = str(
        row["Ticket Subject"]
    ).strip()

    description = str(
        row["Ticket Description"]
    ).strip()

    status = str(
        row["Ticket Status"]
    ).strip()

    resolution = str(
        row["Resolution"]
    ).strip()

    priority = str(
        row["Ticket Priority"]
    ).strip()

    channel = str(
        row["Ticket Channel"]
    ).strip()

    satisfaction = str(
        row["Customer Satisfaction Rating"]
    ).strip()


    # ----------------------------------------------
    # Text used for embedding
    # ----------------------------------------------

    content = f"""
Product: {product}

Ticket Type: {ticket_type}

Ticket Subject: {subject}

Ticket Description: {description}

Ticket Status: {status}

Resolution: {resolution}

Ticket Priority: {priority}

Ticket Channel: {channel}

Customer Satisfaction Rating: {satisfaction}
"""


    # ----------------------------------------------
    # Metadata
    # ----------------------------------------------

    metadata = {

        "ticket_id": str(
            row["Ticket ID"]
        ),

        "product": product,

        "ticket_type": ticket_type,

        "priority": priority,

        "status": status
    }


    # ----------------------------------------------
    # LangChain Document
    # ----------------------------------------------

    document = Document(
        page_content=content,
        metadata=metadata
    )

    documents.append(document)


print(
    f"Documents created: {len(documents):,}"
)


# ==================================================
# CREATE EMBEDDINGS
# ==================================================

print("\nCreating embeddings...")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# ==================================================
# CREATE CHROMA DATABASE
# ==================================================

print("\nCreating Chroma vector database...")

vectorstore = Chroma.from_documents(

    documents=documents,

    embedding=embeddings,

    collection_name=COLLECTION_NAME,

    persist_directory=CHROMA_PATH
)


# ==================================================
# COMPLETE
# ==================================================

print("\n======================================")
print("Chroma database created successfully")
print("======================================")

print(
    f"Database location: {CHROMA_PATH}"
)

print(
    f"Documents stored: {len(documents):,}"
)

print("\nYou can now run:")

print("streamlit run app.py")