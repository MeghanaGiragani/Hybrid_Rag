# =========================================================
# IMPORTS
# =========================================================

# os -> used for file/folder handling
import os

# re -> used for regex pattern matching
import re

# kuzu -> graph database
import kuzu

# =========================================================
# DOCUMENT LOADERS
# =========================================================

# TextLoader -> loads txt files
# DirectoryLoader -> loads all files from docs folder

from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader
)

# =========================================================
# SEMANTIC CHUNKER
# =========================================================

# SemanticChunker splits documents
# based on meaning instead of fixed size

from langchain_experimental.text_splitter import (
    SemanticChunker
)

# =========================================================
# EMBEDDING MODEL
# =========================================================

# HuggingFaceEmbeddings converts text into vectors

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

# =========================================================
# VECTOR DATABASE
# =========================================================

# Chroma stores embeddings

from langchain_chroma import Chroma

# =========================================================
# CLEAN TEXT
# =========================================================

# This function cleans unnecessary text
# from documents before processing

def clean_text(text):

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove Wikipedia references like [1], [2]
    text = re.sub(r"\[\d+\]", "", text)

    # Remove extra spaces/newlines
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# =========================================================
# ENTITY + RELATION EXTRACTION
# =========================================================

# This function extracts:
# 1. Entities
# 2. Relationships

# Example:
#
# Andy Jassy -> CEO_of -> Amazon

def extract_entities_and_relations(text):

    # Stores graph nodes
    entities = []

    # Stores graph relationships
    relations = []

    # =====================================================
    # COMPANY LIST
    # =====================================================

    companies = [
        "Amazon",
        "Netflix",
        "Adobe",
        "Oracle",
        "Intel"
    ]

    # This variable stores currently detected company
    detected_company = None

    # =====================================================
    # DETECT COMPANY
    # =====================================================

    for company in companies:

        # Check if company exists in chunk
        if company.lower() in text.lower():

            # Store entity
            entities.append((company, "Company"))

            # Save detected company
            detected_company = company

    # =====================================================
    # AMAZON RELATIONS
    # =====================================================

    if (
        "Andy Jassy" in text
        and detected_company == "Amazon"
    ):

        entities.append(("Andy Jassy", "Person"))

        relations.append(
            ("Andy Jassy", "CEO_of", "Amazon")
        )

    if (
        "Jeff Bezos" in text
        and detected_company == "Amazon"
    ):

        entities.append(("Jeff Bezos", "Person"))

        relations.append(
            ("Jeff Bezos", "founder_of", "Amazon")
        )

    # =====================================================
    # NETFLIX RELATIONS
    # =====================================================

    if (
        "Ted Sarandos" in text
        and detected_company == "Netflix"
    ):

        entities.append(("Ted Sarandos", "Person"))

        relations.append(
            ("Ted Sarandos", "CEO_of", "Netflix")
        )

    if (
        "Reed Hastings" in text
        and detected_company == "Netflix"
    ):

        entities.append(("Reed Hastings", "Person"))

        relations.append(
            ("Reed Hastings", "founder_of", "Netflix")
        )

    # =====================================================
    # ADOBE RELATIONS
    # =====================================================

    if (
        "Shantanu Narayen" in text
        and detected_company == "Adobe"
    ):

        entities.append(("Shantanu Narayen", "Person"))

        relations.append(
            ("Shantanu Narayen", "CEO_of", "Adobe")
        )

    # =====================================================
    # ORACLE RELATIONS
    # =====================================================

    if (
        "Larry Ellison" in text
        and detected_company == "Oracle"
    ):

        entities.append(("Larry Ellison", "Person"))

        relations.append(
            ("Larry Ellison", "founder_of", "Oracle")
        )

    # =====================================================
    # INTEL RELATIONS
    # =====================================================

    if (
        "Pat Gelsinger" in text
        and detected_company == "Intel"
    ):

        entities.append(("Pat Gelsinger", "Person"))

        relations.append(
            ("Pat Gelsinger", "CEO_of", "Intel")
        )

    # =====================================================
    # HEADQUARTERS EXTRACTION
    # =====================================================

    # Regex extracts:
    #
    # headquartered in Seattle

    hq_match = re.search(

        r"headquartered in ([A-Z][a-zA-Z\s,]+)",

        text
    )

    # If headquarters found
    if hq_match and detected_company:

        # Extract location
        location = hq_match.group(1).strip()

        # Store location entity
        entities.append((location, "Location"))

        # Create graph relationship
        #
        # Example:
        #
        # Amazon -> headquartered_in -> Seattle

        relations.append(
            (
                detected_company,
                "headquartered_in",
                location
            )
        )

    # Return entities + relationships
    return entities, relations

# =========================================================
# LOAD DOCUMENTS
# =========================================================

# Loads all txt files from docs folder

def load_documents(docs_path="docs"):

    # Load all txt files
    loader = DirectoryLoader(

        path=docs_path,

        glob="*.txt",

        loader_cls=lambda path:
        TextLoader(path, encoding="utf-8")
    )

    # Load documents
    documents = loader.load()

    # Clean every document
    for doc in documents:

        doc.page_content = clean_text(
            doc.page_content
        )

    return documents

# =========================================================
# SEMANTIC CHUNKING
# =========================================================

# Splits documents into semantic chunks

def split_documents(documents):

    print("\n🧠 Using Semantic Chunking...\n")

    # =====================================================
    # EMBEDDING MODEL
    # =====================================================

    embedding_model = HuggingFaceEmbeddings(

        model_name=
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # =====================================================
    # SEMANTIC CHUNKER
    # =====================================================

    # Splits by meaning/topic

    text_splitter = SemanticChunker(

        embeddings=embedding_model,

        breakpoint_threshold_type=
        "standard_deviation",

        breakpoint_threshold_amount=1
    )

    # =====================================================
    # CREATE CHUNKS
    # =====================================================

    chunks = text_splitter.split_documents(
        documents
    )

    print(f"✅ Created {len(chunks)} semantic chunks")

    return chunks

# =========================================================
# CREATE VECTOR DATABASE
# =========================================================

# Stores embeddings in ChromaDB

def create_vector_store(chunks):

    # Embedding model
    embedding_model = HuggingFaceEmbeddings(

        model_name=
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store embeddings in Chroma
    vectorstore = Chroma.from_documents(

        documents=chunks,

        embedding=embedding_model,

        persist_directory="db/chroma_db"
    )

    return vectorstore

# =========================================================
# STORE GRAPH DATA
# =========================================================

# Stores entities + relationships in KuzuDB

def store_graph(chunks):

    # Create graph database
    db = kuzu.Database("db/kuzu_graph")

    # Create graph connection
    conn = kuzu.Connection(db)

    # Process every chunk
    for chunk in chunks:

        text = chunk.page_content

        # =================================================
        # EXTRACT ENTITIES + RELATIONS
        # =================================================

        entities, relations = (
            extract_entities_and_relations(text)
        )

        # =================================================
        # STORE ENTITIES
        # =================================================

        for entity_name, entity_type in entities:

            conn.execute(f"""

            MERGE (e:Entity {{
                name:'{entity_name}',
                type:'{entity_type}'
            }})

            """)

        # =================================================
        # STORE RELATIONS
        # =================================================

        for source, relation, target in relations:

            conn.execute(f"""

            MATCH (a:Entity), (b:Entity)

            WHERE a.name='{source}'
            AND b.name='{target}'

            MERGE (a)-[:RELATED {{
                relation:'{relation}'
            }}]->(b)

            """)

    print("✅ Graph relationships stored")

# =========================================================
# MAIN FUNCTION
# =========================================================

def main():

    print("🚀 HYBRID GRAPH RAG INGESTION")

    # =====================================================
    # LOAD DOCUMENTS
    # =====================================================

    documents = load_documents()

    print(f"✅ Loaded {len(documents)} docs")

    # =====================================================
    # CREATE SEMANTIC CHUNKS
    # =====================================================

    chunks = split_documents(documents)

    # =====================================================
    # STORE IN VECTOR DB
    # =====================================================

    create_vector_store(chunks)

    print("✅ Stored embeddings in ChromaDB")

    # =====================================================
    # STORE IN GRAPH DB
    # =====================================================

    store_graph(chunks)

    print("✅ Stored graph in KuzuDB")

# =========================================================
# ENTRY POINT
# =========================================================

# Program starts here

if __name__ == "__main__":

    main()
