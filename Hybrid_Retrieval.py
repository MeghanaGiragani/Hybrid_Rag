# =========================================================
# HYBRID GRAPH RAG RETRIEVAL
# =========================================================

# This file performs:
#
# -> Vector Retrieval
# -> Graph Retrieval
#
# using:
#
# -> ChromaDB
# -> KuzuDB

# =========================================================
# IMPORTS
# =========================================================

import kuzu

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_chroma import Chroma

# =========================================================
# VECTOR RETRIEVAL
# =========================================================

# Retrieves semantic information
# from vector database

def vector_retrieval(query):

    print("\n🔍 VECTOR RETRIEVAL\n")

    # =====================================================
    # LOAD EMBEDDING MODEL
    # =====================================================

    embedding_model = HuggingFaceEmbeddings(

        model_name=
        "sentence-transformers/all-MiniLM-L6-v2"

    )

    # =====================================================
    # LOAD CHROMA DATABASE
    # =====================================================

    vectorstore = Chroma(

        persist_directory="db/chroma_db",

        embedding_function=embedding_model
    )

    # =====================================================
    # SIMILARITY SEARCH
    # =====================================================

    results = vectorstore.similarity_search(

        query,

        k=3
    )

    # =====================================================
    # PRINT RESULTS
    # =====================================================

    for index, result in enumerate(results):

        print(f"\n📄 Result {index+1}\n")

        print(result.page_content)

# =========================================================
# GRAPH RETRIEVAL
# =========================================================

# Retrieves relationships
# from graph database

def graph_retrieval():

    print("\n🚀 GRAPH RETRIEVAL\n")

    # =====================================================
    # CONNECT TO KUZU DB
    # =====================================================

    db = kuzu.Database("db/kuzu_graph")

    conn = kuzu.Connection(db)

    # =====================================================
    # CYPHER QUERY
    # =====================================================

    result = conn.execute("""

    MATCH (a:Entity)-[r:RELATED]->(b:Entity)

    RETURN
    a.name,
    r.relation,
    b.name

    """)

    # =====================================================
    # PRINT RELATIONSHIPS
    # =====================================================

    while result.has_next():

        print(result.get_next())

# =========================================================
# SEARCH SPECIFIC COMPANY
# =========================================================

def search_company(company_name):

    print(f"\n🏢 COMPANY SEARCH : {company_name}\n")

    db = kuzu.Database("db/kuzu_graph")

    conn = kuzu.Connection(db)

    result = conn.execute(f"""

    MATCH (a:Entity)-[r:RELATED]->(b:Entity)

    WHERE b.name = '{company_name}'

    RETURN
    a.name,
    r.relation,
    b.name

    """)

    while result.has_next():

        print(result.get_next())

# =========================================================
# MAIN FUNCTION
# =========================================================

def main():

    # =====================================================
    # USER QUERY
    # =====================================================

    query = "Who is founder of Amazon?"

    # =====================================================
    # VECTOR SEARCH
    # =====================================================

    vector_retrieval(query)

    # =====================================================
    # GRAPH SEARCH
    # =====================================================

    graph_retrieval()

    # =====================================================
    # SPECIFIC COMPANY SEARCH
    # =====================================================

    search_company("Amazon")

# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()
