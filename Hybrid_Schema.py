# =========================================================
# GRAPH SCHEMA CREATION
# =========================================================

# This file creates:
#
# -> Node Tables
# -> Relationship Tables
#
# inside KuzuDB

# =========================================================
# IMPORT KUZU
# =========================================================

import kuzu

# =========================================================
# CREATE DATABASE
# =========================================================

# Database path

db = kuzu.Database("db/kuzu_graph")

# =========================================================
# CREATE CONNECTION
# =========================================================

conn = kuzu.Connection(db)

# =========================================================
# CREATE ENTITY NODE TABLE
# =========================================================

# Stores:
#
# -> Companies
# -> Persons
# -> Locations

conn.execute("""

CREATE NODE TABLE IF NOT EXISTS Entity(

    name STRING,

    type STRING,

    PRIMARY KEY(name)

)

""")

# =========================================================
# CREATE RELATIONSHIP TABLE
# =========================================================

# Stores graph relationships
#
# Example:
#
# Jeff Bezos -> founder_of -> Amazon

conn.execute("""

CREATE REL TABLE IF NOT EXISTS RELATED(

    FROM Entity TO Entity,

    relation STRING

)

""")

# =========================================================
# SUCCESS MESSAGE
# =========================================================

print("✅ Graph Schema Created Successfully")
