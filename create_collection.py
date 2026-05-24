# Create Collecttion 
# from qdrant_client import QdrantClient
# from qdrant_client.models import VectorParams, Distance

# client = QdrantClient(
#     host="localhost",
#     port=6333,
# )

# client.create_collection(
#     collection_name="fashion_catalog",
#     vectors_config=VectorParams(
#         size=384,
#         distance=Distance.COSINE,
#     ),
# )

# print("Collection created")


# Create Indexes:
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

client = QdrantClient(host="localhost", port=6333)

indexes = {
    "category": PayloadSchemaType.KEYWORD,
    "source":   PayloadSchemaType.KEYWORD,
    "gender":   PayloadSchemaType.KEYWORD,
    "color":    PayloadSchemaType.KEYWORD,
    "price":    PayloadSchemaType.FLOAT,
}

for field, schema_type in indexes.items():
    client.create_payload_index(
        collection_name="fashion_catalog",
        field_name=field,
        field_schema=schema_type,
    )
    print(f"Index created: {field}")

print("All indexes created")