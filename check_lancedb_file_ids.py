import lancedb

db = lancedb.connect("data/lancedb")
table = db.open_table("document_embeddings")
df = table.to_pandas()

print("All unique file_ids in the table:")
print(df["file_id"].unique())

file_id = "126f6a7b-e291-494d-855b-0dafb9a05859"
print(f"\nRows for file_id {file_id}:")
print(df[df["file_id"] == file_id])
