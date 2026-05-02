import chromadb

#client is like library card , library is full of collections , collection is full of documents and documents have metadata and embedding
client = chromadb.PersistentClient(path="./my_chroma_db")

collection = client.get_or_create_collection(name="vehicles")
print("Collection created:", collection.name)

#Add data to the collection
collection.add(
    documents=["car is a vehicle", "bike is a vehicle", "bus is a vehicle","cycle is a vehicle"],
    metadatas=[{"type": "car"}, {"type": "bike"}, {"type": "bus"},{"type": "cycle"}],
    ids=["1", "2", "3","4"]
)

#Query the collection
results = collection.query(
    query_texts=["car is a vehicle"],
    n_results=2
)
print("Query results:", results)

#get all data from the collection
data = collection.get()
print("Current data in collection:")
for doc, ids, meta in zip(data['documents'], data['ids'], data['metadatas']):
    print(f"Document: {doc}, Metadata: {meta}")

#Update an Existing Document
collection.update(
    ids=["1"],
    documents=["car is a four-wheeled vehicle"],
    metadatas=[{"type": "car", "updated": True}]
)

#Filter documents by metadata
filtered_results = collection.get(
    where={"type": "car"}
)
print("Filtered results:", filtered_results)
#filtered results is a dictionary with keys 'ids', 'documents', and 'metadatas'. Each key maps to a list of corresponding values for the filtered documents.
# For example, if there is one document with metadata type "car", the output will show the document text, its ID, and its metadata in a structured format.

data = collection.get(include=["documents", "embeddings"]) #no need to put ids as it is already included in the default output
print("Current data in collection:")
for doc, emb, ids in zip(data['documents'], data['embeddings'],data['ids']):
    print(f"Document: {doc}, Embedding: {emb[:7]}, ID: {ids}")


#Semantic Search
semantic_results = collection.query(
    query_texts=["Vehicle which run on no fuel"],
    n_results=2,
    include=["documents", "metadatas"],
    #where={"type": "car"}
)
print(len(semantic_results))
#lesser the cosine similary score, more similar the document is to the query. So, the document with the lowest cosine similarity score will be considered the most relevant result for the query.
print("Semantic search results:")
for doc, meta in zip(semantic_results['documents'], semantic_results['metadatas']):
    print(f"Document: {doc}, Metadata: {meta}")