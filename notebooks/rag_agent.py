import chromadb
import pandas as pd
from transformers import pipeline

# VECTOR_DB_PATH = "C:/Users/shiva/OneDrive/Documents/Material/Amazon_Ecommorce_agent/data/vectorstore/"
# COLLECTION_NAME = "products"
# EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# from chromadb.utils import embedding_functions

# # Initialize LLM
# llm = pipeline("text2text-generation", model="google/flan-t5-base")

# embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
#     model_name=EMBED_MODEL
# )

# # Connect to persistent Chroma vectorstore
# client_chroma = chromadb.PersistentClient(path=VECTOR_DB_PATH)

# # Load the existing collection
# collection = client_chroma.get_or_create_collection(
#     name=COLLECTION_NAME,
    
# )

# print("✅ Connected to existing Chroma vectorstore")

# def retrieve_products(query, n_results=5):
#     """Retrieve top products for a query and return DataFrame + raw results."""
#     results = collection.query(query_texts=[query], n_results=n_results)

#     # Extract all metadata and distances properly
#     metadatas = results["metadatas"][0]
#     distances = results["distances"][0]

#     # Convert to DataFrame
#     df = pd.DataFrame(metadatas)

#     # Add similarity score
#     df["similarity"] = distances

#     # Sort by similarity (lower = closer in embedding space)
#     df = df.sort_values(by="similarity", ascending=True).reset_index(drop=True)

#     return df, results

# def generate_rag_response(query, top_k=5):
#     """Combine retrieval and generation for product recommendations."""
#     retrieved_df, results = retrieve_products(query, n_results=top_k)
#     if retrieved_df is None or results is None:
#         return None, "⚠️ No relevant products found."

#     # 🧠 Build concise product context
#     context = "\n".join([
#         f"{item.get('name','N/A')} — ₹{item.get('discount_price','N/A')} | ⭐ {item.get('ratings','N/A')} | {item.get('sub_category','')}"
#         for item in results["metadatas"][0]
#     ])

#     # 🗣️ Strongly worded generation prompt
#     prompt = f"""
# You are an expert e-commerce product recommendation assistant.
# The user is searching for: "{query}"

# Here are some matching products:
# {context}

# Now write the final response in this format:
# - Product Name — ₹<price> — ⭐ <rating> — <short reason in ≤ 12 words>

# After listing up to {top_k} products, write 1–2 sentences recommending the single best one overall.

# Rules:
# - Do NOT repeat these instructions.
# - Do NOT include JSON, code, or markdown.
# - Write plain text only.
# """.strip()

#     # 🧩 Generate response
#     output = llm(prompt, max_new_tokens=256)[0]["generated_text"].strip()

#     return retrieved_df, output

def generate_rag_response(query, collection, top_k=5):
    """Combine retrieval and generation for product recommendations."""
    results = collection.query(query_texts=[query], n_results=top_k)
    retrieved_df = pd.DataFrame(results["metadatas"][0])
    retrieved_df["similarity_score"] = results["distances"][0]

    # Format context
    context = "\n".join([
        f"{item.get('name','N/A')} — ₹{item.get('discount_price','N/A')} | ⭐ {item.get('ratings','N/A')} | {item.get('sub_category','')}"
        for item in results["metadatas"][0]
    ])

    # Your LLM call (example placeholder)
    prompt = f"""
    You are an expert e-commerce assistant. Recommend products for: "{query}"
    Based on the following data:
    {context}

    Format each as:
    - Product Name — ₹price — ⭐ rating — short reason (≤ 12 words)
    Finally, recommend the best one overall.
    """.strip()

    # Replace this line with your actual model call
    # output = llm(prompt, max_new_tokens=256)[0]["generated_text"].strip()
    output = f"[Mock output] Based on {len(retrieved_df)} retrieved products."

    return retrieved_df, output
