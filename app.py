from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd

# --- Configuration ---
VECTOR_DB_PATH = "path"
COLLECTION_NAME = "products"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Initialize ChromaDB (once at startup) ---
print("🔹 Loading ChromaDB persistent client...")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)
print("✅ Vectorstore connected successfully!")


def generate_rag_response(query, collection, top_k=5):
    """Combine retrieval and generation for product recommendations."""
    try:
        results = collection.query(query_texts=[query], n_results=top_k)
        
        if not results["metadatas"] or not results["metadatas"][0]:
            return None, "No products found matching your query."
        
        retrieved_df = pd.DataFrame(results["metadatas"][0])
        retrieved_df["similarity_score"] = results["distances"][0]

        # Format context
        context = "\n".join([
            f"{item.get('name','N/A')} — ₹{item.get('discount_price','N/A')} | ⭐ {item.get('ratings','N/A')} | {item.get('sub_category','')}"
            for item in results["metadatas"][0]
        ])

        # Generate response
        prompt = f"""
        You are an expert e-commerce assistant. Recommend products for: "{query}"
        Based on the following data:
        {context}

        Format each as:
        - Product Name — ₹price — ⭐ rating — short reason (≤ 12 words)
        Finally, recommend the best one overall.
        """.strip()

        # Replace with actual LLM call if available
        # For now, use intelligent mock response
        output = f"""Based on your search for "{query}", here are the top {len(retrieved_df)} products:

{context}

💡 Recommendation: Consider the highest-rated product within your budget. Check reviews and specifications before purchase."""

        return retrieved_df, output
    
    except Exception as e:
        print(f"Error in generate_rag_response: {e}")
        return None, f"Error retrieving products: {str(e)}"


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({
                'error': 'Please enter a search query',
                'success': False
            }), 400
        
        print(f"🔍 Processing query: {query}")
        
        # Get recommendations from RAG
        products_df, ai_response = generate_rag_response(query, collection=collection, top_k=6)
        
        # Convert DataFrame to list of dicts
        products = []
        if products_df is not None and len(products_df) > 0:
            # Ensure all required fields exist
            for col in ['name', 'discount_price', 'ratings', 'link', 'image']:
                if col not in products_df.columns:
                    products_df[col] = None
            
            products = products_df.to_dict('records')
            print(f"✅ Found {len(products)} products")
        else:
            print("⚠️ No products found")
        
        return jsonify({
            'ai_response': ai_response,
            'products': products,
            'success': True
        })
    
    except Exception as e:
        print(f"❌ Error in search endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Search failed: {str(e)}',
            'success': False
        }), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting Amazon E-commerce Flask Server...")
    print("📱 Open your browser at: http://localhost:5000")
    print("="*50 + "\n")

    app.run(debug=True, host='127.0.0.1', port=5001)
