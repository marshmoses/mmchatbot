
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_community.llms import GooglePalm
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Define MongoDB connection parameters
connection_string = 'mongodb+srv://mosesmacharia084:qKQkJdvQOMRXOiO9@cluster0.gcck2ys.mongodb.net/'
database_name = "ebay_watch_data"
collection_name = "emwatch"

# Initialize SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Connect to MongoDB
client = MongoClient(connection_string)
db = client[database_name]
collection = db[collection_name]

# Function to calculate similarity between query and documents
def calculate_similarity(query_embedding, embeddings):
    if len(embeddings) == 0:
        return np.array([])  # Return an empty array if embeddings list is empty
    similarities = cosine_similarity([query_embedding], embeddings)
    return similarities[0]

# Function to execute MongoDB query
def execute_mongodb_query(query_text):
    # Generate embedding for the query
    query_embedding = model.encode([query_text])[0]
    
    # Retrieve embeddings from MongoDB collection
    embeddings = []
    documents = collection.find({}, {"_id": 0, "embedding": 1})
    for doc in documents:
        embeddings.append(doc['embedding'])
    
    # Calculate similarity scores
    similarities = calculate_similarity(query_embedding, embeddings)
    
    # Check if similarities is empty
    if len(similarities) == 0:
        return None  # Return None if there are no similarities
    
    # Find the index of the most similar document
    most_similar_index = np.argmax(similarities)
    skip_count = 0
    for document in collection.find():
        if skip_count == most_similar_index:
            return document
        skip_count += 1
    return None

# Function to optimize text using GooglePalm
def optimize_text_with_googlepalm(text):
    try:
        # Initialize GooglePalm language model
        api_key ='AIzaSyAPKvCTDopzRha-V6VMkDMuoaiVCc9YdQE' # Replace with your Google API key
        palm = GooglePalm(google_api_key=api_key, temperature=0.2)

        # Generate optimized text
        optimized_text = palm.predict(text)

        return optimized_text
    except NotImplementedError:
        # Handle the NotImplementedError here
        print("GooglePalm is deprecated. Consider updating your code.")
        return text  # Returning the original text as a fallback
# Example usage
query_text = input("Enter your query: ")
query_result = execute_mongodb_query(query_text)
if query_result:
    print("Most Similar Document:")
    # Format the document information into a paragraph
    document_paragraph = "Hello customer, welcome!\n\nWe have this amazing watch for you:\n\n"
    for key, value in query_result.items():
        if key != "embedding" and value is not None:
            document_paragraph += f"{key.capitalize()}: {value}\n"

    # Optimize the document paragraph using GooglePalm
    optimized_result = optimize_text_with_googlepalm(document_paragraph)
    print(optimized_result)
else:
    print("No documents found.")

