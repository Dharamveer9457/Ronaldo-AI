import os
import openai
import sys
import langchain
from langchain.document_loaders import PyPDFLoader
import pinecone
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
import tensorflow as tf


sys.path.append("../..")
sys.path.append("/path/to/pinecone-client")

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

openai.api_key = os.environ["OPENAI_API_KEY"]

loader = PyPDFLoader("./docs/CR7.pdf")
# pages = loader.load()
# page = pages[1]
# print(page.page_content)

pinecone_api_key = "440b97e9-3714-498a-90f0-04c61b347062"

# Create a Pinecone client
# pinecone_client = pinecone.Client(pinecone_api_key)
pinecone_client = pinecone.init(api_key=pinecone_api_key, environment="my_env")
# Load the PDF dataset
pdf_dataset = []
with open("./docs/CR7.pdf", "rb") as f:
    pdf_bytes = f.read()
    pdf_dataset.append(pdf_bytes)

# Embed the PDF dataset using LangChain's OpenAIEmbeddings wrapper
embeddings = OpenAIEmbeddings(openai.api_key)

# Index and store the embeddings in Pinecone
pinecone_client.index_vectors(embeddings, "pdf_embeddings")

# Create a LangChain RetrievalQA model
qa_model = langchain.RetrievalQA(retriever=pinecone_client.as_retriever())

# Train the model
qa_model.train(["pdf_embeddings"], pinecone_client)

# Save the trained model
qa_model.save_model("trained_model.pt")

# Load the trained model
qa_model = tf.saved_model.load("trained_model.pt")

# Ask the model a question
question = "When was ronaldo born?"

# Answer the question using the model
answer = qa_model.run(question)

# Print the answer
print(answer)
