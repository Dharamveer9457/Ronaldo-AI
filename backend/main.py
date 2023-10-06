from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("./docs/CR7.pdf")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function = len)
texts = text_splitter.split_documents(data)

# print (f'Now you have {len(texts)} documents')

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV') # You may need to switch with your env

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  
    environment=PINECONE_API_ENV 
)
index_name = "cr7ai" # pinecone index 

docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)



from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

app = Flask(__name__)

CORS(app, resources={r"/ask": {"origins": ["http://127.0.0.1:5500"]}})

@app.route('/')
def index():
    return jsonify("Welcome to Cristianopedia")

@app.route('/ask', methods = ['GET', 'POST'])
def ask():
    try:
        # Get the query from the client side
        query = request.json.get('query')

        # Initialize the OpenAI model
        llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

        # Load the QA chain
        chain = load_qa_chain(llm, chain_type="stuff")

        # Perform a similarity search to get relevant documents
        docs = docsearch.similarity_search(query)

        # Run the query through the chain
        answer = chain.run(input_documents=docs, question=query)

        # # Return the answer as JSON response
        response = {
            'answer': answer,
            'document_ids': [1234567890, 9876543210]
        }

        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# @app.route('/documents/<document_id>')
# def get_document(document_id):
#     # Get the full text of the document from the Pinecone DB.
#     document = pinecone.init.get_document(document_id)
#     console.log(document)

#     # Return the full text of the document as a JSON response.
#     return jsonify({'text': document.text})

if __name__ == '__main__':
    app.run(debug=True)

