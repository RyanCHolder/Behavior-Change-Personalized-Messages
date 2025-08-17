import pandas as pd
import numpy as np
from openai import OpenAI
import json

#script to create vector store on openai client and upload files for use in the llm behavior suggestions and messages

#get api key
with open("data/api_key.json",'r') as f:
    keys = json.load(f)
client = OpenAI(api_key=keys['api_key'])

#create the vector store (only need to do once, no need to rerun this unless you delete the vector store)
def create_vector_store(name="knowledge_base"):
    vector_store = client.vector_stores.create(
        name=name
    )
    with open("data/vector_store_id.json", 'w') as f:
        json.dump({name:vector_store.id},f)
    return vector_store

#upload a file to the vector store
def add_file(file_path, vector_store_id):
    #load file into client
    with open(file_path, "rb") as file_content:
        result = client.files.create(
            file=file_content,
            purpose="assistants"
        )
    file_id = result.id

    #load file from client into vector store
    result = client.vector_stores.files.create(
        vector_store_id=vector_store_id,
        file_id=file_id
    )
    print("File Added")

#get the vector store id from the client
#id argument specifies expected position in list of vector stores for the one you want
def get_vec_store_id(id=0):
    vec_stores = client.vector_stores.list()
    vec_id = vec_stores.data[0].id
    return vec_id

#check status of file uploads to vector store
def check_status(vector_store_id):
    result = client.vector_stores.files.list(
        vector_store_id=vector_store_id
    )
    print(result)

if __name__ == "__main__":
    # create_vector_store()
    vector_store_id = json.load(open("data/vector_store_id.json"))["knowledge_base"]
    # add_file("data/combined_output.json",vector_store_id)
    # add_file("data/tmb.json",vector_store_id)
    check_status(vector_store_id)
    
