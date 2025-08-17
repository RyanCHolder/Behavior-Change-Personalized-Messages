import pandas as pd
import numpy as np
from openai import OpenAI
import json
import time
import os

#script to prompt the llm to make predictions on behavior improvements

#get api key
with open("data/api_key.json",'r') as f:
    keys = json.load(f)
client = OpenAI(api_key=keys['api_key'])

#prompt llm to generate its behavior change suggestion and explain reasoning (see guess_prompt.txt)
def make_guess(user_df,tmb_df,pid,vector_store_id,prompt_path,reasoning=False):
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    prompt = f"user ID: {pid}\n" + prompt
    num_tries = 0
    while num_tries < 2:
        print(f"Generating Response for {pid} (try {num_tries+1})")
        try:
            if reasoning:
                pass
            else:
                response = client.responses.create(
                    model="gpt-5",
                    input=prompt,
                    tools=[{
                    "type": "file_search",
                    "vector_store_ids": [vector_store_id]
                    }]
                )
            print("Response Complete")
            return format_response(response.output_text,pid)
            
        except Exception as e:
            print(f"Rate limit hit, waiting 60 seconds... ({e})")
            time.sleep(60)
            num_tries = num_tries + 1
    print(f"Failed to generate output for {pid}")

#format the response of the llm to separate out the guess and the explanation in a dictionary
def format_response(response,pid):
    # split into lines
    parts = response.strip().split("\n", 2)  # split into at most 3 parts

    #separate guess and explanation
    guess = parts[0].strip()
    explanation = parts[-1].strip()

    return {"pid":pid,"guess":guess,"explanation":explanation}
    
#main function to prompt model for guesses for all users
#saves output as each response is received, to start from a later user, change the start_id argument
def guess_all_users(user_df,tmb_df,vector_store_id,prompt_path,reasoning=False,start_id=0):
    for i, pid in enumerate(user_df["pid"].unique()):
        if i < start_id:
            continue
        if reasoning:
            outfile = "saved_output/llm_responses/reasoning_guesses.csv"
            pass
        else:
            outfile = "saved_output/llm_responses/data_only_guesses.csv"
            response = make_guess(user_df, tmb_df, pid, vector_store_id, prompt_path)

        # convert to DataFrame row
        df_row = pd.DataFrame([response])

        # append to csv
        if i == 0 and not os.path.exists(outfile):
            df_row.to_csv(outfile, index=False, mode="w")   # first write, include header
        else:
            df_row.to_csv(outfile, index=False, mode="a", header=False)  # append without header

        print(f"Saved response for {pid}")
        # print("Waiting 60 seconds between prompts")
        # time.sleep(60)


if __name__ == "__main__":
    vector_store_id = json.load(open("data/vector_store_id.json"))["knowledge_base"]
    user_df = pd.read_csv("data/combined_output.csv")
    tmb_df = pd.read_csv("data/tmb.csv")
    prompt_path = "prompts/guess_prompt(long).txt"
    guess_all_users(user_df,tmb_df,vector_store_id,prompt_path,start_id=39)