import pandas as pd
from openai import OpenAI
from pathlib import Path

#get embeddings for textual form of the data

#get api key
with open("data/api_key.json",'r') as f:
    keys = json.load(f)
client = OpenAI(api_key=keys['api_key'])

#keys in the data for ema data
ema_keys = ['active','engaged','fatigue','anxious','depressed','irritable','social']

#cuts down excessively long text data points
def truncate(text, max_chars):
    return text[:max_chars].rsplit(" ", 1)[0] + "..." if len(text) > max_chars else text

#compute the embedding for the prompt using openai
def get_embedding(prompt_text):
    try:
        response = client.embeddings.create(
            input=[prompt_text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print("Embedding error:", e)
        return [0.0] * 1536

#check if a specific value is NaN, give a default indicator if so, otherwise return the original value
#fmt argument for specifying string formatting if desired
def check_missing(value,fmt=None,default="[Data Unavailable]"):
    if pd.isna(value) or value == "":
        return default
    else:
        if fmt is not None:
            return ("{"+fmt+"}").format(value)
        else:
            return value

#main function to loop through each day of each user's data and create the embeddings
def create_all_embeddings(df):
    grouped_prompts = []

    for pid in df["pid"].unique():
        # get data for this participant
        rows = df[df["pid"] == pid]

        max_days = len(rows)
        for i in range(max_days):
            #assemble prompt, using [Data Unavailable] to indicate missing data
            row = rows.iloc[i]
            #ema data
            e_chunk = ", ".join([check_missing(row[key]) for key in ema_keys])

            prompt = (
                f"Participant ID: {pid}, Day {i+1}\n"
                f"Daily Movement Index: {check_missing(row['acc'],':.3f')}\n"
                f"Time Out of Home: {check_missing(row['out'],':.1f')} mins\n"
                f"Watch Used From {check_missing(row['first'],':.1f')} to {check_missing(row['last'],':.1f')} mins\n"
                f"Mean Heart Rate: {check_missing(row['mean_rate'])}\n"
                f"EMA Response: {truncate(e_chunk, 300)}\n"
                f"Audio Transcript: {truncate(check_missing(row['transcript']), 2000)}\n\n"
            )

            grouped_prompts.append({
                "pid": pid,
                "day": i + 1,
                "prompt": prompt
            })

    print("Generating embeddings...")
    for entry in grouped_prompts:
        entry["embedding"] = get_embedding(entry["prompt"])

    df_output = pd.DataFrame(grouped_prompts)
    df_output.to_pickle("data/daily_prompts_and_embeddings.pkl")

if __name__ == "__main__":
    #load data
    df = pd.read_csv("combined_output.csv")
    create_all_embeddings(df)

    
