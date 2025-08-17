import pandas as pd

#function to generate non-personalized messages to compare with the llm personalization

#map guess onto change
guess_map = {
    '(0, 0)': "maintain you current lifestlye.",
    '(0, 1)': "increase daily activites that are physically engaging like regular exercise, sports, or other physical activites.",
    '(0, -1)': "decrease daily activites that are physically engaging like regular exercise, sports, or other physical activites.",
    '(1, 1)': "increase daily activites that are mentally engaging like reading, sports, puzzles, or games.",
    '(1, -1)': "decrease daily activites that are mentally engaging like reading, sports, puzzles or games.",
    '(2, 1)': "increase daily activities that may cause anxiety like trying new activites that are out of your comfort zone",
    '(2, -1)':"increase daily activities that reduce anxiety such as meditating, being outside, or other activities that you find calming",
    '(3, 1)': "increase daily activites that are socially engaging like spending time with friends and family, meeting new people, or participating in social events.",
    '(3, -1)': "decrease daily activites that are socially engaging like spending time with friends and family, meeting new people, or participating in social events."
}


#generate template message for specified user
def generate_message(row):
    guess = row['best_guess']
    message = f"According to your daily activity, our models suggest that \
to improve your cognitive health scores you should {guess_map[guess]}"

    return {"pid":row['pid'],"message":message}

if __name__ == "__main__":
    suggestions_df = pd.read_csv("saved_output/timeCMA_responses/suggestions.csv")
    outputs = []
    for idx, row in suggestions_df.iterrows():
        output = generate_message(row)
        outputs.append(output)

    df = pd.DataFrame(outputs)
    df.to_csv("saved_output/timeCMA_responses/template_messages.csv")
