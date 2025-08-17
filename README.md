# Behavior-Change-Personalized-Messages
For the purposes of reproducing our methods:

## Data
Firstly, in the data folder, there are a number of template files. These files represent what our data looks like, however we are unable to share this because the data contain personal information for the participants. The data are a combination of wearable sensor readings such as mean heart rate and mean accelerometer reading, a number of self-reported metrics such as anxiety level and mental engagement level, results from a regularly administered N-back cognitive performance test, and spoken transcripts of the user talking about their day. This is contained in the combined_output_template.csv file. The combined_output_templated.json is to contain the same data, however the json format is better suited for the llm file upload. 
Similarly, the tmb_template.csv and json files contain the performance of the user on a number of the Test-My-Brain cognitive health assessments for which we used one set of test results per user.

## Scripts
The general pipeline of the contained scripts is as follows:
Neural network suggestions:
* Use embed.py to generate embeddings for the user data.
* Use model_gen.py in train mode to create the neural network model
* Use nn_guess.py to get behavior suggestions from the neural network
LLM suggestions:
* Use prep_vector_store.py to create a vector store and upload the two json files
* Use llm_guess.py to create the behavior suggestions from the llm
* Use llm_message.py to create the personalized message based on the suggestion and user data

## Output
In the output folder are the suggestions output from both our neural network and LLM. The personalized messages contained personal information regarding the participants so we cannot share that here.
