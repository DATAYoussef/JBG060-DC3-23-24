import sys

import requests
import json
import pandas as pd

# Load the data
data = pd.read_csv('data/articles_summary_cleaned.csv')
summaries = data["summary"].tolist()
responses = []

# Define the starting index and where you want to stop
start_index = 0
stop_index = 18524

# Check if there is a saved index in a file
try:
    with open('index.txt', 'r') as file:
        start_index = int(file.read())
except FileNotFoundError:
    pass

url = 'http://localhost:3001/v1/completions'
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}

# Load existing responses if the file exists
try:
    df = pd.read_csv('data/llama_responses.csv')
    responses = df.to_dict(orient='records')
except FileNotFoundError:
    pass

# Loop through articles from start_index to stop_index
for i in range(start_index, stop_index):
    article = summaries[i]

    data = {
        "prompt": f"\n\n### Instructions:\When reviewing the following article summary:\n\n{article}\n\nFirst, "
                  f"identify the general category of the article (e.g. 'Food', 'Technology', 'Health', etc.) and then "
                  f"list the main topics within that category along with their associated sentiments. Start your "
                  f"response with the sentiment (Positive, Negative, or Neutral), followed by the general category "
                  f"and then the main topics. Use this format: [Sentiment, General Category, Topic1, Topic2, "
                  f"...]. For example: ['Positive', 'Technology', 'Innovation', 'Sustainability']\nIf the article is "
                  f"about food, try to remain consistent with previous responses regarding food categories and "
                  f"topics.### Response:\n",
        "stop": [
            "###"
        ],
        "max_tokens": 100,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_data = response.json()
        responses.append(response_data)
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")

    # Append new responses to the CSV file every 10 iterations
    if (i + 1) % 10 == 0:
        df = pd.DataFrame(responses)
        df.to_csv('data/llama_responses.csv', index=False, mode='a', header=False)  # Append to the CSV file

        # Save the current index to a file
        with open('index.txt', 'w') as file:
            file.write(str(i + 1))  # Increment by 1 to start from the next index when resumed

        # Empty the responses list to free up memory for the next batch
        responses.clear()

    print(i)

# Write responses to a CSV file one final time after completing all iterations
df = pd.DataFrame(responses)
df.to_csv('data/llama_responses.csv', index=False, mode='a', header=False)  # Append to the CSV file

# Save the final index to a file
with open('index.txt', 'w') as file:
    file.write(str(stop_index))
