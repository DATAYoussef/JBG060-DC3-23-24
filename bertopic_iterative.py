# IMPORTS
from bertopic import BERTopic
import warnings
import pandas as pd
import os
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS, SpectralClustering
warnings.filterwarnings("ignore")

df = pd.read_csv("data/articles_summary_cleaned.csv", parse_dates=["date"]) # Read data into 'df' dataframe
# print(df.shape) # Print dataframe shape

docs = df["summary"].tolist()

iterations = 10 # Number of iterations to run

keyword_sets = [
    (['hunger', 'food insecurity', 'conflict'], 'hunger'),
    (['refugees', 'displaced'], 'refugees'),
    (['humanitarian'], 'humanitarian'),
    (['conflict', 'fighting', 'murder', 'military'], 'conflict'),
    (["politics", "government", "elections", "independence"], 'politics'),
    (['aid', 'assistance', 'relief'], 'aid'),
    (['food', 'crop', 'famine'], 'crops')
]


def get_relevant_topics(bertopic_model, keywords, top_n):
    if type(keywords) is str: keywords = [keywords]  # If a single string is provided convert it to list type

    relevant_topics = list()  # Initilize an empty list of relevant topics

    for keyword in keywords:  # Iterate through list of keywords

        # Find the top n number of topics related to the current keyword(s)
        topics = bertopic_model.find_topics(keyword, top_n=top_n)

        # Add the topics to the list of relevant topics in the form of (topic_id, relevancy)
        relevant_topics.extend(
            zip(topics[0], topics[1])  # topics[0] = topic_id, topics[1] = relevancy
        )

    relevant_topics.sort(key=lambda x: x[1])  # Sort the list of topics on ASCENDING ORDER of relevancy

    # Get a list of the set of unique topics (with greates relevancy in case of duplicate topics)
    relevant_topics = list(dict(relevant_topics).items())

    relevant_topics.sort(key=lambda x: x[1],
                         reverse=True)  # Now sort the list of topics on DESCENDING ORDER of relevancy

    return relevant_topics[:10]  # Return a list of the top_n unique relevant topics


if os.path.exists('southsudan_model'):
    bertopic = BERTopic.load('southsudan_model')
else:
    bertopic = BERTopic(language="english", calculate_probabilities=True, verbose=True)  # Initialize the BERTopic model

    bertopic.fit_transform(docs)  # Fit the model to the list of article summaries
    bertopic.save("southsudan_model")  # Save the trained model as "southsudan_model"

if os.path.exists('kmeans_model'):
    kmeans_model = BERTopic.load('kmeans_model')
else:
    cluster_model = KMeans(n_clusters=10)
    kmeans_model = BERTopic(language="english", calculate_probabilities=True, verbose=True,
                            hdbscan_model=cluster_model)  # Initialize the BERTopic model

    kmeans_model.fit_transform(docs)  # Fit the model to the list of article summaries
    kmeans_model.save("kmeans_model")  # Save the trained model

if os.path.exists('agglomerative_model'):
    agglomerative_model = BERTopic.load('agglomerative_model')
else:
    cluster_model = AgglomerativeClustering(n_clusters=10)
    agglomerative_model = BERTopic(language="english", calculate_probabilities=True, verbose=True,
                                   hdbscan_model=cluster_model)  # Initialize the BERTopic model

    agglomerative_model.fit_transform(docs)  # Fit the model to the list of article summaries
    agglomerative_model.save("agglomerative_model")  # Save the trained model

if os.path.exists('optics_model'):
    optics_model = BERTopic.load('optics_model')
else:
    cluster_model = OPTICS(min_samples=5)  # Customize the OPTICS parameters as needed
    optics_model = BERTopic(language="english", calculate_probabilities=True, verbose=True, hdbscan_model=cluster_model)

    optics_model.fit_transform(docs)
    optics_model.save("optics_model")

if os.path.exists('spectral_model'):
    spectral_model = BERTopic.load('spectral_model')
else:
    cluster_model = SpectralClustering(n_clusters=10)  # Customize the Spectral Clustering parameters as needed
    spectral_model = BERTopic(language="english", calculate_probabilities=True, verbose=True,
                              hdbscan_model=cluster_model)

    spectral_model.fit_transform(docs)
    spectral_model.save("spectral_model")

models = [kmeans_model, agglomerative_model, optics_model, bertopic]

for model in models:
    for keywords, label in keyword_sets:
        # Get the top 10 topics related to the current set of keywords
        relevant_topics = get_relevant_topics(bertopic_model=agglomerative_model, keywords=keywords, top_n=15)

        # Create a list of topic IDs
        topic_ids = [el[0] for el in relevant_topics]

        # # Print the relevant topics
        # print(f"Top 10 topics related to '{label}':")
        # for topic_id, relevancy in relevant_topics:
        #     print(topic_id, relevancy)

        # Add a boolean column to the 'df' DataFrame if the topic is in the list of relevant topics
        df[label] = [t in topic_ids for t in model.topics_]

unsorted = df[(df["hunger"]==False) & (df["refugees"] == False) & (df["humanitarian"] == False) & (df["conflict"] == False) & (df["politics"] == False) & (df["aid"] == False) & (df["crops"] == False)]

lengths_per_iteration = []

for i in range(iterations):
    lengths = []
    #fit new models on unsorted data
    try:
        bertopic = BERTopic(language="english", calculate_probabilities=True, verbose=True)  # Initialize the BERTopic model
        bertopic.fit_transform(unsorted["summary"].tolist())  # Fit the model to the list of article summaries

        cluster_model = KMeans(n_clusters=10)
        kmeans_model = BERTopic(language="english", calculate_probabilities=True, verbose=True,
                                hdbscan_model=cluster_model)  # Initialize the BERTopic model
        kmeans_model.fit_transform(unsorted["summary"].tolist())  # Fit the model to the list of article summaries

        cluster_model = AgglomerativeClustering(n_clusters=10)
        agglomerative_model = BERTopic(language="english", calculate_probabilities=True, verbose=True,
                                          hdbscan_model=cluster_model)  # Initialize the BERTopic model

        agglomerative_model.fit_transform(unsorted["summary"].tolist())  # Fit the model to the list of article summaries

        cluster_model = OPTICS(min_samples=5)  # Customize the OPTICS parameters as needed
        optics_model = BERTopic(language="english", calculate_probabilities=True, verbose=True, hdbscan_model=cluster_model)
        optics_model.fit_transform(unsorted["summary"].tolist())

        cluster_model = SpectralClustering(n_clusters=10)  # Customize the Spectral Clustering parameters as needed
        spectral_model = BERTopic(language="english", calculate_probabilities=True, verbose=True,
                                        hdbscan_model=cluster_model)
        spectral_model.fit_transform(unsorted["summary"].tolist())
    except:
        print(lengths_per_iteration)
        with open("lengths.txt", "w") as f:
            # write lengths to file
            for length in lengths_per_iteration:
                f.write(str(length) + "\n")

        df.to_csv("data/bertopic_10_iter", index=False)
        exit()


    models = [kmeans_model, agglomerative_model, optics_model, spectral_model, bertopic]

    for model in models:
        for keywords, label in keyword_sets:
            # Get the top 10 topics related to the current set of keywords
            relevant_topics = get_relevant_topics(bertopic_model=model, keywords=keywords, top_n=10)

            # Create a list of topic IDs
            topic_ids = [el[0] for el in relevant_topics]

            # # Print the relevant topics
            # print(f"Top 10 topics related to '{label}':")
            # for topic_id, relevancy in relevant_topics:
            #     print(topic_id, relevancy)

            # Add a boolean column to 'unsorted' DataFrame if the topic is in the list of relevant topics
            unsorted[label] = [t in topic_ids for t in bertopic.topics_]
        # store length of unsorted
        length = len(unsorted[(unsorted["hunger"]==False) & (df["refugees"] == False) & (df["humanitarian"] == False) & (df["conflict"] == False) & (df["politics"] == False) & (df["aid"] == False) & (df["crops"] == False)])
        lengths.append(length)

    df.update(unsorted)

    unsorted = unsorted[(unsorted["hunger"]==False) & (df["refugees"] == False) & (df["humanitarian"] == False) & (df["conflict"] == False) & (df["politics"] == False) & (df["aid"] == False) & (df["crops"] == False)]

    lengths_per_iteration.append(lengths)

print(lengths_per_iteration)
with open("lengths.txt", "w") as f:
    #write lengths to file
    for length in lengths_per_iteration:
        f.write(str(length) + "\n")

df.to_csv("data/bertopic_10_iter", index=False)


