from transformers import pipeline
import pandas as pd
# import numpy as np
from tqdm import tqdm

# tqdm.pandas()

# text = "South Sudan blablabla"
# result = sentiment_analyzer(text)
# sentiment_label = result[0]['label']
# sentiment_score = result[0]['score']
#
# print(f"{sentiment_label}, {sentiment_score}")

articles_summary_cleaned = pd.read_csv('../data/articles_summary_cleaned.csv')
articles_summary_cleaned['date'] = articles_summary_cleaned['date'].apply(lambda x: x[:-3].replace('-', '_'))
# datums = list(articles_summary_cleaned['date'].unique())
# articles_summary_cleaned = articles_summary_cleaned[articles_summary_cleaned['date'].isin(datums)]

# articles_summary_cleaned = articles_summary_cleaned.iloc[:5]
# articles_summary_cleaned['sentiment'] = articles_summary_cleaned['summary'].progress_apply(lambda x: str(pipeline("sentiment-analysis", model='distilbert-base-uncased-finetuned-sst-2-english')(x)[0]['label']), )

# abc = pipeline("sentiment-analysis", model='distilbert-base-uncased-finetuned-sst-2-english')(list(articles_summary_cleaned['summary'])) # [0]['label']
# abc1 = [x['label'] for x in abc]
# abc2 = [x['score'] for x in abc]

sentiment_pipeline = pipeline("sentiment-analysis", model='distilbert-base-uncased-finetuned-sst-2-english')
sentiment_results = []
sentiment_scores = []

for text in tqdm(list(articles_summary_cleaned['summary'])):
    sent = sentiment_pipeline(text)
    sentiment_results.append(sent[0]['label'])
    sentiment_scores.append(sent[0]['score'])

articles_summary_cleaned['sentiment'] = sentiment_results
articles_summary_cleaned['sentiment_score'] = sentiment_scores
articles_summary_cleaned.to_pickle('../data/articles_summary_cleaned_sentiment.pkl')

food_crises_cleaned = pd.read_csv('../data/food_crises_cleaned.csv')
food_crises = food_crises_cleaned.dropna(subset=['ipc'])
food = food_crises.groupby(['year_month']).agg({'ipc': 'mean'}).reset_index()

print(len(food))
