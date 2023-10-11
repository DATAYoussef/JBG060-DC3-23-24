import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn import LinearRegression

articles_sent = pd.read_pickle('../data/articles_summary_cleaned_sentiment.pkl')
food_crises_cleaned = pd.read_csv('../data/food_crises_cleaned.csv')
food_crises = food_crises_cleaned.dropna(subset=['ipc'])
food = food_crises.groupby(['year_month']).agg({'ipc': 'mean'}).reset_index()

print(food.info())
print(articles_sent.info())

