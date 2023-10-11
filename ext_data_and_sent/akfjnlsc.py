import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sudan_with_dist = pd.read_csv('../data/external_data/4_acleddata_2011-05-19-2023-09-28-South_Sudan_with_district.csv')
# change event_date column to datetime
sudan_with_dist['event_date2'] = pd.to_datetime(sudan_with_dist['event_date'])
# articles_sent = pd.read_pickle('data/articles_summary_cleaned_sentiment.pkl')
food_crises_cleaned = pd.read_csv('../data/food_crises_cleaned.csv')
food_crises = food_crises_cleaned.dropna(subset=['ipc'])
food = food_crises.groupby(['year_month']).agg({'ipc': 'mean'}).reset_index()
food['year_month'] = pd.to_datetime(food['year_month'], format='%Y_%m')

print(sudan_with_dist['event_type'].value_counts())

print(sudan_with_dist[['event_date', 'event_date2']])


fig, ax1 = plt.subplots()
ax1.set_xlabel('time')
ax1.set_ylabel('ipc', color='red')
ax1.plot(food['year_month'], food['ipc'], color='red')

ax2 = ax1.twinx()
ax2.set_ylabel('fat', color='blue')
ax2.plot(sudan_with_dist['event_date2'], sudan_with_dist['fatalities'], color='blue')
fig.tight_layout()
plt.show()


fig2, ax1 = plt.subplots()
ax1.set_xlabel('time')
ax1.set_ylabel('ipc', color='red')
ax1.plot(food['year_month'], food['ipc'], color='red')

ax3 = ax1.twinx()
ax3.set_ylabel('fat_sum', color='green')
sudan_with_dist['month'] = sudan_with_dist['event_date2'].dt.to_period('M')
monthly_fat = sudan_with_dist.groupby('month')['fatalities'].sum().reset_index()
ax3.plot([x.to_timestamp() for x in monthly_fat['month']], monthly_fat['fatalities'], color='green')
fig.tight_layout()
plt.show()


fig3, ax1 = plt.subplots()
ax1.set_xlabel('time')
ax1.set_ylabel('ipc', color='red')
ax1.plot(food['year_month'], food['ipc'], color='red')

ax4 = ax1.twinx()
ax4.set_ylabel('event_sum', color='orange')
sudan_with_dist['month'] = sudan_with_dist['event_date2'].dt.to_period('M')
monthly_event = sudan_with_dist.groupby('month')['event_date'].count().reset_index()
print("k")
ax4.plot([x.to_timestamp() for x in monthly_event['month']], monthly_event['event_date'], color='orange')
fig.tight_layout()
plt.show()