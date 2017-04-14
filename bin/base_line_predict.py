from __future__ import print_function
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import datetime

if __name__ == '__main__':

  df = pd.read_csv("res/conclusion/training_20min_avg_travel_time.csv")
  df['from'] = pd.to_datetime(df['time_window'].str.split(',').str[0].str.replace('[', ''))
  df['end'] = pd.to_datetime(df['time_window'].str.split(',').str[1].str.replace(')', ''))
  df = df.drop('time_window', 1)

  df_test = pd.read_csv("res/conclusion/test1_20min_avg_travel_time.csv")
  df_test['from'] = pd.to_datetime(df_test['time_window'].str.split(',').str[0].str.replace('[', ''))
  df_test['end'] = pd.to_datetime(df_test['time_window'].str.split(',').str[1].str.replace(')', ''))
  df_test = df_test.drop('time_window', 1)

  df_all = pd.concat([df, df_test])

  days = pd.date_range("2016-10-18", "2016-10-24")
  times = pd.date_range("08:00", "09:40", freq="20min")
  testing_time = []
  for d in days:
    for t in times:
      testing_time.append(str(d.date()) + " " + str(t.time()))

  print(testing_time)
  exit()
  rf = joblib.load('rf.pkl')

  for category, data in df_all.groupby(['intersection_id', 'tollgate_id']):
    data = data.sort_values(by='from').reset_index(drop=True)
    for d in days:
      for t in times:

  # rf.predict()



