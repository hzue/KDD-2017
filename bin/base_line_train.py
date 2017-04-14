from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pandas.tseries.offsets import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import logging
import coloredlogs
log = logging.getLogger('kdd')
opt = {
  'log-level': 'WARNING'
}
coloredlogs.install(level=opt['log-level'])

def dlog(msg):
  log.info("\n" + str(msg))

if __name__ == '__main__':
  ## read file
  df = pd.read_csv("res/conclusion/training_20min_avg_travel_time.csv")

  ## show info
  dlog("Any missing value: " + str(np.any(df.isnull() == True)))
  dlog("Data type:\n" + str(df.dtypes))
  dlog("Brief look:\n" + str(df.head()))

  # init
  train_X = np.empty((0, 25))
  train_Y = np.asarray([])

  df['from'] = pd.to_datetime(df['time_window'].str.split(',').str[0].str.replace('[', ''))
  df['end'] = pd.to_datetime(df['time_window'].str.split(',').str[1].str.replace(')', ''))
  df = df.drop('time_window', 1)

  intersection_id_map = {'A': 1, 'B': 2, 'C': 3}
  for category, data in df.groupby(['intersection_id', 'tollgate_id']):
    tmp = data[(data['from'].dt.hour >= 8) & (data['from'].dt.hour < 10)]
    for ind in tmp.index.tolist():

      if len(df.iloc[ind-20:ind]) == 20:
        f = np.append(df.iloc[ind-20:ind]['avg_travel_time'].values, \
                [df.iloc[ind]['from'].hour, df.iloc[ind]['from'].dayofweek, \
                df.iloc[ind]['from'].minute, \
                intersection_id_map[df.iloc[ind]['intersection_id']], df.iloc[ind]['tollgate_id']])

        train_X = np.append(train_X, [f], axis=0)
        train_Y = np.append(train_Y, df.iloc[ind]['avg_travel_time'])

  print("----- start training -----")
  rf = RandomForestRegressor(n_estimators=600, criterion='mae', max_features='sqrt')
  rf.fit(train_X, train_Y)
  joblib.dump(rf, 'rf.pkl')

