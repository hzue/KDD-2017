import pandas as pd
import pandas as pd
import numpy as np
from datetime import datetime
from subprocess import check_output
import util
from pprint import pprint
from predictor import ml
import scipy
import itertools
import os
from scipy.interpolate import interp1d

def grab_data_within_range(filepath, start_date, end_date, fill_missing_value=False):
  df = util.read_conclusion_file(filepath)
  df = df[(df['from'] <= datetime.strptime(end_date + " 23:59:59", "%Y-%m-%d %H:%M:%S")) \
        & (df['from'] > datetime.strptime(start_date + " 00:00:00", "%Y-%m-%d %H:%M:%S"))]
  # df = df[((df['from'].dt.hour >= 8) & (df['from'].dt.hour < 10)) | ((df['from'].dt.hour >= 17) & (df['from'].dt.hour < 19))]
  # df = df[(df['from'].dt.hour >= 4) & (df['from'].dt.hour <= 20)]
  return df

def fill_missing_value(df, start_date, end_date, mode, df_train=None):

  if mode == 'test' and df_train == None: assert False, 'Wrong usage when fill missing value!'

  days = pd.date_range(start_date, end_date)
  times = pd.date_range('00:00', '23:59', freq="20min")
  test_missing_days = []

  # create date string
  for d in days:
    for t in times:
      test_missing_days.append(str(d.date()) + " " + str(t.time()))

  # count with differnet route
  for category, data in df.groupby(['intersection_id', 'tollgate_id']):
    # print(category)
    for day in test_missing_days:
      tmp = data[data['from'] == datetime.strptime(day, "%Y-%m-%d %H:%M:%S")]
      if len(tmp) == 0:
        cur_day = datetime.strptime(day, "%Y-%m-%d %H:%M:%S")
        near = [
          data[data['from'] == (cur_day - pd.Timedelta(minutes=20))],
          data[data['from'] == (cur_day + pd.Timedelta(minutes=20))],
          data[data['from'] == (cur_day - pd.Timedelta(minutes=40))],
          data[data['from'] == (cur_day + pd.Timedelta(minutes=40))]
        ]
        left_count = len(near[0]) + len(near[1])
        right_count = len(near[2]) + len(near[3])
        # get median
        if left_count + right_count < 2 or left_count == 0 or right_count == 0:
          a = 0
          if mode == 'train':
            a = data[(data['from'].dt.weekday == cur_day.weekday()) & \
                     (data['from'].dt.hour == cur_day.hour) & \
                     (data['from'].dt.minute == cur_day.minute)]['avg_travel_time'].median()
            if a == np.nan: print(a)

          elif mode == 'test':
            d = df_train.groupby(['intersection_id', 'tollgate_id']).get_group(category)
            a = d[(d['from'].dt.weekday == cur_day.weekday()) & \
                   (d['from'].dt.hour == cur_day.hour) & \
                   (d['from'].dt.minute == cur_day.minute)]['avg_travel_time'].median()

          df = df.append({
              'intersection_id': category[0],
              'tollgate_id': int(category[1]),
              'avg_travel_time': a,
              'from': datetime.strptime(day, "%Y-%m-%d %H:%M:%S"),
              'end': datetime.strptime(day, "%Y-%m-%d %H:%M:%S") + pd.Timedelta(minutes=20)
            }, ignore_index=True)
        # get interpolation
        else:
          ind = [1, 2, 4, 5]
          exist_ind = []
          exist_near = []
          for i, v in enumerate(near):
            if len(v) > 0:
              exist_ind.append(ind[i])
              exist_near.append(v['avg_travel_time'].tolist()[0])
          # different policy interpolation
          f = None
          if left_count + right_count < 4: f = interp1d(exist_ind, exist_near, kind='slinear')
          else: f = interp1d(exist_ind, exist_near, kind='cubic')

          df = df.append({
              'intersection_id': category[0],
              'tollgate_id': int(category[1]),
              'avg_travel_time': float(f(3)),
              'from': datetime.strptime(day, "%Y-%m-%d %H:%M:%S"),
              'end': datetime.strptime(day, "%Y-%m-%d %H:%M:%S") + pd.Timedelta(minutes=20)
            }, ignore_index=True)
  f = open('oo', 'w')
  f.write("intersection_id,tollgate_id,time_window,avg_travel_time\n")
  for index, row in df.iterrows():
    f.write("{},{},\"[{},{})\",{}\n".format(row['intersection_id'], row['tollgate_id'], row['from'], row['end'], row['avg_travel_time']))
  return df

if __name__ == '__main__':

  df = grab_data_within_range('res/conclusion/training_20min_avg_travel_time.csv', \
          '2016-07-19', '2016-10-17')
  fill_missing_value(df, '2016-07-19', '2016-10-17', 'train')
