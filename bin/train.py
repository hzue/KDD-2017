from __future__ import print_function
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pandas.tseries.offsets import *
from scipy.interpolate import interp1d
from subprocess import check_output
import util


def train(opt):
  df = util.read_conclusion_file(opt['train_src'])

  ## for validation training set
  df = df[(df['from'] <= datetime.strptime(opt['end_date'] + " 23:59:59", "%Y-%m-%d %H:%M:%S")) & (df['from'].dt.hour < int(opt['test_miss_end_time'].split(":")[0])) & (df['from'].dt.hour >= int(opt['test_miss_start_time'].split(":")[0]))]

  ## fill missing value
  days = pd.date_range(opt['start_date'], opt['end_date'])
  times = pd.date_range(opt['test_miss_start_time'], opt['test_miss_end_time'], freq="20min")
  test_missing_days = []

  # create date string
  for d in days:
    for t in times:
      test_missing_days.append(str(d.date()) + " " + str(t.time()))

  # count with differnet route
  for category, data in df.groupby(['intersection_id', 'tollgate_id']):
    cannot_inter = 0
    miss = 0
    total = 0
    # print(category)
    for day in test_missing_days:
      total += 1
      tmp = data[data['from'] == datetime.strptime(day, "%Y-%m-%d %H:%M:%S")]
      if len(tmp) == 0:
        miss += 1
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
          a = data[(data['from'].dt.weekday == cur_day.weekday()) & \
                   (data['from'].dt.hour == cur_day.hour) & \
                   (data['from'].dt.minute == cur_day.minute)]['avg_travel_time'].median()
          df = df.append({
              'intersection_id': category[0],
              'tollgate_id': int(category[1]),
              'avg_travel_time': a,
              'from': datetime.strptime(day, "%Y-%m-%d %H:%M:%S"),
              'end': datetime.strptime(day, "%Y-%m-%d %H:%M:%S") + pd.Timedelta(minutes=20)
            }, ignore_index=True)
          cannot_inter += 1
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

    # print("miss         : " + str(miss))
    # print("cannot inter : " + str(cannot_inter))
    # print("total        : " + str(total))

  test_times = []
  days = pd.date_range(opt['start_date'], opt['end_date'])
  times = pd.date_range(opt['pred_start_time'], opt['pred_end_time'], freq="20min")
  for d in days:
    for t in times:
      test_times.append(str(d.date()) + " " + str(t.time()))
  print(test_times)
  exit()

  features = []
  Y = []
  g = df.groupby(['intersection_id', 'tollgate_id'])
  route_map = list(g.groups.keys())
  for category, data in g:
    for date in test_times:
      # init feature var
      cur_day = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
      three_days_avg = []
      route = [0, 0, 0, 0, 0, 0]
      weekday = [0, 0, 0, 0, 0, 0, 0]

      # gen feature
      route[util.find_route_index(route_map, category)] = 1
      weekday[cur_day.weekday()] = 1
      three_days_avg.append(data[data['from'] == (cur_day - pd.Timedelta(minutes=20))].iloc[0]['avg_travel_time'])
      three_days_avg.append(data[data['from'] == (cur_day - pd.Timedelta(minutes=40))].iloc[0]['avg_travel_time'])
      three_days_avg.append(data[data['from'] == (cur_day - pd.Timedelta(minutes=60))].iloc[0]['avg_travel_time'])
      hour_minute = cur_day.hour * 60 + cur_day.minute

      features.append(three_days_avg + route + weekday + [hour_minute])
      Y.append(data[data['from'] == cur_day].iloc[0]['avg_travel_time'])

  util.gen_feature_file(features, Y, opt['out_train_feature_file'] + ".feature")
  check_output("svm-scale -s %s %s > %s" % (opt['scale_model'], opt['out_train_feature_file'] + ".feature", opt['out_train_feature_file'] + ".scale"), shell=True)




