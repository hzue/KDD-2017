from subprocess import check_output
import util
from pprint import pprint
import pandas as pd
import math
from datetime import datetime
import lambda_func
import file_handler as fh
import numpy as np
from scipy import stats

ROUTE_SCORE_MAP = {'A-2': -0.5, 'A-3': -0.5, 'C-3': -1.5, 'B-3': -1.5, 'B-1': -3, 'C-1': -3}
ROUTE_IN_OUT_LANES_NUM = {'A-2': [3, 1], 'A-3': [3, 1], 'C-3': [3, 1], 'B-3': [1, 1], 'B-1': [1, 2], 'C-1': [3, 2]}

class dataframe:

  def add_date_basic_info(df):
    df['weekday'] = df['from'].dt.weekday
    cols = ["w{}".format(_) for _ in range(0, 7)]
    df = pd.concat([df, pd.DataFrame(columns=cols, dtype=int)], axis=1)
    df.loc[:, cols] = df.loc[:, cols].fillna(0)
    for i, v in enumerate(cols):
      df.loc[df['weekday'] == i, v] = 1
    df['month'] = df['from'].dt.month
    df['hour'] = df['from'].dt.hour
    df['minute'] = df['from'].dt.minute
    df['time_encoding'] = np.sin(df['hour'] * 60 + df['minute'] / 1440. * 360. * np.pi / 180.)
    df['date_delta'] = df['from'].map(lambda x: (x - datetime(2016, 7, 19)).days)
    return df

  def add_route_info(df):
    route, link = fh.generate_link_route_info()
    glist = list(df.groupby(['intersection_id', 'tollgate_id']).groups)

    route_len_map = {}
    route_width_mean_map = {}
    for r in glist:
      k = "{}-{}".format(r[0], r[1])
      sum1 = 0; sum2 = 0; count = 0
      for each_link in route[k]:
        sum1 += int(link[each_link]['length'])
        sum2 += int(link[each_link]['width'])
        count += 1
      route_width_mean_map[k] = sum2 / count
      route_len_map[k] = sum1

    df['route'] = df['intersection_id'].map(str) + "-" + df['tollgate_id'].map(str)
    df['route_len'] = df['route'].map(lambda x: route_len_map[x])
    df['route_score'] = df['route'].map(lambda x: ROUTE_SCORE_MAP[x])
    df['route_width_mean'] = df['route'].map(lambda x: route_width_mean_map[x])

    return df

  def add_holiday(df):
    df['holiday'] = df['from'].map(lambda x: lambda_func.judge_holiday(x))
    return df

  def add_six_hr_mean(df):
    df['before_six_hr_mean'] = np.nan
    return df

  def add_history_minute(df):
    df['history_minute'] = df.apply()
    return df


class dataset:

  @classmethod
  def generate_data(cls, df, train_start_date, train_end_date, test_start_date, test_end_date):
    X = []; y = []; test_X = []; test_y = []

    df = dataframe.add_date_basic_info(df)
    df = dataframe.add_route_info(df)
    df = dataframe.add_holiday(df)

    ## lookup correlation
    # for key in df:
    #   print(df[key].dtype)
    #   if df[key].dtype == np.int or df[key].dtype == np.float:
    #     print(stats.pearsonr(df[key], df['avg_travel_time']))
    #     print(stats.spearmanr(df[key], df['avg_travel_time']))
    # exit()

    return cls.__generate_features_from_df(df, train_start_date, train_end_date, test_start_date, test_end_date)

  @classmethod
  def __generate_features_from_df(cls, df, train_start_date, train_end_date, test_start_date, test_end_date):
    X = []; y = []; test_X = []; test_y = []
    trs = datetime.strptime(train_start_date + " 00:00:00", "%Y-%m-%d %H:%M:%S")
    tre = datetime.strptime(train_end_date + " 23:59:59", "%Y-%m-%d %H:%M:%S")
    tes = datetime.strptime(test_start_date + " 00:00:00", "%Y-%m-%d %H:%M:%S")
    tee = datetime.strptime(test_end_date + " 23:59:59", "%Y-%m-%d %H:%M:%S")

    df_train = df[(df['from'] >= trs) & (df['from'] <= tre)]

    # df_given = df[(df['from'] >= tes) & (df['from'] <= tee) & \
    #            ((df['from'].dt.hour < 8) | ((df['from'].dt.hour >= 10) & (df['from'].dt.hour < 17)) | (df['from'].dt.hour >= 19))]
    # df_train = pd.concat([df_train, df_given], ignore_index=True)

    df_test = df[(df['from'] >= tes) & (df['from'] <= tee) & \
              (((df['from'].dt.hour >= 8) & (df['from'].dt.hour < 10)) | ((df['from'].dt.hour >= 17) & (df['from'].dt.hour < 19)))]

    df_train.reset_index(inplace=True,drop=True)
    df_test.reset_index(inplace=True,drop=True)

    X, y = cls.iter_dataframe(df_train)
    test_X, test_y = cls.iter_dataframe(df_test)

    return X, y, test_X, test_y, df_train, df_test

  def iter_dataframe(df):
    X = []; y = []
    route_info, link_info = fh.generate_link_route_info()
    g = df.groupby(['intersection_id', 'tollgate_id'])
    route_map = list(g.groups.keys())

    for index, row in df.iterrows():
      y.append(row['avg_travel_time'])

      X_tmp = [
        row['time_encoding'],
        row['route_len'],
        row['route_width_mean'],
        row['route_score'],
        row['w1'],
        row['w2'],
        row['w3'],
        row['w4'],
        row['w5'],
        row['w6'],
        row['holiday'],
        row['month'],
        row['date_delta'],
      ]

      # link_feature = [0 for i in range(0, 24)]
      # for each_link in route_info[str(row['intersection_id']) + "-" + str(row['tollgate_id'])]:
      #   link_feature[int(each_link) - 100] = 1
      # X_tmp += link_feature

      X_tmp += ROUTE_IN_OUT_LANES_NUM[str(row['intersection_id']) + "-" + str(row['tollgate_id'])]

      route = [0, 0, 0, 0, 0, 0]
      route[util.find_route_index(route_map, tuple([row['intersection_id'], row['tollgate_id']]))] = 1
      X_tmp += route

      for x in X_tmp:
        if np.isnan(x): assert False, "missing value !!"

      X.append(X_tmp)

    return X, y

