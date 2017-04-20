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

CUR_ML = None

################### common ###################
def grab_data_within_range(filepath, start_date, end_date, fill_missing_value=False):
  df = util.read_conclusion_file(filepath)

  df = df[(df['from'] <= datetime.strptime(end_date + " 23:59:59", "%Y-%m-%d %H:%M:%S")) \
        & (df['from'] > datetime.strptime(start_date + " 00:00:00", "%Y-%m-%d %H:%M:%S"))]

  df = df[(df['from'].dt.hour >= 4) & (df['from'].dt.hour <= 20)]

  # df = df[((df['from'].dt.hour >= 8) & (df['from'].dt.hour < 10)) | ((df['from'].dt.hour >= 17) & (df['from'].dt.hour < 19))]

  return df

def generate_link_route_info():
  route_file_path = 'res/dataSets/training/routes (table 4).csv'
  link_file_path = 'res/dataSets/training/links (table 3).csv'
  route = {}; link = {}

  f_r = open(route_file_path, 'r')
  f_r.readline()
  for line in f_r.readlines():
    col = line.strip().split("\",")
    col = [c.replace("\"", "") for c in col]
    route["{}-{}".format(col[0], col[1])] = col[2].split(',')

  f_l = open(link_file_path, 'r')
  f_l.readline()
  for line in f_l.readlines():
    col = line.strip().split("\",")
    col = [c.replace("\"", "") for c in col]
    link[col[0]] = {'length': col[1], 'width': col[2], 'lanes': col[3], 'in_top': col[4], 'out_top': col[5], 'lane_width': col[6]}
  pprint(link)

  return route, link

def generate_submit_file(df_test_iter, prefix, submit_file_name):
  pred_y = CUR_ML.read_result(prefix, submit_file_name)
  write_submit_file(df_test_iter, pred_y, prefix, submit_file_name)

def write_submit_file(df_test_iter, pred_y, prefix, submit_file_name):
  submit_file = open("{0}/{1}".format(prefix, submit_file_name), 'w')
  submit_file.write("intersection_id,tollgate_id,time_window,avg_travel_time\n")
  for index, row in df_test_iter:
    submit_file.write("{},{},\"[{},{})\",{}\n".format(row['intersection_id'], row['tollgate_id'], row['from'], row['end'], pred_y[index]))
  submit_file.close()

def gen_feature_array(df):
  X = []; y = []
  g = df.groupby(['intersection_id', 'tollgate_id'])
  route_map = list(g.groups.keys())
  df_test_iter = df.iterrows()
  df_test_iter, df_test_iter_backup = itertools.tee(df_test_iter)
  for index, row in df_test_iter:
    # label
    y.append(row['avg_travel_time'])

    # [feature]
    X_tmp = []
    X_tmp += [
        # row['q1_1'],
        # row['q2_1'],
        # row['q3_1'],
        # row['mean1'],
        # row['median1'],
        # row['var1'],
        row['q1_2'],
        row['q2_2'],
        row['q3_2'],
        row['mean2'],
        row['median2'],
        row['var2'],
        row['hour'] * 60 + row['minute'] # need to improve, no effect
    ]

    # [feature]
    weekday = [0, 0, 0, 0, 0, 0, 0]
    weekday[row['weekday']] = 1
    X_tmp += weekday

    # [feature]
    route = [0, 0, 0, 0, 0, 0]
    # rout_map
    route[util.find_route_index(route_map, tuple([row['intersection_id'], row['tollgate_id']]))] = 1
    X_tmp += route

    # fianl feature
    X.append(X_tmp)

  return X, y, df_test_iter_backup

################### testing ###################
def generate_testing_data(train_start_date, train_end_date, \
        test_start_date, test_end_date, prefix, fill_missing_value=False):

  df_train = grab_data_within_range('res/conclusion/training_20min_avg_travel_time.csv', \
          train_start_date, train_end_date, fill_missing_value)
  df_train = add_training_dataframe_column(df_train)

  df_test = generate_testing_dataframe(test_start_date, test_end_date, df_train)
  df_test = add_testing_dataframe_column(df_test, df_train)

  X, y, df_test_iter = gen_feature_array(df_test)
  CUR_ML.predict(X, y, prefix)
  return df_test_iter

def generate_testing_dataframe(test_start_date, test_end_date, df_train):
  test_dates = []
  days = pd.date_range(test_start_date, test_end_date)
  times1 = pd.date_range('08:00', '09:40', freq="20min")
  times2 = pd.date_range('17:00', '18:40', freq="20min")
  route_group = list(df_train.groupby(['intersection_id', 'tollgate_id']).groups.keys())
  df_test = pd.DataFrame(columns=[_ for _ in df_train])
  for route in route_group:
    for d in days:
      for t in times1.append(times2):
        day = str(d.date()) + " " + str(t.time())
        df_test = df_test.append({
            'intersection_id': route[0],
            'tollgate_id': int(route[1]),
            'avg_travel_time': 0.0,
            'from': datetime.strptime(day, "%Y-%m-%d %H:%M:%S"),
            'end': datetime.strptime(day, "%Y-%m-%d %H:%M:%S") + pd.Timedelta(minutes=20)
          }, ignore_index=True)
  df_test.tollgate_id = df_test.tollgate_id.astype(int)
  return df_test

def add_testing_dataframe_column(df_test, df_train):

  df_test['weekday'] = df_test['from'].dt.weekday
  df_test['hour'] = df_test['from'].dt.hour
  df_test['minute'] = df_test['from'].dt.minute

  ts = list(df_train.groupby(['weekday', 'hour']).groups.keys())
  for t in ts:
    tmp_train_df = df_train.loc[(df_train['weekday'] == t[0]) & (df_train['hour'] == t[1])]['avg_travel_time']
    mask = (df_test['weekday'] == t[0]) & (df_test['hour'] == t[1])
    df_test.loc[mask, 'mean2'] = tmp_train_df.mean()
    df_test.loc[mask, 'median2'] = tmp_train_df.median()
    df_test.loc[mask, 'var2'] = tmp_train_df.var()
    df_test.loc[mask, 'q1_2'] = tmp_train_df.quantile(.25)
    df_test.loc[mask, 'q2_2'] = tmp_train_df.quantile(.5)
    df_test.loc[mask, 'q3_2'] = tmp_train_df.quantile(.75)

  ts = list(df_train.groupby(['weekday', 'hour', 'minute']).groups.keys())
  for t in ts:
    tmp_train_df = df_train.loc[(df_train['weekday'] == t[0]) & (df_train['hour'] == t[1]) & (df_train['minute'] == t[2])]['avg_travel_time']
    mask = (df_test['weekday'] == t[0]) & (df_test['hour'] == t[1]) & (df_test['minute'] == t[2])
    df_test.loc[mask, 'mean1'] = tmp_train_df.mean()
    df_test.loc[mask, 'median1'] = tmp_train_df.median()
    df_test.loc[mask, 'var1'] = tmp_train_df.var()
    df_test.loc[mask, 'q1_1'] = tmp_train_df.quantile(.25)
    df_test.loc[mask, 'q2_1'] = tmp_train_df.quantile(.5)
    df_test.loc[mask, 'q3_1'] = tmp_train_df.quantile(.75)

  return df_test


################### training ###################
def generate_training_data(start_date, end_date, prefix, fill_missing_value=False):
  df = grab_data_within_range('res/conclusion/training_20min_avg_travel_time.csv', \
          start_date, end_date, fill_missing_value)
  df = add_training_dataframe_column(df)
  X, y, _ = gen_feature_array(df)
  CUR_ML.train(X, y, prefix)

def add_training_dataframe_column(df):

  route, link = generate_link_route_info()

  df['weekday'] = df['from'].dt.weekday
  df['hour'] = df['from'].dt.hour
  df['minute'] = df['from'].dt.minute

  df['mean1'] = df.groupby(['weekday', 'hour', 'minute'])['avg_travel_time'].transform(np.mean)
  df['mean2'] = df.groupby(['weekday', 'hour'])['avg_travel_time'].transform(np.mean)

  df['var1'] = df.groupby(['weekday', 'hour', 'minute'])['avg_travel_time'].transform(np.var)
  df['var2'] = df.groupby(['weekday', 'hour'])['avg_travel_time'].transform(np.var)

  df['median1'] = df.groupby(['weekday', 'hour', 'minute'])['avg_travel_time'].transform(np.median)
  df['median2'] = df.groupby(['weekday', 'hour'])['avg_travel_time'].transform(np.median)

  df['q1_1'] = df.groupby(['weekday', 'hour', 'minute'])['avg_travel_time'].transform(lambda x: np.percentile(x, 25))
  df['q2_1'] = df.groupby(['weekday', 'hour', 'minute'])['avg_travel_time'].transform(lambda x: np.percentile(x, 50))
  df['q3_1'] = df.groupby(['weekday', 'hour', 'minute'])['avg_travel_time'].transform(lambda x: np.percentile(x, 75))

  df['q1_2'] = df.groupby(['weekday', 'hour'])['avg_travel_time'].transform(lambda x: np.percentile(x, 25))
  df['q2_2'] = df.groupby(['weekday', 'hour'])['avg_travel_time'].transform(lambda x: np.percentile(x, 50))
  df['q3_2'] = df.groupby(['weekday', 'hour'])['avg_travel_time'].transform(lambda x: np.percentile(x, 75))

  return df


if __name__ == '__main__':
  generate_link_route_info()
  exit()

  # some setting
  prefix = 'result/new'
  CUR_ML = ml.rf

  # check file path
  submit_file_name='submit.csv'
  if not os.path.exists(prefix): os.mkdir(prefix)

  ## run!
  generate_training_data(start_date='2016-07-19', end_date='2016-09-30', prefix=prefix)
  # for submit
  # generate_training_data(start_date='2016-07-19', end_date='2016-10-17', prefix=prefix)

  df_test_iter = generate_testing_data(test_start_date='2016-10-11', test_end_date='2016-10-17' \
                    , train_start_date='2016-07-19', train_end_date='2016-09-30', prefix=prefix)
  # for submit
  # df_test_iter = generate_testing_data(test_start_date='2016-10-18', test_end_date='2016-10-24' \
  #                   , train_start_date='2016-07-19', train_end_date='2016-10-10', prefix=prefix)

  generate_submit_file(df_test_iter, prefix, submit_file_name)

  mape = util.evaluation('{}/{}'.format(prefix, submit_file_name), 'res/conclusion/testing_ans.csv')
  print("mape: " + str(mape))

