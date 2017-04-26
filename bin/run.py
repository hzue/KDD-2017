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
import lambda_func
import file_handler as fh

CUR_ML = None
ROUTE_SCORE_MAP = {'A-2': -0.5, 'A-3': -0.5, 'C-3': -1.5, 'B-3': -1.5, 'B-1': -3, 'C-1': -3}
ROUTE_IN_OUT_LANES_NUM = {'A-2': [3, 1], 'A-3': [3, 1], 'C-3': [3, 1], 'B-3': [1, 1], 'B-1': [1, 2], 'C-1': [3, 2]}

################### common ###################

@util.timeit
def gen_feature_array(df, drop_nan=False):

  # [feature]
  df['holiday'] = df['from'].apply(lambda x: lambda_func.judge_holiday(x))

  # [feature]
  # w1 = fh.generate_weather_info('res/dataSets/testing_phase1/weather (table 7)_test1.csv')
  # w2 = fh.generate_weather_info('res/dataSets/training/weather (table 7)_training.csv')
  # df_weather = pd.concat([w2, w1], ignore_index=True)
  # df['temp'] = df.apply(lambda x: lambda_func.get_temp(x, df_weather), axis=1)
  # df['rel_humidity'] = df.apply(lambda x: lambda_func.get_rel_h(x, df_weather), axis=1)

  X = []; y = []
  g = df.groupby(['intersection_id', 'tollgate_id'])
  route_map = list(g.groups.keys())
  df_test_iter = df.iterrows()
  df_test_iter, df_test_iter_backup = itertools.tee(df_test_iter)
  route_info, link_info = fh.generate_link_route_info()
  for index, row in df_test_iter:
    # label
    y.append(row['avg_travel_time'])

    # [feature]
    X_tmp = []
    X_tmp += [
        # row['route_min'],
        # row['route_max'],
        # row['route_std'],
        # row['route_mean'],
        # row['route_q1'],
        # row['route_q2'],
        # row['route_q3'],

        # row['q1_1'],
        # row['q2_1'],
        # row['q3_1'],
        # row['mean1'],
        # row['median1'],
        # row['var1']

        # row['q1_2'],
        # row['q2_2'],
        # row['q3_2'],
        # row['mean2'],
        # row['median2'],
        # row['var2'],

        # row['temp'],
        # row['std'],
        # row['holiday'],
        # row['date_delta'],
        # row['min'],
        # row['max'],

        row['route_len'],
        row['route_width_mean'],
        row['route_score'],
        np.sin(float(row['hour'] * 60 + row['minute']) / 1440. * 360. * np.pi / 180.)
    ]

    # [feature]
    link_feature = [0 for i in range(0, 24)]
    for each_link in route_info[str(row['intersection_id']) + "-" + str(row['tollgate_id'])]:
      link_feature[int(each_link) - 100] = 1
    X_tmp += link_feature

    # [feature]
    X_tmp += ROUTE_IN_OUT_LANES_NUM[str(row['intersection_id']) + "-" + str(row['tollgate_id'])]

    # [feature]
    weekday = [0, 0, 0, 0, 0, 0, 0]
    weekday[row['weekday']] = 1
    weekday.pop(0) # idont know why, but it works
    X_tmp += weekday

    # [feature]
    route = [0, 0, 0, 0, 0, 0]
    route[util.find_route_index(route_map, tuple([row['intersection_id'], row['tollgate_id']]))] = 1
    X_tmp += route

    # [feature]
    # month = [0, 0, 0, 0,]
    # month[int(row['month']) - 7] = 1
    # X_tmp += month

    for x in X_tmp:
      if np.isnan(x): assert False, "missing value !!"

    # fianl feature
    X.append(X_tmp)

  return X, y, df_test_iter_backup

################### testing ###################
@util.timeit
def generate_testing_data(train_start_date, train_end_date, \
        test_start_date, test_end_date, prefix):

  df_train = fh.grab_data_within_range('res/conclusion/training_20min_avg_travel_time.csv', \
          train_start_date, train_end_date)
  df_train = add_training_dataframe_column(df_train)

  df_test_given = fh.grab_data_within_range('res/conclusion/testing_data_{}_{}.csv'.format(test_start_date, test_end_date), \
          test_start_date, test_end_date)

  df_test = generate_testing_dataframe(test_start_date, test_end_date, df_train)
  df_test = add_testing_dataframe_column(df_test, df_train, df_test_given)

  X, y, df_test_iter = gen_feature_array(df_test)
  CUR_ML.predict(X, y, prefix)
  return df_test_iter

@util.timeit
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

@util.timeit
def add_testing_dataframe_column(df_test, df_train, df_test_given):

  df_test['weekday'] = df_test['from'].dt.weekday
  df_test['month'] = df_test['from'].dt.month
  df_test['hour'] = df_test['from'].dt.hour
  df_test['minute'] = df_test['from'].dt.minute
  df_test['date_delta'] = df_test['from'].apply(lambda x: (x - datetime(2016, 7, 19)).days)

  # this part can move to common section to prevent duplicating code
  route, link = fh.generate_link_route_info()
  df_test['route_len'] = df_test.apply(lambda x: lambda_func.sum_path_len(x, link, route), axis=1)
  df_test['route_width_mean'] = df_test.apply(lambda x: lambda_func.count_route_width_mean(x, link, route), axis=1)
  df_test['route_score'] = df_test.apply(lambda x: lambda_func.map_route_score(x, ROUTE_SCORE_MAP), axis=1)

  ts = list(df_train.groupby(['intersection_id', 'tollgate_id', 'weekday']).groups.keys())
  for t in ts:
    tmp_train_df = df_train.loc[(df_train['intersection_id'] == t[0]) & (df_train['tollgate_id'] == t[1]) & (df_train['weekday'] == t[2])]['avg_travel_time']
    mask = (df_test['intersection_id'] == t[0]) & (df_test['tollgate_id'] == t[1]) & (df_test['weekday'] == t[2])
    df_test.loc[mask, 'route_mean'] = tmp_train_df.mean()
    df_test.loc[mask, 'route_min']  = tmp_train_df.min()
    df_test.loc[mask, 'route_max']  = tmp_train_df.max()
    df_test.loc[mask, 'route_std']  = tmp_train_df.std()
    df_test.loc[mask, 'route_q1']   = tmp_train_df.quantile(.25)
    df_test.loc[mask, 'route_q2']   = tmp_train_df.quantile(.5)
    df_test.loc[mask, 'route_q3']   = tmp_train_df.quantile(.75)

  ts = list(df_train.groupby(['weekday', 'hour']).groups.keys())
  for t in ts:
    tmp_train_df = df_train.loc[(df_train['weekday'] == t[0]) & (df_train['hour'] == t[1])]['avg_travel_time']
    mask = (df_test['weekday'] == t[0]) & (df_test['hour'] == t[1])
    df_test.loc[mask, 'mean2'] = tmp_train_df.mean()
    df_test.loc[mask, 'median2'] = tmp_train_df.median()
    df_test.loc[mask, 'min'] = tmp_train_df.min()
    df_test.loc[mask, 'max'] = tmp_train_df.max()
    df_test.loc[mask, 'std'] = tmp_train_df.std()
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

@util.timeit
def generate_training_data(start_date, end_date, prefix):
  df = fh.grab_data_within_range('res/conclusion/training_20min_avg_travel_time.csv', \
          start_date, end_date)
  df = add_training_dataframe_column(df)
  X, y, _ = gen_feature_array(df)
  CUR_ML.train(X, y, prefix)

@util.timeit
def add_training_dataframe_column(df):
  route, link = fh.generate_link_route_info()

  # g = df.groupby(['intersection_id', 'tollgate_id'])
  # df['_before1_hr_mean'] = df.apply(lambda xs: lambda_func.get_before_1hr_mean(xs, df), axis=1)

  df['weekday'] = df['from'].dt.weekday
  df['month'] = df['from'].dt.month
  df['hour'] = df['from'].dt.hour
  df['minute'] = df['from'].dt.minute
  df['date_delta'] = df['from'].apply(lambda x: (x - datetime(2016, 7, 19)).days)
  df['route_len'] = df.apply(lambda x: lambda_func.sum_path_len(x, link, route), axis=1)
  df['route_width_mean'] = df.apply(lambda x: lambda_func.count_route_width_mean(x, link, route), axis=1)
  df['route_score'] = df.apply(lambda x: lambda_func.map_route_score(x, ROUTE_SCORE_MAP), axis=1)

  inter_toll_week_hour_gourp = df.groupby(['intersection_id', 'tollgate_id', 'weekday'])['avg_travel_time']
  df['route_mean'] = inter_toll_week_hour_gourp.transform(np.mean)
  df['route_max']  = inter_toll_week_hour_gourp.transform(np.max)
  df['route_min']  = inter_toll_week_hour_gourp.transform(np.min)
  df['route_std']  = inter_toll_week_hour_gourp.transform(np.std)
  df['route_q1']   = inter_toll_week_hour_gourp.transform(lambda x: np.percentile(x, 25))
  df['route_q2']   = inter_toll_week_hour_gourp.transform(lambda x: np.percentile(x, 50))
  df['route_q3']   = inter_toll_week_hour_gourp.transform(lambda x: np.percentile(x, 75))

  week_hour_group = df.groupby(['weekday', 'hour'])
  df['min']     = week_hour_group['avg_travel_time'].transform(np.min)
  df['max']     = week_hour_group['avg_travel_time'].transform(np.max)
  df['std']     = week_hour_group['avg_travel_time'].transform(np.std)
  df['mean2']   = week_hour_group['avg_travel_time'].transform(np.mean)
  df['var2']    = week_hour_group['avg_travel_time'].transform(np.var)
  df['median2'] = week_hour_group['avg_travel_time'].transform(np.median)
  df['q1_2']    = week_hour_group['avg_travel_time'].transform(lambda x: np.percentile(x, 25))
  df['q2_2']    = week_hour_group['avg_travel_time'].transform(lambda x: np.percentile(x, 50))
  df['q3_2']    = week_hour_group['avg_travel_time'].transform(lambda x: np.percentile(x, 75))

  week_hour_min_group = df.groupby(['weekday', 'hour', 'minute'])
  df['mean1']   = week_hour_min_group['avg_travel_time'].transform(np.mean)
  df['var1']    = week_hour_min_group['avg_travel_time'].transform(np.var)
  df['median1'] = week_hour_min_group['avg_travel_time'].transform(np.median)
  df['q1_1']    = week_hour_min_group['avg_travel_time'].transform(lambda x: np.percentile(x, 25))
  df['q2_1']    = week_hour_min_group['avg_travel_time'].transform(lambda x: np.percentile(x, 50))
  df['q3_1']    = week_hour_min_group['avg_travel_time'].transform(lambda x: np.percentile(x, 75))

  return df

if __name__ == '__main__':
  # some setting
  prefix = 'result/new'
  CUR_ML = ml.rf

  # check file path
  submit_file_name='submit.csv'
  if not os.path.exists(prefix): os.mkdir(prefix)

  ## run!
  generate_training_data(start_date='2016-07-19', end_date='2016-10-10', prefix=prefix)
  df_test_iter = generate_testing_data(test_start_date='2016-10-11', test_end_date='2016-10-17' \
                    , train_start_date='2016-07-19', train_end_date='2016-10-10', prefix=prefix)

  # for submit
  # generate_training_data(start_date='2016-07-19', end_date='2016-10-17', prefix=prefix)
  # df_test_iter = generate_testing_data(test_start_date='2016-10-18', test_end_date='2016-10-24' \
  #             , train_start_date='2016-07-19', train_end_date='2016-10-10', prefix=prefix)

  fh.generate_submit_file(df_test_iter, prefix, submit_file_name, CUR_ML.read_result)

  mape = util.evaluation('{}/{}'.format(prefix, submit_file_name), 'res/conclusion/testing_ans.csv')
  pprint("mape: " + str(mape))

