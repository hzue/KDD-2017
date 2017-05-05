import file_handler as fh
import numpy as np
from pprint import pprint
from datetime import datetime, timedelta
import util
import copy
import itertools
from dataframe import dataframe
import pandas as pd
import keras

#################################################################################
@util.flow_logger
def generate_data(train_file, test_given_file, \
        train_start_date, train_end_date, test_start_date, test_end_date):

  df_train = fh.read_conclusion_file(train_file, train_start_date, train_end_date)
  df_test_given = fh.read_conclusion_file(test_given_file, test_start_date, test_end_date)
  df_test = generate_test_dataframe(df_test_given, test_start_date, test_end_date)

  df = dataframe.concat([df_train, df_test_given, df_test], axis='row')

  df = feature.add_date_basic_info(df)
  df = feature.add_holiday(df)
  df = feature.add_route_info(df)
  df = feature.add_link_info(df)
  df = feature.add_history_travel_info(df)
  df = feature.add_poly_sim_value(df)

  df_train, df_test = split_train_test(df, test_start_date, test_end_date)

  route, link = fh.read_link_route_info()

  # sklearn svr
  # feature_list = ['all_previous_week_same_hour_q1', 'r3', 'w2', 'route_width_mean', 'time_encoding', 'w6', 'w3', 'r5', 'one_der', 'two_der']

  # sklearn rf
  # feature_list = ['all_previous_week_same_hour_max', 'autoencode_route_dim1', 'w5', 'w6', '115','120', 'route_len', 'r4']

  # libsvr
  # feature_list = ['all_previous_week_same_hour_q2', 'route_len', 'all_previous_week_same_hour_q3', 'all_previous_week_same_minute_mean', 'all_previous_week_same_minute_q3', 'all_previous_week_same_hour_mean', 'w4', 'all_previous_week_same_minute_mean', 'all_previous_week_same_minute_q1']
  # feature_list += ['all_previous_week_same_hour_q2']

  # '''
  feature_list = list(link.keys()) # 24
  feature_list += [
    'time_encoding',
    'w0',
    'w1',
    'w2',
    'w3',
    'w4',
    'w5',
    'w6',
    'autoencode_weekday_dim1',
    'autoencode_weekday_dim2',
    # 'autoencode_route_dim1',
    # 'autoencode_route_dim2',
    # 'r0',
    # 'r1',
    # 'r2',
    # 'r3',
    # 'r4',
    # 'r5',
    # 'route_len',
    # 'route_width_mean',
    # 'route_score',
    # 'route_in_lanes_num',
    # 'route_out_lanes_num',
    'holiday',
    'previous_week_same_time_travel_time',
    # --
    'all_previous_week_same_minute_mean',
    'all_previous_week_same_minute_q1',
    'all_previous_week_same_minute_q2',
    'all_previous_week_same_minute_q3',
    'all_previous_week_same_minute_min',
    'all_previous_week_same_minute_max',
    # --
    'all_previous_week_same_minute_mean',
    'all_previous_week_same_hour_mean',
    'all_previous_week_same_hour_q2',
    'all_previous_week_same_hour_q2',
    'all_previous_week_same_hour_q3',
    'all_previous_week_same_hour_min',
    'all_previous_week_same_hour_max',
    'poly_sim',
    'one_der',
    'two_der',
  ]
  # '''

  X, y, train_info_map, df_train = generate_output_ds(df_train, feature_list, 'train')
  test_X, test_y, test_info_map, df_test = generate_output_ds(df_test, feature_list, 'test')

  return X, y, train_info_map, test_X, test_y, test_info_map, feature_list, df_train, df_test

# for convenience, using pandas to generate date
def generate_test_dataframe(template, test_start_date, test_end_date):
  df = {}
  for h in dataframe.header(template): df[h] = []
  groups, _, _ = dataframe.groupby(template, ['intersection_id', 'tollgate_id'])
  days = pd.date_range(test_start_date, test_end_date)
  times1 = pd.date_range('08:00', '09:40', freq="20min")
  times2 = pd.date_range('17:00', '18:40', freq="20min")
  for inter, toll in groups:
    for d in days:
      for t in times1.append(times2):
        day = str(d.date()) + " " + str(t.time())
        df['intersection_id'].append(inter)
        df['tollgate_id'].append(toll)
        df['avg_travel_time'].append(0.0)
        df['from'].append(datetime.strptime(day, "%Y-%m-%d %H:%M:%S"))
        df['end'].append(datetime.strptime(day, "%Y-%m-%d %H:%M:%S") + pd.Timedelta(minutes=20))
  return df

def split_train_test(df, test_start_date, test_end_date):
  df_train = {}; df_test = {}
  for key in df.keys():
    df_train[key] = []
    df_test[key] = []

  divide = datetime.strptime(test_start_date + " 00:00:00", "%Y-%m-%d %H:%M:%S")
  for i, time in enumerate(df['from']):
    if time < divide: dataframe.append_by_index(df_train, df, i)
    elif (time.hour >= 8 and time.hour < 10) or ( time.hour >= 17 and time.hour < 19 ):
      dataframe.append_by_index(df_test, df, i)
  return df_train, df_test

def generate_output_ds(df, reserve_list, mode):
  X = []; y = []; info_map = []
  remove_index = []
  for i in range(0, dataframe.length(df)):
    tmp_X = []
    for key in reserve_list: tmp_X.append(df[key][i])

    if np.any(np.isnan(tmp_X)):
      if mode == 'test': assert False, "nan in test set !"
      remove_index.append(i)
      continue

    y.append(df['avg_travel_time'][i])
    info_map.append({
      'intersection_id': df['intersection_id'][i],
      'tollgate_id': df['tollgate_id'][i],
      'from': df['from'][i],
      'end': df['end'][i],
    })
    X.append(tmp_X)

  for each_key in df:
    tmp_np = np.asarray(df[each_key])
    tmp_np = np.delete(tmp_np, remove_index, axis=0)
    df[each_key] = tmp_np.tolist()

  return X, y, info_map, df

#################################################################################
class feature:

  @staticmethod
  def add_date_basic_info(df):
    df = dataframe.new_col(df, ['minute', 'hour', 'month', 'weekday', 'time_encoding', 'date_delta'])
    df = dataframe.new_col(df, ['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6'], fill=0)
    df = dataframe.new_col(df, ['autoencode_weekday_dim1', 'autoencode_weekday_dim2'])

    encoder = keras.models.load_model('autoencoder_model/weekday.encoder')

    orig_start_time = datetime(2016, 7, 19, 0, 0 ,0)
    for i, time in enumerate(df['from']):
      df['month'][i] = time.month
      df['time_encoding'][i] = np.sin((time.hour * 60 + time.minute) / 1440. * 360. * np.pi / 180.)
      df['date_delta'][i] = (time - orig_start_time).seconds
      df['w' + str(time.weekday())][i] = 1
      df['weekday'][i] = time.weekday()
      df['hour'][i] = time.hour
      df['minute'][i] = time.minute

      weekday_encoder_input = [[0 for _ in range(7)]]
      weekday_encoder_input[0][time.weekday()] = 1
      encoded_wd = encoder.predict(np.asarray(weekday_encoder_input))
      df['autoencode_weekday_dim1'][i] = encoded_wd[0][0]
      df['autoencode_weekday_dim2'][i] = encoded_wd[0][1]

    return df

  @staticmethod
  def add_link_info(df):
    route, link = fh.read_link_route_info()
    all_link = list(link.keys())
    df = dataframe.new_col(df, all_link, fill=0)
    for i in range(0, dataframe.length(df)):
      for l in route[df['route'][i]]:
        df[l][i] = 1
    return df

  @staticmethod
  def add_route_info(df):
    route, link = fh.read_link_route_info()
    dataframe.new_col(df, ['route', 'route_len', 'route_width_mean', 'route_score',
                           'route_in_lanes_num', 'route_out_lanes_num'])
    df = dataframe.new_col(df, ['r0', 'r1', 'r2', 'r3', 'r4', 'r5'], fill=0)
    df = dataframe.new_col(df, ['autoencode_route_dim1', 'autoencode_route_dim2'])

    # simple mapping as static ds
    ROUTE_MAP = {'A-2': 0, 'A-3': 1, 'C-3': 2, 'B-3': 3, 'B-1': 4, 'C-1': 5}
    ROUTE_SCORE_MAP = {'A-2': -0.5, 'A-3': -0.5, 'C-3': -1.5, 'B-3': -1.5, 'B-1': -3, 'C-1': -3}
    ROUTE_IN_OUT_LANES_NUM = {'A-2': [3, 1], 'A-3': [3, 1], 'C-3': [3, 1], 'B-3': [1, 1], 'B-1': [1, 2], 'C-1': [3, 2]}
    # generate mapping
    glist, _, _ = dataframe.groupby(df, ['intersection_id', 'tollgate_id'])
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

    encoder = keras.models.load_model('autoencoder_model/route.encoder')

    for ind, intersection_id in enumerate(df['intersection_id']):
      df['route'][ind] = "{}-{}".format(intersection_id, df['tollgate_id'][ind])
      df["{}{}".format('r', ROUTE_MAP[df['route'][ind]])][ind] = 1
      df['route_score'][ind] = ROUTE_SCORE_MAP[df['route'][ind]]
      df['route_in_lanes_num'][ind], df['route_out_lanes_num'][ind] = \
          ROUTE_IN_OUT_LANES_NUM[df['route'][ind]]
      df['route_len'][ind] = route_len_map[df['route'][ind]]
      df['route_width_mean'][ind] = route_width_mean_map[df['route'][ind]]

      route_encoder_input = [[0 for _ in range(6)]]
      route_encoder_input[0][ROUTE_MAP["{}-{}".format(intersection_id, df['tollgate_id'][ind])]] = 1
      encoded_route = encoder.predict(np.asarray(route_encoder_input))
      df['autoencode_route_dim1'][ind] = encoded_route[0][0]
      df['autoencode_route_dim2'][ind] = encoded_route[0][1]

    return df

  @staticmethod
  def add_holiday(df):
    df = dataframe.new_col(df, ['holiday'])
    for i, time in enumerate(df['from']):
      date_str = str(time.date())
      wd = time.weekday()
      if date_str == '2016-09-15' or date_str == '2016-09-16' or date_str == '2016-09-17' \
          or date_str == '2016-10-01' or date_str == '2016-10-02' or date_str == '2016-10-03' \
          or date_str == '2016-10-04' or date_str == '2016-10-05' or date_str == '2016-10-06' \
          or date_str == '2016-10-07':
        df['holiday'][i] = 1
      elif date_str == '2016-09-18' or date_str == '2016-10-08' or date_str == '2016-10-09': df['holiday'][i] = 0
      elif wd == 5 or wd == 6: df['holiday'][i] = 1
      else: df['holiday'][i] = 0
    return df

  @staticmethod
  def add_prevoius_2hr_info(df):
    # df = dataframe.new_col(df, [
    #   'prev_2hr_B-1',
    #   'prev_2hr_C-1',
    #   'prev_2hr_A-3',
    #   'prev_2hr_C-3',
    #   'prev_2hr_B-3',
    #   'prev_2hr_A-2',
    # ])
    return df

  @staticmethod
  @util.timeit
  def add_poly_sim_value(df):
    df = dataframe.new_col(df, ['poly_sim', 'one_der', 'two_der'])
    groups, iterable_groups, obj_groups = dataframe.groupby(df, ['intersection_id', 'tollgate_id'])
    for cur_ind, time in enumerate(df['from']):
      poly_list_X = []
      poly_list_y = []
      for i in obj_groups["{}-{}".format(df['intersection_id'][cur_ind], df['tollgate_id'][cur_ind])]:
        if df['from'][i] < df['from'][cur_ind]:
          poly_list_X.append(df['date_delta'][i])
          poly_list_y.append(df['avg_travel_time'][i])

      p = None
      his_data_len = len(poly_list_X)
      if his_data_len == 0:
        continue
      elif his_data_len == 1:
        p = np.poly1d(np.polyfit(poly_list_X, poly_list_y, 0))
      elif his_data_len < 5:
        p = np.poly1d(np.polyfit(poly_list_X, poly_list_y, 1))
      else:
        p = np.poly1d(np.polyfit(poly_list_X[-20:], poly_list_y[-20:], 3))

      p2 = np.polyder(p)
      p3 = np.polyder(p2)

      df['poly_sim'][cur_ind] = p(df['date_delta'][cur_ind])
      df['one_der'][cur_ind]  = p2(df['date_delta'][cur_ind] - 20)
      df['two_der'][cur_ind]  = p3(df['date_delta'][cur_ind] - 20)

    return df

  @staticmethod
  @util.timeit
  def add_history_travel_info(df):
    df = dataframe.new_col(df, [
      'previous_week_same_time_travel_time',

      'all_previous_week_same_minute_mean',
      'all_previous_week_same_minute_q1',
      'all_previous_week_same_minute_q2',
      'all_previous_week_same_minute_q3',
      'all_previous_week_same_minute_min',
      'all_previous_week_same_minute_max',

      'all_previous_week_same_hour_mean',
      'all_previous_week_same_hour_q1',
      'all_previous_week_same_hour_q2',
      'all_previous_week_same_hour_q3',
      'all_previous_week_same_hour_min',
      'all_previous_week_same_hour_max',
    ])
    groups, _, obj_groups  = dataframe.groupby(df, ['intersection_id', 'tollgate_id', 'weekday', 'hour', 'minute'])
    groups2, _, obj_groups2  = dataframe.groupby(df, ['intersection_id', 'tollgate_id', 'weekday', 'hour'])

    for cur_i, time in enumerate(df['from']):
      exist = False; his_list = []
      for ind in obj_groups["{}-{}-{}-{}-{}".format( \
              df['intersection_id'][cur_i], df['tollgate_id'][cur_i], df['weekday'][cur_i], df['hour'][cur_i], df['minute'][cur_i])]:
        if df['from'][ind] == df['from'][cur_i] - timedelta(days=7):
          exist = True
          df['previous_week_same_time_travel_time'][cur_i] = df['avg_travel_time'][ind]
        if df['from'][ind] < df['from'][cur_i]: his_list.append(df['avg_travel_time'][ind])
      if not his_list == []:
        if not exist: df['previous_week_same_time_travel_time'][cur_i] = np.mean(his_list)
        df['all_previous_week_same_minute_mean'][cur_i] = np.mean(his_list)
        df['all_previous_week_same_minute_q1'][cur_i]   = np.percentile(his_list, 25)
        df['all_previous_week_same_minute_q2'][cur_i]   = np.percentile(his_list, 50)
        df['all_previous_week_same_minute_q3'][cur_i]   = np.percentile(his_list, 75)
        df['all_previous_week_same_minute_min'][cur_i]  = np.min(his_list)
        df['all_previous_week_same_minute_max'][cur_i]  = np.max(his_list)

      his_list = []
      for ind in obj_groups2["{}-{}-{}-{}".format( \
              df['intersection_id'][cur_i], df['tollgate_id'][cur_i], df['weekday'][cur_i], df['hour'][cur_i])]:
        if df['from'][ind] < df['from'][cur_i]: his_list.append(df['avg_travel_time'][ind])
      if not his_list == []:
        df['all_previous_week_same_hour_mean'][cur_i] = np.mean(his_list)
        df['all_previous_week_same_hour_q1'][cur_i]   = np.percentile(his_list, 25)
        df['all_previous_week_same_hour_q2'][cur_i]   = np.percentile(his_list, 50)
        df['all_previous_week_same_hour_q3'][cur_i]   = np.percentile(his_list, 75)
        df['all_previous_week_same_hour_min'][cur_i]  = np.min(his_list)
        df['all_previous_week_same_hour_max'][cur_i]  = np.max(his_list)

    return df




