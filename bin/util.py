import pandas as pd
import numpy as np
import time
from datetime import datetime

def timeit(method):
  def timed(*args, **kw):
    ts = time.time()
    result = method(*args, **kw)
    print("[Timeit Log] Start \'{}\' --------------".format(method.__name__))
    te = time.time()
    print('{} sec\n'.format(te - ts))
    return result
  return timed

def read_conclusion_file(csv_file):
  df = pd.read_csv(csv_file)
  df['from'] = pd.to_datetime(df['time_window'].str.split(',').str[0].str.replace('[', ''))
  df['end'] = pd.to_datetime(df['time_window'].str.split(',').str[1].str.replace(')', ''))
  df = df.drop('time_window', 1)
  return df

def find_route_index(route_map, route):
  for i, v in enumerate(route_map):
    if v[0] == route[0] and v[1] == route[1]:
      return i
  assert False, "no this route"

def gen_feature_file(features, labels, file_path='hohohahalala.feature'):
  f = open(file_path, 'w')
  line = ""
  for i, v in enumerate(features):
    line += str(labels[i])
    for i2, v2 in enumerate(v):
      line += " %d:%s" % (i2 + 1, v2)
    line += "\n"
  f.write(line)
  f.close()

def evaluation(pred_file, ans_file):
  pred = _read_file(pred_file)
  ans = _read_file(ans_file)
  mape = 0.0; route_count = 0
  for each_ans_route_id in ans:
    time_count = 0; tmp_sum = 0.0
    for each_ans_id_time in ans[each_ans_route_id]:
      if each_ans_route_id in pred and each_ans_id_time in pred[each_ans_route_id]:
        tmp_sum += abs((pred[each_ans_route_id][each_ans_id_time] - ans[each_ans_route_id][each_ans_id_time]) / ans[each_ans_route_id][each_ans_id_time])
      else: assert False, 'pred file error!'
      time_count += 1
    mape += tmp_sum / time_count
    route_count += 1
  mape /= route_count
  return mape

def _read_file(in_file): # private function
  fr = open(in_file, 'r')
  fr.readline() # remove header line
  data = fr.readlines()
  fr.close()
  result = {}
  for each_data in data:
    each_data  = each_data.replace('"', '').replace('\n', '').split(',')
    route_id = each_data[0] + '-' + each_data[1]
    if route_id not in result.keys(): result[route_id] = {}
    result[route_id][each_data[2] + "," + each_data[3]] = float(each_data[4])
  return result

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
