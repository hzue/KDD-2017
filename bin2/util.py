import pandas as pd
import numpy as np
import time
import datetime

def timeit(method):
  def timed(*args, **kw):
    ts = time.time()
    result = method(*args, **kw)
    print("[Timeit] Start \'{}\' --------------".format(method.__name__))
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
