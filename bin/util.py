import pandas as pd
import numpy as np
import datetime

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

# def get_testing_date_list(start, end):
#   days = pd.date_range(start, end)
#   times1 = pd.date_range("08:00", "09:40", freq="20min")
#   times2 = pd.date_range("15:00", "16:40", freq="20min")
#   testing_time = []
#   for d in days:
#     for t in times1:
#       testing_time.append(str(d.date()) + " " + str(t.time()))
#     for t in times2:
#       testing_time.append(str(d.date()) + " " + str(t.time()))
#   return testing_time
