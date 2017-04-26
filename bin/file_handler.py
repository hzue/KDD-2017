import pandas as pd
from datetime import datetime
import util
from pprint import pprint


def grab_data_within_range(filepath, start_date, end_date):
  df = util.read_conclusion_file(filepath)
  df = df[(df['from'] <= datetime.strptime(end_date + " 23:59:59", "%Y-%m-%d %H:%M:%S")) \
        & (df['from'] > datetime.strptime(start_date + " 00:00:00", "%Y-%m-%d %H:%M:%S"))]

  # df = df[(df['from'] < datetime.strptime("2016-10-01" + " 00:00:00", "%Y-%m-%d %H:%M:%S")) | (df['from'] > datetime.strptime("2016-10-09" + " 23:59:59", "%Y-%m-%d %H:%M:%S"))]
  # df = df[(df['from'] < datetime.strptime("2016-09-15" + " 00:00:00", "%Y-%m-%d %H:%M:%S")) | (df['from'] > datetime.strptime("2016-09-18" + " 23:59:59", "%Y-%m-%d %H:%M:%S"))]
  df = df[(df['from'].dt.hour >= 2) & (df['from'].dt.hour <= 22)]
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

  return route, link

def generate_weather_info(filepath):
  df = pd.read_csv(filepath)
  df.hour = df.hour.astype(int)
  df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
  return df

def generate_volumn_info(filepath):
  return util.read_conclusion_file(filepath)

def write_submit_file(df_test, pred_y, prefix, submit_file_name):
  submit_file = open("{0}/{1}".format(prefix, submit_file_name), 'w')
  submit_file.write("intersection_id,tollgate_id,time_window,avg_travel_time\n")
  for index, row in df_test.iterrows():
    p = pred_y[index]
    # if row['intersection_id'] == 'A' and row['tollgate_id'] == 2: p = pred_y[index]
    # else: p = pred_y[index] - 7
    submit_file.write("{},{},\"[{},{})\",{}\n".format(row['intersection_id'], row['tollgate_id'], row['from'], row['end'], p))
  submit_file.close()

def generate_submit_file(df_test, prefix, submit_file_name, read_method):
  pred_y = read_method(prefix)
  write_submit_file(df_test, pred_y, prefix, submit_file_name)

