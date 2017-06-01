import numpy as np
from dataframe import dataframe
from datetime import datetime
import util

###############################################################################
def read_csv_file(filepath, start_date, end_date):
  df = {}
  f = open(filepath, 'r')
  header = f.readline().strip().replace("\"" ,"")
  header = header.split(",")
  content = f.readlines()
  entry_len = len(content)

  for h in header:
    df[h] = [ np.nan for _ in range(entry_len)]

  for i in range(0, entry_len):
    cols = [ _.replace("\"", "") for _ in content[i].strip().split("\",")]
    for i2, v in enumerate(cols):
      df[header[i2]][i] = v

  return df

###############################################################################
def read_conclusion_file(filepath, start_date, end_date):
  df = {}
  f = open(filepath, 'r')

  start = datetime.strptime(start_date + " 00:00:00", "%Y-%m-%d %H:%M:%S")
  end = datetime.strptime(end_date + " 23:59:59", "%Y-%m-%d %H:%M:%S")

  header = f.readline().strip().replace("\"" ,"")
  header = header.split(",") + ['from', 'end']

  time_window_index = header.index('time_window')
  assert time_window_index, "Key time_window not found, conclusion fil eformat error!"

  content = f.readlines()

  for h in header: df[h] = []

  for i in range(0, len(content)):
    cols = [ _.replace("\"", "") for _ in content[i].strip().split("\",")]
    f, e = cols[time_window_index].replace("[", "").replace(")", "").split(",")
    f = datetime.strptime(f, "%Y-%m-%d %H:%M:%S")

    ###### sample engineering could add here
    if f < start or f > end: continue
    if f.hour < 4 or f.hour > 20: continue
    if f > datetime.strptime("2016-10-01" + " 00:00:00", "%Y-%m-%d %H:%M:%S") and \
       f <= datetime.strptime("2016-10-07" + " 23:59:59", "%Y-%m-%d %H:%M:%S"): continue
    # if f > datetime.strptime("2016-09-15" + " 00:00:00", "%Y-%m-%d %H:%M:%S") and \
       # f <= datetime.strptime("2016-09-18" + " 23:59:59", "%Y-%m-%d %H:%M:%S"): continue
    # if f > datetime.strptime("2016-09-04" + " 00:00:00", "%Y-%m-%d %H:%M:%S") and \
       # f <= datetime.strptime("2016-09-05" + " 23:59:59", "%Y-%m-%d %H:%M:%S"): continue
    ######

    e = datetime.strptime(e, "%Y-%m-%d %H:%M:%S")
    cols += [f, e]
    for i2, v in enumerate(cols):
      df[header[i2]].append(v)

  # arrangement column
  df['avg_travel_time'] = list(map(float, df['avg_travel_time']))
  df.pop('time_window')

  check_format(df)

  return df

def check_format(df):
  validation = []
  for key in df.keys(): validation.append(len(df[key]))
  for i in range(0, len(validation) - 1):
    assert validation[i] == validation[i+1], "Format error!"

###############################################################################
def read_link_route_info():
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

###############################################################################
def generate_submit_file(info_map, prefix, submit_file_name, read_method):
  pred_y = read_method()
  write_submit_file(info_map, pred_y, prefix, submit_file_name)

def write_submit_file(info_map, pred_y, prefix, submit_file_name):
  submit_file = open("{0}/{1}".format(prefix, submit_file_name), 'w')
  submit_file.write("intersection_id,tollgate_id,time_window,avg_travel_time\n")
  for index, row in enumerate(info_map):
    submit_file.write("{},{},\"[{},{})\",{}\n".format(row['intersection_id'], row['tollgate_id'], row['from'], row['end'], pred_y[index]))
  submit_file.close()

