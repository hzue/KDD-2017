import pandas as pd
from datetime import datetime
import numpy as np


# def get_before_1hr_mean(x, df):
#   hr = int(x['from'].hour) - int(x['from'].hour) % 2
#   end = datetime.strptime(str(x['from'].date()) + " {}:00:00".format(hr), "%Y-%m-%d %H:%M:%S")
#   start = end - pd.Timedelta(hours=1)
#   tmp_df = df[(df['from'] >= start) & (df['from'] < end) & (df['intersection_id'] == x['intersection_id']) & (df['tollgate_id'] == x['tollgate_id'])]
#   if len(tmp_df) == 0: return np.nan
#   else: return tmp_df['avg_travel_time'].mean()

def judge_holiday(x):
  tmp = x.strftime("%Y-%m-%d")
  if tmp == '2016-09-15' or tmp == '2016-09-16' or tmp == '2016-09-17' \
          or tmp == '2016-10-01' or tmp == '2016-10-02' or tmp == '2016-10-03' \
          or tmp == '2016-10-04' or tmp == '2016-10-05' or tmp == '2016-10-06' \
          or tmp == '2016-10-07':
    return 1
  elif tmp == '2016-09-18' or tmp == '2016-10-08' or tmp == '2016-10-09': return 0
  elif x.weekday == 5 or x.weekday == 6: return 1
  else: return 0

def get_temp(x, df_weather):
  temp_mask =  df_weather[(df_weather['date'] == x['from'].date()) & \
        (df_weather['hour'] == int(x['from'].hour) - int(x['from'].hour) % 3)]["temperature"]
  if temp_mask.any() == False:
    return np.nan
  else:
    return float(temp_mask.iloc[0])

def get_rel_h(x, df_weather):
  temp_mask =  df_weather[(df_weather['date'] == x['from'].date()) & \
        (df_weather['hour'] == int(x['from'].hour) - int(x['from'].hour) % 3)]["rel_humidity"]
  if temp_mask.any() == False:
    return np.nan
  else:
    return float(temp_mask.iloc[0])

def map_route_score(x, route_score_map):
  return route_score_map["{}-{}".format(x.intersection_id, x.tollgate_id)]

def sum_path_len(x, link, route):
  tmp_sum = 0
  for id in route["{}-{}".format(x.intersection_id, x.tollgate_id)]:
    tmp_sum += int(link[id]['length'])
  return tmp_sum

def count_route_width_mean(x, link, route):
  tmp_sum = 0
  count = 0
  for id in route["{}-{}".format(x.intersection_id, x.tollgate_id)]:
    tmp_sum += int(link[id]['width'])
    count += 1
  return tmp_sum / count



