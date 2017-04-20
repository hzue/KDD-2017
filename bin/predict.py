from __future__ import print_function
from subprocess import check_output
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pandas.tseries.offsets import *
from scipy.interpolate import interp1d
import util

# opt = {
#   'start_date': '2016-10-18',
#   'end_date': '2016-10-24',
#   'test_miss_start_time': '15:00',
#   'test_miss_end_time': '16:40',
#   'pred_start_time': '17:00',
#   'pred_end_time': '18:40',
#   'submit_file': 'submit_file_17_19.csv',
#   'test_file': 'pred_17_19.feature',
#   'train_file': 'train_17_19.scale',
#   'scale_model': 'scale_model_17_19',
#   'tmp_result_file': 'rvkde_17_19-test-result',
#   'train_src': 'res/conclusion/training_20min_avg_travel_time.csv',
#   'test_src': 'res/conclusion/test1_20min_avg_travel_time.csv'
# }

# opt = {
#   'start_date': '2016-10-18',
#   'end_date': '2016-10-24',
#   'test_miss_start_time': '06:00',
#   'test_miss_end_time': '07:40',
#   'pred_start_time': '08:00',
#   'pred_end_time': '09:40',
#   'submit_file': 'submit_file_8_10.csv',
#   'test_file': 'pred_8_10.feature',
#   'train_file': 'train_8_10.scale',
#   'scale_model': 'scale_model_8_10',
#   'tmp_result_file': 'rvkde_8_10-test-result',
#   'train_src': 'res/conclusion/training_20min_avg_travel_time.csv',
#   'test_src': 'res/conclusion/test1_20min_avg_travel_time.csv'
# }

opt = {
  'start_date': '2016-10-11',
  'end_date': '2016-10-17',
  'test_miss_start_time': '06:00',
  'test_miss_end_time': '07:40',
  'pred_start_time': '08:00',
  'pred_end_time': '09:40',
  'submit_file': 'result/val/submit_file_8_10.csv',
  'test_file': 'result/val/pred_8_10.feature',
  'train_file': 'result/val/train_8_10.scale',
  'scale_model': 'result/val/scale_model_8_10',
  'tmp_result_file': 'result/val/rvkde_8_10-test-result',
  'train_src': 'result/val/training_data.csv',
  'test_src': 'result/val/testing_data.csv'
}

opt = {
  'start_date': '2016-10-11',
  'end_date': '2016-10-17',
  'test_miss_start_time': '15:00',
  'test_miss_end_time': '16:40',
  'pred_start_time': '17:00',
  'pred_end_time': '18:40',
  'submit_file': 'result/val/submit_file_17_19.csv',
  'test_file': 'result/val/pred_17_19.feature',
  'train_file': 'result/val/train_17_19.scale',
  'scale_model': 'result/val/scale_model_17_19',
  'tmp_result_file': 'result/val/rvkde_17_19-test-result',
  'train_src': 'result/val/training_data.csv',
  'test_src': 'result/val/testing_data.csv'
}

if __name__ == '__main__':
  df_train = util.read_conclusion_file(opt['train_src'])
  df_test = util.read_conclusion_file(opt['test_src'])

  ## fill missing value
  days = pd.date_range(opt['start_date'], opt['end_date'])
  times = pd.date_range(opt['test_miss_start_time'], opt['test_miss_end_time'], freq="20min")
  test_missing_days = []

  # create date string
  for d in days:
    for t in times:
      test_missing_days.append(str(d.date()) + " " + str(t.time()))

  # count with differnet route
  for category, data in df_test.groupby(['intersection_id', 'tollgate_id']):
    cannot_inter = 0
    miss = 0
    total = 0
    # print(category)
    for day in test_missing_days:
      total += 1
      tmp = data[data['from'] == datetime.strptime(day, "%Y-%m-%d %H:%M:%S")]
      if len(tmp) == 0:
        miss += 1
        cur_day = datetime.strptime(day, "%Y-%m-%d %H:%M:%S")
        near = [
          data[data['from'] == (cur_day - pd.Timedelta(minutes=20))],
          data[data['from'] == (cur_day + pd.Timedelta(minutes=20))],
          data[data['from'] == (cur_day - pd.Timedelta(minutes=40))],
          data[data['from'] == (cur_day + pd.Timedelta(minutes=40))]
        ]
        left_count = len(near[0]) + len(near[1])
        right_count = len(near[2]) + len(near[3])
        # get median
        if left_count + right_count < 2 or left_count == 0 or right_count == 0:
          d = df_train.groupby(['intersection_id', 'tollgate_id']).get_group(category)
          a = d[(d['from'].dt.weekday == cur_day.weekday()) & \
                   (d['from'].dt.hour == cur_day.hour) & \
                   (d['from'].dt.minute == cur_day.minute)]['avg_travel_time'].median()
          df_test = df_test.append({
              'intersection_id': category[0],
              'tollgate_id': int(category[1]),
              'avg_travel_time': a,
              'from': datetime.strptime(day, "%Y-%m-%d %H:%M:%S"),
              'end': datetime.strptime(day, "%Y-%m-%d %H:%M:%S") + pd.Timedelta(minutes=20)
            }, ignore_index=True)
          cannot_inter += 1
        # get interpolation
        else:
          ind = [1, 2, 4, 5]
          exist_ind = []
          exist_near = []
          for i, v in enumerate(near):
            if len(v) > 0:
              exist_ind.append(ind[i])
              exist_near.append(v['avg_travel_time'].tolist()[0])
          # different policy interpolation
          f = None
          if left_count + right_count < 4: f = interp1d(exist_ind, exist_near, kind='slinear')
          else: f = interp1d(exist_ind, exist_near, kind='cubic')

          df_test = df_test.append({
              'intersection_id': category[0],
              'tollgate_id': int(category[1]),
              'avg_travel_time': float(f(3)),
              'from': datetime.strptime(day, "%Y-%m-%d %H:%M:%S"),
              'end': datetime.strptime(day, "%Y-%m-%d %H:%M:%S") + pd.Timedelta(minutes=20)
            }, ignore_index=True)

    # print("miss         : " + str(miss))
    # print("cannot inter : " + str(cannot_inter))
    # print("total        : " + str(total))


  ## create testing date
  days = pd.date_range(opt['start_date'], opt['end_date'])
  times = pd.date_range(opt['pred_start_time'], opt['pred_end_time'], freq="20min")
  test_dates = []
  for d in days:
    for t in times:
      test_dates.append(str(d.date()) + " " + str(t.time()))

  submit_file = open(opt['submit_file'], 'w')
  submit_file.write("intersection_id,tollgate_id,time_window,avg_travel_time\n")
  Y = []
  g = df_test.groupby(['intersection_id', 'tollgate_id'])
  route_map = list(g.groups.keys())
  count = 0
  for category, data in g:
    for date in test_dates:
      # init feature var
      cur_day = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
      three_days_avg = []
      route = [0, 0, 0, 0, 0, 0]
      weekday = [0, 0, 0, 0, 0, 0, 0]

      # gen feature
      route[util.find_route_index(route_map, category)] = 1
      weekday[cur_day.weekday()] = 1
      three_days_avg.append(data[data['from'] == (cur_day - pd.Timedelta(minutes=20))].iloc[0]['avg_travel_time'])
      three_days_avg.append(data[data['from'] == (cur_day - pd.Timedelta(minutes=40))].iloc[0]['avg_travel_time'])
      three_days_avg.append(data[data['from'] == (cur_day - pd.Timedelta(minutes=60))].iloc[0]['avg_travel_time'])
      hour_minute = cur_day.hour * 60 + cur_day.minute

      # features.append(three_days_avg + route + weekday + [hour_minute])
      Y.append(0.0)

      util.gen_feature_file([three_days_avg + route + weekday + [hour_minute]], [0], opt['test_file'])
      check_output("svm-scale -r %s %s > %s" % (opt['scale_model'], opt['test_file'], opt['test_file'].replace('feature', 'scale')), shell=True)
      check_output("./sbin/rvkde --best --predict --regress -v %s -V %s -b 5 --ks 13 --kt 17 > %s" % (opt['train_file'], opt['test_file'].replace('feature', 'scale'), opt['tmp_result_file']), shell=True)
      pred_val = check_output("head -n 4 %s | tail -n 1 | cut -d ' ' -f 2"  % (opt['tmp_result_file']), shell=True)
      pred_val = float(pred_val.decode("utf-8"))

      data = data.append({
          'intersection_id': category[0],
          'tollgate_id': int(category[1]),
          'avg_travel_time': pred_val,
          'from': cur_day,
          'end': cur_day + pd.Timedelta(minutes=20)
        }, ignore_index=True)

      submit_file.write("%s,%d,\"[%s,%s)\",%f\n" % (category[0], int(category[1]), cur_day, cur_day + pd.Timedelta(minutes=20), pred_val))

  submit_file.close()


