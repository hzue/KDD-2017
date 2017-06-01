import math
from datetime import datetime, timedelta

def gen_testing_file(in_file, out_file, start, end, mode):
  # Step 0: Check output type
  test_timing = None
  if mode == 'data': test_timing = [6, 8 ,15, 17]
  elif mode == 'ans': test_timing = [8, 10 ,17, 19]
  elif mode == 'train': pass
  else: assert False, "mode error, only 'data' & 'ans'"

  # Step 1: Load trajectories
  fr = open(in_file, 'r')
  fr.readline()  # skip the header
  traj_data = fr.readlines()
  fr.close()

  # Step 2: Create a dictionary to store travel time for each route per time window
  testing_start_date = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
  testing_end_date = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")

  travel_times = {}  # key: route_id. Value is also a dictionary of which key is the start time for the time window and value is a list of travel times
  for i in range(len(traj_data)):
    each_traj = traj_data[i].replace('"', '').split(',')

    trace_start_time = each_traj[3]
    intersection_id = each_traj[0]
    tollgate_id = each_traj[1]

    trace_start_time = datetime.strptime(trace_start_time, "%Y-%m-%d %H:%M:%S")
    if not (testing_start_date < trace_start_time < testing_end_date): continue
    if mode != 'train':
      if not (0 <= trace_start_time.hour - test_timing[0] and trace_start_time.hour - test_timing[1] < 0) and \
         not (0 <= trace_start_time.hour - test_timing[2] and trace_start_time.hour - test_timing[3] < 0): continue
    time_window_minute = math.floor(trace_start_time.minute / 20) * 20
    start_time_window = datetime(trace_start_time.year, trace_start_time.month, trace_start_time.day,
                                 trace_start_time.hour, time_window_minute, 0)

    route_id = intersection_id + '-' + tollgate_id
    if route_id not in travel_times.keys():
      travel_times[route_id] = {}

    tt = float(each_traj[-1]) # travel time

    if start_time_window not in travel_times[route_id].keys():
      travel_times[route_id][start_time_window] = [tt]
    else:
      travel_times[route_id][start_time_window].append(tt)

  # Step 3: Calculate average travel time for each route per time window
  fw = open(out_file, 'w')
  fw.writelines(','.join(['"intersection_id"', '"tollgate_id"', '"time_window"', '"avg_travel_time"']) + '\n')
  for route in travel_times.keys():
    route_time_windows = list(travel_times[route].keys())
    route_time_windows.sort()
    for time_window_start in route_time_windows:
      time_window_end = time_window_start + timedelta(minutes=20)
      tt_set = travel_times[route][time_window_start]
      avg_tt = round(sum(tt_set) / float(len(tt_set)), 2)
      out_line = ','.join(['"' + route.split('-')[0] + '"', '"' + route.split('-')[1] + '"',
                           '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"',
                           '"' + str(avg_tt) + '"']) + '\n'
      fw.writelines(out_line)
  fw.close()

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

if __name__ == '__main__': # example usage of above functions if directly run this program
  # gen_testing_file('res/dataSets/training/trajectories(table 5)_training.csv', \
  #         'result/val/training_data.csv', '2016-07-19 00:00:00', '2016-10-01 00:00:00', 'train')
  # gen_testing_file('res/dataSets/training/trajectories(table 5)_training.csv', \
  #         'result/testing_data.csv', '2016-10-11 00:00:00', '2016-10-18 00:00:00', 'data')
  # gen_testing_file('res/dataSets/training/trajectories(table 5)_training.csv', \
  #         'result/testing_ans.csv', '2017-10-11 00:00:00', '2016-10-18 00:00:00', 'ans')

  gen_testing_file('res/dataSets/all/trajectories(table_5)_training_phase2.csv', \
          'res/conclusion/training_data_phase2.csv', '2016-07-19 00:00:00', '2016-10-25 00:00:00', 'train')
  gen_testing_file('res/dataSets/dataSet_phase2/trajectories(table 5)_test2.csv', \
          'res/conclusion/testing_data_2016-10-25_2016_10_31.csv', '2016-10-25 00:00:00', '2016-11-01 00:00:00', 'data')
  gen_testing_file('res/dataSets/dataSet_phase2/trajectories(table_5)_training2.csv', \
          'res/conclusion/testing_ans_2016-10-18_2016-10-24.csv', '2016-10-18 00:00:00', '2016-10-25 00:00:00', 'ans')

