import time
from colorama import init, Fore
init(autoreset=True)

def timeit(method):
  def timed(*args, **kw):
    ts = time.time()
    result = method(*args, **kw)
    te = time.time()
    print("[Timeit Log] Start \'{}\': {} sec".format(method.__name__, te - ts))
    return result
  return timed

def flow_logger(method):
  def log(*args, **kw):
    print(Fore.BLUE + "[Flow Logger] {}: {}".format( \
            time.asctime(time.localtime(time.time())), method.__name__))
    return method(*args, **kw)
  return log

def find_route_index(route_map, route):
  for i, v in enumerate(route_map):
    if v[0] == route[0] and v[1] == route[1]:
      return i
  assert False, "no this route"

@flow_logger
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

def evaluation2(pred, ans):
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

