import os
from train import train

prefix = 'result/val2'
if not os.path.exists(prefix): os.mkdir(prefix)

########################################################################
opt_train_8_10 = {
  'train_src': 'res/conclusion/training_20min_avg_travel_time.csv',
  'start_date': '2016-07-19',
  'end_date': '2016-10-10',
  'test_miss_start_time': '06:00',
  'test_miss_end_time': '09:40',
  'pred_start_time': '08:00',
  'pred_end_time': '09:40',
  'scale_model': prefix + '/scale_model_8_10',
  'out_train_feature_file': prefix + '/train_8_10'
}
print("[train 8-10]")
train(opt_train_8_10)

########################################################################
opt_train_17_19 = {
  'train_src': 'res/conclusion/training_20min_avg_travel_time.csv',
  'start_date': '2016-07-19',
  'end_date': '2016-10-10',
  'test_miss_start_time': '15:00',
  'test_miss_end_time': '18:40',
  'pred_start_time': '17:00',
  'pred_end_time': '18:40',
  'scale_model': prefix + '/scale_model_17_19',
  'out_train_feature_file': prefix + '/train_17_19'
}
print("[train 17-19]")
# train(opt_train_17_19)

########################################################################
opt_pred_8_10 = {
  'start_date': '2016-10-11',
  'end_date': '2016-10-17',
  'test_miss_start_time': '06:00',
  'test_miss_end_time': '07:40',
  'pred_start_time': '08:00',
  'pred_end_time': '09:40',
  'submit_file': prefix + '/submit_file_8_10.csv',
  'test_file': prefix + '/pred_8_10.feature',
  'train_file': opt_train_8_10['out_train_feature_file'] + '.scale',
  'scale_model': opt_train_8_10['scale_model'],
  'tmp_result_file': prefix + '/rvkde_8_10-test-result',
  'train_src': 'res/val/training_data.csv',
  'test_src': 'res/val/testing_data.csv'
}

