from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler as Scaler
from predictor import supervised_learning
import numpy as np
import os
import file_handler as fh
import data
import feature_selection
import sys
import math
import util
import gc
import sys
np.seterr(all='ignore')

if __name__ == '__main__':

    ####### simple parse argv #######
    arg = {'prefix': 'default'}
    for i in range(1, len(sys.argv) - 1, 2):
        arg[sys.argv[i].replace("-", "")] = sys.argv[i+1]

    ####### setting #######
    submit_file_name = 'submit.csv'
    arg['prefix'] = 'result/' + arg['prefix']
    supervised_learning.prefix = arg['prefix']

    if arg['mode'] == 'validation':
        train_start_date = '2016-07-19'
        train_end_date = '2016-10-17'
        test_start_date = '2016-10-18'
        test_end_date = '2016-10-24'

    elif arg['mode'] == 'submit':
        train_start_date = '2016-07-19'
        train_end_date = '2016-10-24'
        test_start_date = '2016-10-25'
        test_end_date = '2016-10-31'

    else: raise ValueError('No this mode!')

    ####### check file path #######
    if not os.path.exists(arg['prefix']): os.mkdir(arg['prefix'])

    ####### generate X, y, test_X, test_y #######
    X, y, test_X, test_y, test_info_map = data.generate_data( \
                    'res/conclusion/training_data_phase2.csv', \
                    'res/conclusion/testing_data_{}_{}.csv'.format(test_start_date, test_end_date), \
                    train_start_date, train_end_date, test_start_date, test_end_date)

    ####### scale feature #######
    scale = Scaler(feature_range=(-1,1))
    X = scale.fit_transform(X)
    test_X = scale.transform(test_X)

    ####### feature selection #######
    # selected_list = feature_selection.forward_selection( \
    #     X, y, KNeighborsRegressor(n_neighbors=20, weights='distance', n_jobs=2))
    # exit()

    svr_selected_list = [58, 42, 24, 39, 47, 59, 53, 28, 29, 34, 1, 8, 52, 48, 3, 50, 16, 9, 62]
    knn_selected_list = [41, 1, 13, 21, 33, 3, 31, 0, 67, 59, 35, 37, 9, 11, 15, 16, 34, 4, 5, 7, 36,
         12, 40, 2, 6, 8, 10, 18, 19, 22, 32, 42, 20, 14, 23, 17, 43, 46, 45, 48, 53, 30, 50, 52, 64,
         66, 62, 44, 63, 65, 47, 51, 49, 28, 56, 55]
    rf_selected_list = [41, 59, 65, 6, 57, 67, 37, 38, 7, 17, 35, 20]

    svr_X, svr_test_X = feature_selection.transform([X, test_X], svr_selected_list)
    knn_X, knn_test_X = feature_selection.transform([X, test_X], knn_selected_list)
    rf_X, rf_test_X = feature_selection.transform([X, test_X], rf_selected_list)

    ####### build model #######

    svr = supervised_learning.svr()
    svr.fit(svr_X, y)
    svr_test_y = svr.predict(svr_test_X)

    knn = KNeighborsRegressor(n_neighbors=40, weights='distance')
    knn.fit(knn_X, y)
    knn_test_y = knn.predict(knn_test_X)

    test_y = []
    for i in range(len(knn_test_y)): test_y.append((knn_test_y[i] + svr_test_y[i]) / 2)

    fh.write_submit_file(test_info_map, test_y, arg['prefix'], submit_file_name)
    if arg['mode'] == 'validation':
        mape = util.evaluation('{}/{}'.format(arg['prefix'], submit_file_name), 'res/conclusion/testing_ans_2016-10-18_2016-10-24.csv')
        print("mape: {}".format(str(mape)))

    gc.collect()

