from subprocess import check_output
from pprint import pprint
import pandas as pd
import math
import util

class ml:

  prefix = "result/default"

  class rf:

    name        = 'Random Forest'
    test_file   = 'test.rf'
    train_file  = 'train.rf'
    model_file  = 'rf-train-model'
    score_file  = 'rf-feature-score'
    result_file = 'rf-test-result'

    def _gen_file(X, filepath, y=None):
      if y == None: y = [_ for _ in range(0, len(X))]
      f = open(filepath, 'w')
      for i, x in enumerate(X):
        f.write(str(y[i]) + ',')
        f.write(','.join(list(map(str, x))) + '\n')

    @classmethod
    @util.flow_logger
    def fit(cls, X, y):
      cls._gen_file(X, "{0}/{1}".format(ml.prefix, cls.train_file), y=y)
      feature_num = len(check_output('head -n 1 {0}/{1}'.format( \
                      ml.prefix, cls.train_file \
                    ), shell=True).decode('utf-8').split(",")) - 1
      mtry = math.ceil(feature_num / 3)
      check_output('Rscript sbin/rf-train.r regression 400 {0} {1}/{2} {1}/{4} > {1}/{3}'.format( \
                  mtry, ml.prefix, cls.train_file, cls.score_file, cls.model_file
                ), shell=True)

    @classmethod
    @util.flow_logger
    def predict(cls, X):
      cls._gen_file(X, "{0}/{1}".format(ml.prefix, cls.test_file))
      check_output('Rscript sbin/rf-predict.r regression {0}/{1} {0}/{2} > {0}/{3}'.format( \
                  ml.prefix, cls.model_file, cls.test_file, cls.result_file \
                ), shell=True)
      return cls.read_result()

    @classmethod
    def read_result(cls):
      result_file = open("{0}/{1}".format(ml.prefix, cls.result_file), 'r')
      results = [ float(line.strip()) for line in result_file.readlines()]
      result_file.close()
      return results


  class rvkde:

    name             = 'RVKDE'
    test_file        = 'test' # sub-filename may be 'feature' or 'scale'
    train_file       = 'train' # sub-filename may be 'feature' or 'scale'
    scale_model_file = 'scale-model'
    model_file       = 'rvkde-train-model'
    result_file      = 'rvkde-test-result'

    def _gen_file(X, filepath, y=None):
      if y == None: y = [_ for _ in range(0, len(X))]
      f = open(filepath, 'w')
      for i, x in enumerate(X):
        f.write(str(y[i]))
        for ind, each_feature in enumerate(x):
          f.write(" {}:{}".format(ind + 1, each_feature))
        f.write("\n")
      f.close()

    @classmethod
    @util.flow_logger
    def fit(cls, X, y):
      cls._gen_file(X, "{}/{}.feature".format(ml.prefix, cls.train_file), y=y)
      check_output("svm-scale -s {2}/{0} {2}/{1}.feature > {2}/{1}.scale".format( \
                  cls.scale_model_file, cls.train_file, ml.prefix \
                ), shell=True)
      check_output("./sbin/rvkde --best --cv --regress -n 5 -v {0}/{1}.scale -b 10,15,1 --ks 1,34,2 --kt 1,100,4 > {0}/{2}".format( \
                  ml.prefix, cls.train_file, cls.model_file \
                ), shell=True)

    @classmethod
    @util.flow_logger
    def predict(cls, X):
      cls._gen_file(X, "{}/{}.feature".format(ml.prefix, cls.test_file))
      check_output("svm-scale -r {2}/{0} {2}/{1}.feature > {2}/{1}.scale".format( \
                  cls.scale_model_file, cls.test_file, ml.prefix \
                ), shell=True)
      [a, b, ks, kt, score] = check_output("head {0}/{1} -n 2 | tail -n 1".format( \
                  ml.prefix, cls.model_file \
                ), shell=True).decode('utf-8').split(' ')
      check_output("./sbin/rvkde --best --predict --regress -v {0}/{1}.scale -V {0}/{2}.scale -b {3} --ks {4} --kt {5} > {0}/{6}".format( \
                  ml.prefix, cls.train_file, cls.test_file, b, ks, kt, cls.result_file \
                ), shell=True)
      return cls.read_result()

    @classmethod
    def read_result(cls):
      results = check_output("head {0}/{1} -n -2 | sed -e '1,3d'".format( \
                  ml.prefix, cls.result_file \
                ), shell=True).decode('utf-8').split('\n')
      results.pop()
      results = [ float(_.split(' ')[1]) for _ in results]
      return results

  class svr:

    name             = 'Support Vector Regressor'
    test_file        = 'test' # sub-filename may be 'feature' or 'scale'
    train_file       = 'train' # sub-filename may be 'feature' or 'scale'
    scale_model_file = 'scale-model'
    model_file       = 'svr-train-model'
    result_file      = 'svr-test-result'

    def _gen_file(X, filepath, y=None):
      if y == None: y = [_ for _ in range(0, len(X))]
      f = open(filepath, 'w')
      for i, x in enumerate(X):
        f.write(str(y[i]))
        for ind, each_feature in enumerate(x):
          f.write(" {}:{}".format(ind + 1, each_feature))
        f.write("\n")
      f.close()

    @classmethod
    @util.flow_logger
    def fit(cls, X, y):
      cls._gen_file(X, "{}/{}.feature".format(ml.prefix, cls.train_file), y=y)
      check_output("svm-scale -s {2}/{0} {2}/{1}.feature > {2}/{1}.scale".format( \
                  cls.scale_model_file, cls.train_file, ml.prefix \
                ), shell=True)
      check_output("svm-train -s 3 {0}/{1}.scale {0}/{2}".format( \
                    ml.prefix, cls.train_file, cls.model_file \
                ), shell=True)

    @classmethod
    @util.flow_logger
    def predict(cls, X):
      cls._gen_file(X, "{}/{}.feature".format(ml.prefix, cls.test_file))
      check_output("svm-scale -r {2}/{0} {2}/{1}.feature > {2}/{1}.scale".format( \
                  cls.scale_model_file, cls.test_file, ml.prefix \
                ), shell=True)
      check_output("svm-predict {0}/{1}.scale {0}/{2} {0}/{3}".format( \
                  ml.prefix, cls.test_file, cls.model_file, cls.result_file \
                ), shell=True)
      return cls.read_result()

    @classmethod
    def read_result(cls):
      result_file = open("{0}/{1}".format(ml.prefix, cls.result_file), 'r')
      results = [ float(line.strip()) for line in result_file.readlines()]
      result_file.close()
      return results

