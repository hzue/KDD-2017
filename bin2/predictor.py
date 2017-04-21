from subprocess import check_output
from pprint import pprint
import pandas as pd
import math

class ml:

  class rf:

    test_file = 'test.rf'
    train_file = 'train.rf'
    model_file = 'rf-train-model'
    score_file = 'rf-feature-score'
    result_file = 'rf-test-result'

    def _gen_file(X, y, filepath):
      f = open(filepath, 'w')
      for i, x in enumerate(X):
        f.write(str(y[i]) + ',')
        f.write(','.join(list(map(str, x))) + '\n')

    @classmethod
    def train(cls, X, y, prefix):
      cls._gen_file(X, y, "{0}/{1}".format(prefix, cls.train_file))
      feature_num = len(check_output('head -n 1 {0}/{1}'.format( \
                      prefix, cls.train_file \
                    ), shell=True).decode('utf-8').split(",")) - 1
      mtry = math.ceil(feature_num / 3)
      check_output('Rscript sbin/rf-train.r regression 100 {0} {1}/{2} {1}/{4} > {1}/{3}'.format( \
                  mtry, prefix, cls.train_file, cls.score_file, cls.model_file
                ), shell=True)

    @classmethod
    def predict(cls, X, y, prefix):
      cls._gen_file(X, y, "{0}/{1}".format(prefix, cls.test_file))
      check_output('Rscript sbin/rf-predict.r regression {0}/{1} {0}/{2} > {0}/{3}'.format( \
                  prefix, cls.model_file, cls.test_file, cls.result_file \
                ), shell=True)

    @classmethod
    def read_result(cls, prefix, submit_file_name):
      result_file = open("{0}/{1}".format(prefix, cls.result_file), 'r')
      results = [ float(line.strip()) for line in result_file.readlines()]
      result_file.close()
      return results


  class rvkde:

    test_file        = 'test' # sub-filename may be 'feature' or 'scale'
    train_file       = 'train' # sub-filename may be 'feature' or 'scale'
    scale_model_file = 'scale-model'
    model_file       = 'rvkde-train-model'
    result_file      = 'rvkde-test-result'

    def _gen_file(X, y, filepath):
      f = open(filepath, 'w')
      for i, x in enumerate(X):
        f.write(str(y[i]))
        for ind, each_feature in enumerate(x):
          f.write(" {}:{}".format(ind + 1, each_feature))
        f.write("\n")
      f.close()

    @classmethod
    def train(cls, X, y, prefix):
      cls._gen_file(X, y, "{}/{}.feature".format(prefix, cls.train_file))
      check_output("svm-scale -s {2}/{0} {2}/{1}.feature > {2}/{1}.scale".format( \
                  cls.scale_model_file, cls.train_file, prefix \
                ), shell=True)
      check_output("./sbin/rvkde --best --cv --regress -n 5 -v {0}/{1}.scale -b 10,15,1 --ks 1,34,2 --kt 1,100,4 > {0}/{2}".format( \
                  prefix, cls.train_file, cls.model_file \
                ), shell=True)

    @classmethod
    def predict(cls, X, y, prefix):
      cls._gen_file(X, y, "{}/{}.feature".format(prefix, cls.test_file))
      check_output("svm-scale -r {2}/{0} {2}/{1}.feature > {2}/{1}.scale".format( \
                  cls.scale_model_file, cls.test_file, prefix \
                ), shell=True)
      [a, b, ks, kt, score] = check_output("head {0}/{1} -n 2 | tail -n 1".format( \
                  prefix, cls.model_file \
                ), shell=True).decode('utf-8').split(' ')
      check_output("./sbin/rvkde --best --predict --regress -v {0}/{1}.scale -V {0}/{2}.scale -b {3} --ks {4} --kt {5} > {0}/{6}".format( \
                  prefix, cls.train_file, cls.test_file, b, ks, kt, cls.result_file \
                ), shell=True)

    @classmethod
    def read_result(cls, prefix, submit_file_name):
      results = check_output("head {0}/{1} -n -2 | sed -e '1,3d'".format( \
                  prefix, cls.result_file \
                ), shell=True).decode('utf-8').split('\n')
      results.pop()
      results = [ float(_.split(' ')[1]) for _ in results]
      return results

