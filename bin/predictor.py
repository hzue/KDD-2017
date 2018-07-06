from subprocess import check_output
from pprint import pprint
from abc import ABCMeta, abstractmethod
import pandas as pd
import math
import util

######################################################
class supervised_learning_interface(metaclass=ABCMeta):

  ## some note of metaclass
  #
  # c_name   is str
  # c_parent is tuple
  # c_attr   is dict
  #
  # def my_type(c_name, c_parent, c_attr):
  #     ...
  #   return type(c_name, c_parent, c_attr)
  #
  # class my_type(type):
  #   def __new__(cls, c_name, c_parent, c_attr):
  #     ...
  #     return super().__new__(cls, c_name, c_parent, c_attr)
  #     # return type(c_name, c_parent, c_attr)
  #
  # __metaclass__  = my_type (py2)

  @abstractmethod
  def fit(X, y):
      pass

  @abstractmethod
  def predict(X):
      pass

######################################################

class supervised_learning:

    prefix = "result/default"

    class rf(supervised_learning_interface):

        name                = 'Random Forest'
        test_file     = 'test.rf'
        train_file    = 'train.rf'
        model_file    = 'rf-train-model'
        score_file    = 'rf-feature-score'
        result_file = 'rf-test-result'

        def _gen_file(self, X, filepath, y=None):
            if y == None: y = [_ for _ in range(0, len(X))]
            f = open(filepath, 'w')
            for i, x in enumerate(X):
                f.write(str(y[i]) + ',')
                f.write(','.join(list(map(str, x))) + '\n')

        @util.flow_logger
        def fit(self, X, y):
            self._gen_file(X, "{0}/{1}".format(supervised_learning.prefix, self.train_file), y=y)
            feature_num = len(check_output('head -n 1 {0}/{1}'.format( \
                                           supervised_learning.prefix, self.train_file \
                                           ), shell=True).decode('utf-8').split(",")) - 1
            mtry = math.ceil(feature_num / 3)
            check_output('Rscript sbin/rf-train.r regression 400 {0} {1}/{2} {1}/{4} > {1}/{3}'.format( \
                         mtry, supervised_learning.prefix, self.train_file, self.score_file, self.model_file \
                         ), shell=True)

        @util.flow_logger
        def predict(self, X):
            self._gen_file(X, "{0}/{1}".format(supervised_learning.prefix, self.test_file))
            check_output('Rscript sbin/rf-predict.r regression {0}/{1} {0}/{2} > {0}/{3}'.format( \
                         supervised_learning.prefix, self.model_file, self.test_file, self.result_file \
                         ), shell=True)
            return self.read_result()

        def read_result(self):
            result_file = open("{0}/{1}".format(supervised_learning.prefix, self.result_file), 'r')
            results = [ float(line.strip()) for line in result_file.readlines()]
            result_file.close()
            return results


    class rvkde(supervised_learning_interface):

        name = 'RVKDE'
        test_file = 'test' # sub-filename may be 'feature' or 'scale'
        train_file = 'train' # sub-filename may be 'feature' or 'scale'
        scale_model_file = 'scale-model'
        model_file = 'rvkde-train-model'
        result_file = 'rvkde-test-result'

        def _gen_file(self, X, filepath, y=None):
            if y == None: y = [_ for _ in range(0, len(X))]
            f = open(filepath, 'w')
            for i, x in enumerate(X):
                f.write(str(y[i]))
                for ind, each_feature in enumerate(x):
                    f.write(" {}:{}".format(ind + 1, each_feature))
                f.write("\n")
            f.close()

        def fit(self, X, y):
            self._gen_file(X, "{}/{}.feature".format(supervised_learning.prefix, self.train_file), y=y)
            check_output("svm-scale -s {2}/{0} {2}/{1}.feature > {2}/{1}.scale".format( \
                         self.scale_model_file, self.train_file, supervised_learning.prefix \
                         ), shell=True)
            check_output("./sbin/rvkde --best --cv --regress -n 5 -v {0}/{1}.scale -b 10,15,1 --ks 1,34,2 --kt 1,100,4 > {0}/{2}".format( \
                         supervised_learning.prefix, self.train_file, self.model_file \
                         ), shell=True)

        def predict(self, X):
            self._gen_file(X, "{}/{}.feature".format(supervised_learning.prefix, self.test_file))
            check_output("svm-scale -r {2}/{0} {2}/{1}.feature > {2}/{1}.scale".format( \
                         self.scale_model_file, self.test_file, supervised_learning.prefix \
                         ), shell=True)
            [a, b, ks, kt, score] = check_output("head {0}/{1} -n 2 | tail -n 1".format( \
                                                 supervised_learning.prefix, self.model_file \
                                                 ), shell=True).decode('utf-8').split(' ')
            check_output("./sbin/rvkde --best --predict --regress -v {0}/{1}.scale -V {0}/{2}.scale -b {3} --ks {4} --kt {5} > {0}/{6}".format( \
                         supervised_learning.prefix, self.train_file, self.test_file, b, ks, kt, self.result_file \
                         ), shell=True)
            return self.read_result()

        def read_result(self):
            results = check_output("head {0}/{1} -n -2 | sed -e '1,3d'".format( \
                                    supervised_learning.prefix, self.result_file \
                                ), shell=True).decode('utf-8').split('\n')
            results.pop()
            results = [ float(_.split(' ')[1]) for _ in results]
            return results

    class svr(supervised_learning_interface):

        name = 'Support Vector Regressor'
        test_file = 'test' # sub-filename may be 'feature' or 'scale'
        train_file = 'train' # sub-filename may be 'feature' or 'scale'
        scale_model_file = 'scale-model'
        model_file = 'svr-train-model'
        result_file = 'svr-test-result'
        C = None

        def __init__(self, C=128):
            self.C = C

        def _gen_file(self, X, filepath, y=None):
            if y == None: y = [_ for _ in range(0, len(X))]
            f = open(filepath, 'w')
            for i, x in enumerate(X):
                f.write(str(y[i]))
                for ind, each_feature in enumerate(x):
                    f.write(" {}:{}".format(ind + 1, each_feature))
                f.write("\n")
            f.close()

        # @util.flow_logger
        def fit(self, X, y):
            self._gen_file(X, "{}/{}.feature".format(supervised_learning.prefix, self.train_file), y=y)
            check_output("svm-train -s 3 -c {3} {0}/{1}.feature {0}/{2}".format( \
                         supervised_learning.prefix, self.train_file, self.model_file, self.C \
                         ), shell=True)

        # @util.flow_logger
        def predict(self, X):
            self._gen_file(X, "{}/{}.feature".format(supervised_learning.prefix, self.test_file))
            check_output("svm-predict {0}/{1}.feature {0}/{2} {0}/{3}".format( \
                         supervised_learning.prefix, self.test_file, self.model_file, self.result_file \
                         ), shell=True)
            return self.read_result()

        def read_result(self):
            result_file = open("{0}/{1}".format(supervised_learning.prefix, self.result_file), 'r')
            results = [ float(line.strip()) for line in result_file.readlines()]
            result_file.close()
            return results

