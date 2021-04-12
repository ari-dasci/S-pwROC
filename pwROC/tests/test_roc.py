import unittest
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pwROC import MetricsROCBD, MetricsROC
from pyspark.sql import SparkSession


class TestSparkExplOut(unittest.TestCase):
    def setUp(self):
        """
        Set Up results and maintenances data sets
        """
        self.rng = np.random.RandomState(0)
        self.sc = SparkSession.builder\
                              .appName("AppTest") \
                              .master("local[8]") \
                              .getOrCreate()
        timestamp = pd.date_range('2020-01-01', periods=300, freq='H').astype(np.int64) // 10**9
        self.scores = pd.DataFrame({'unix': timestamp,
                                    'scores': np.random.binomial(1, 0.1, size=300)})
        self.scores_bd = self.sc.createDataFrame(
            self.scores
        )
        self.maintenances = np.sort(timestamp[self.rng.randint(300, size=15)])[::-1]
        self.roc_metric_bd = MetricsROCBD()
        self.roc_metric = MetricsROC()

    def test_get_score_bd(self):
        self.scores_bd = self.scores_bd.withColumn("Id", F.monotonically_increasing_id())\
             .repartitionByRange(8, 'Id')\
             .withColumn("Id", F.spark_partition_id())

        print(self.roc_metric_bd.get_score(self.scores_bd, self.maintenances))

    def test_get_score_all_windows_bd(self):
        self.scores_bd = self.scores_bd.withColumn("Id", F.monotonically_increasing_id())\
             .repartitionByRange(8, 'Id')\
             .withColumn("Id", F.spark_partition_id())

        print(self.roc_metric_bd.get_score_all_windows(self.scores_bd, self.maintenances))

    def test_append_next_alarm(self):
        scores_with_alarm = self.roc_metric.\
            _MetricsROC__append_next_alarm(self.scores, self.maintenances)
        self.assertEqual(scores_with_alarm.shape, (300, 4))
    def test_append_windows_to_next_alarm(self):
        scores_with_alarm = self.roc_metric.\
            _MetricsROC__append_windows_to_next_alarm(self.scores, self.maintenances, 6)
        self.assertEqual(scores_with_alarm.shape, (300, 5))
    def test_include_score(self):
        scores_with_alarm = self.roc_metric.\
            _MetricsROC__append_windows_to_next_alarm(self.scores, self.maintenances, 6)
        aggregated_scores = self.roc_metric.\
            _MetricsROC__include_score(scores_with_alarm)
        self.assertEqual(aggregated_scores.shape, (51, 4))
    def test_get_score(self):
        fpr, _, _, auc = self.roc_metric.get_score(self.scores, self.maintenances, window_length=2)
        self.assertEqual(len(fpr), 4)
        self.assertEqual(type(auc), np.float64)

    def test_get_score_all_windows(self):
        scores_list, auc_list = self.roc_metric.get_score_all_windows(
            self.scores, self.maintenances,
            min_window_length=1, max_window_length=2,
            num_windows=2
        )
        self.assertEqual(scores_list.shape, (7, 4))
        self.assertEqual(len(auc_list), 2)


if __name__ == '__main__':
    unittest.main()
