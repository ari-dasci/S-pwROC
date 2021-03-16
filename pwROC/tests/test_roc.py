import unittest
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from evaluacion import MetricsROC
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
        self.scores = self.sc.createDataFrame(
            pd.DataFrame({
                'unix': timestamp,
                'scores': np.random.binomial(1, 0.1, size=300)
            })
        )
        self.maintenances = np.sort(timestamp[self.rng.randint(300, size=15)])[::-1]
        self.roc_metric = MetricsROC()

    def test_get_score(self):
        self.scores = self.scores.withColumn("Id", F.monotonically_increasing_id())\
             .repartitionByRange(8, 'Id')\
             .withColumn("Id", F.spark_partition_id())

        print(self.roc_metric.get_score(self.scores, self.maintenances))

    def test_get_score_all_windows(self):
        self.scores = self.scores.withColumn("Id", F.monotonically_increasing_id())\
             .repartitionByRange(8, 'Id')\
             .withColumn("Id", F.spark_partition_id())

        print(self.roc_metric.get_score_all_windows(self.scores, self.maintenances))

if __name__ == '__main__':
    unittest.main()
