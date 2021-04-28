import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyspark.sql.functions as F
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pwROC import MetricsROC
from pyspark.sql.types import DoubleType, StructType, StructField, IntegerType, BooleanType
from sklearn import metrics


def previous_interval_(last_maintenance):
    return F.udf(
        lambda timestamp: timestamp < last_maintenance,
        BooleanType()
    )


def in_interval(start, end):
    length_repetition = len(start)
    return F.udf(
        lambda timestamp:
        not(np.any(np.logical_and(np.repeat(timestamp, length_repetition) > start,
                                  np.repeat(timestamp, length_repetition) < end + 1800))),
        BooleanType()
    )


def filter_maintenances(results, maintenances):
    last_maintenance = int(maintenances.iloc[-1:, 0])
    results = results\
        .where(previous_interval_(last_maintenance)('unix'))\
        .where(in_interval(np.array(maintenances.unix_start),
                           np.array(maintenances.unix_end))('unix'))

    return(results)
