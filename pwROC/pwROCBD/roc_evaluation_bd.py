import numpy as np
import pandas as pd

from sklearn import metrics
from pyspark.sql.types import DoubleType, StructType, StructField, IntegerType
from pyspark.sql import SparkSession
from math import exp


class MetricsROCBD:
    """
    ROC metric obtainance
    """
    def __init__(self, revert=False, agg_method="mean"):
        """
        Constructor of the class for ROC metric obtainance
        Parameters
        ----------
        revert: boolean
            Value indicating if low scores are the anomalous ones.
        agg_method: string
            Aggregation method. Currently supported: "mean" and "median".
        """
        self.revert = revert
        self.agg_method = agg_method
        if agg_method in ["mean", "NAB"]:
            self.agg_function = np.mean
        elif agg_method == "median":
            self.agg_function = np.median
        elif agg_method == "ccdf":
            self.threshold = 0.95
            self.agg_function = lambda x: np.mean(x > self.threshold)
        if revert:
            self.agg_function = lambda x: -self.agg_function(x)

    def __append_windows_to_next_alarm(self, algorithm_results, maintenances, window_length,
                                       single_negative_window=False):
        """
        Append the next alarm in the dataset, the distance to that alarm and the
        number of windows length to the next alarm
        Parameters
        ----------
        algoritm_results: dataframe
            Algorithm results with unix, Label and Id features.
        maintenances: dataframe
            Maintenances dataframe with unix column that represents the maintenances.
        window_length: float Time
            length window.
        single_negative_window: boolean
            Boolean that represents if one or many negative windows should be considered.
        Returns
        -------
        dataframe
            Input data frame with computed NextAlarm, TimeDistance and WindowsDistance
        """
        struct_fields = [StructField('unix', IntegerType(), True),
                         StructField('scores', DoubleType(), True),
                         StructField('Id', IntegerType(), True),
                         StructField('NextAlarm', IntegerType(), True),
                         StructField('TimeDistance', DoubleType(), True),
                         StructField('WindowsDistance', IntegerType(), True)
                         ]

        schema = StructType(struct_fields)

        def get_next_alarm(timestamp):
            posterior_alarms = maintenances[maintenances > timestamp]
            next_alarm = posterior_alarms[0] if len(posterior_alarms) > 0 else None
            return next_alarm

        v_get_next_alarm = np.vectorize(get_next_alarm)

        def udf_append_next_alarm(sample_pd):
            sample_pd['NextAlarm'] = v_get_next_alarm(sample_pd['unix'])
            sample_pd['TimeDistance'] = (sample_pd['NextAlarm'] - sample_pd['unix']) / 3600.0
            sample_pd['WindowsDistance'] = sample_pd['TimeDistance'] // window_length
            return sample_pd

        def udf_append_next_alarm_single_negative_window(sample_pd):
            sample_pd['NextAlarm'] = v_get_next_alarm(sample_pd['unix'])
            sample_pd['TimeDistance'] = (sample_pd['NextAlarm'] - sample_pd['unix']) / 3600.0
            sample_pd['WindowsDistance'] = sample_pd['TimeDistance'] > window_length
            return sample_pd

        if(single_negative_window):
            algorithm_results = algorithm_results.groupby('Id').\
                applyInPandas(udf_append_next_alarm_single_negative_window, schema)
        else:
            algorithm_results = algorithm_results.groupby('Id').\
                applyInPandas(udf_append_next_alarm, schema)

        return algorithm_results

    def __append_next_alarm(self, algorithm_results, maintenances):
        """
        Append the next alarm in the dataset, the distance to that alarm and the
        number of windows length to the next alarm
        Parameters
        ----------
        algoritm_results: dataframe
            Algorithm results with unix and Label features.
        maintenances: dataframe
            Maintenances dataframe with unix column that represents the maintenances.
        """
        struct_fields = [StructField('unix', IntegerType(), True),
                         StructField('scores', DoubleType(), True),
                         StructField('Id', IntegerType(), True),
                         StructField('NextAlarm', IntegerType(), True),
                         StructField('TimeDistance', DoubleType(), True)
                         ]

        schema = StructType(struct_fields)

        def get_next_alarm(timestamp):
            posterior_alarms = maintenances[maintenances > timestamp]
            if len(posterior_alarms) > 0:
                next_alarm = posterior_alarms[0]
            else:
                next_alarm = -1
            return next_alarm

        v_get_next_alarm = np.vectorize(get_next_alarm)

        def udf_append_next_alarm(sample_pd):
            sample_pd['NextAlarm'] = v_get_next_alarm(sample_pd['unix'])
            sample_pd['TimeDistance'] = (sample_pd['NextAlarm'] - sample_pd['unix']) / 3600.0
            return sample_pd

        algorithm_results = algorithm_results.groupby('Id').\
            applyInPandas(udf_append_next_alarm, schema)

        return algorithm_results

    def __include_score(self, algorithm_results):
        """
        Compute the proportion of anomalies in each interval
        Parameters
        ----------
        algoritm_results: dataframe
            Algorithm results with unix and Label features.
        Returns
        -------
        dataframe
            Input data grouped by NextAlarm,WindowsDistance and with Score feature
        """
        struct_fields = [StructField('NextAlarm', IntegerType(), True),
                         StructField('WindowsDistance', IntegerType(), True),
                         StructField('Score', DoubleType(), True),
                         StructField('Label', IntegerType(), True)]
        schema = StructType(struct_fields)


        def udf_score_by_window(sample_pd):
            agg_score = self.agg_function(sample_pd['scores'])
            next_alarm = sample_pd['NextAlarm'][0]

            score_by_window = pd.DataFrame({
                "NextAlarm": [next_alarm],
                "WindowsDistance": [sample_pd['WindowsDistance'][0]],
                "Score": [agg_score],
                "Label": [sample_pd['WindowsDistance'][0] == 0 and next_alarm != -1],
            })
            return score_by_window

        aggregated_results = algorithm_results\
            .groupby('NextAlarm', 'WindowsDistance')\
            .applyInPandas(udf_score_by_window, schema)

        return aggregated_results

    def __get_score_window(self, algorithm_results, window_length,
                           single_negative_window=False):
        """
        Compute the proportion of anomalies in each interval
        Parameters
        ----------
        algoritm_results: dataframe
            Algorithm results with unix and Label features.
        window_length: float
            Length of the window previous to a maintenance considered to be a positive
        single_negative_window : boolean
        Returns
        -------
        dataframe
            Input data grouped by NextAlarm,WindowsDistance and with Score feature
        """

        structs_window = [StructField('NextAlarm', IntegerType(), True),
                          StructField('WindowsDistance', IntegerType(), True),
                          StructField('scores', DoubleType(), True)]
        schema_window = StructType(structs_window)
        structs_score = [StructField('NextAlarm', IntegerType(), True),
                         StructField('WindowsDistance', IntegerType(), True),
                         StructField('Score', DoubleType(), True),
                         StructField('Label', IntegerType(), True)]
        schema_score = StructType(structs_score)

        def udf_compute_window(sample_pd):
            if self.agg_method == "NAB":
                def nab_score(timestamp):
                    ratio = timestamp/window_length

                    if ratio < 0.01:
                        ratio = 0.01

                    score = 2 / (1+exp(-15*ratio)) - 1
                    return(score)

                v_nab_score = np.vectorize(nab_score)
                sample_pd['NABWeight'] = v_nab_score(sample_pd['TimeDistance'])
                sample_pd['scores'] = sample_pd['NABWeight']*sample_pd['scores'] / np.sum(sample_pd['NABWeight'])

            sample_pd['WindowsDistance'] = sample_pd['TimeDistance'] // window_length
            return(sample_pd[['NextAlarm', 'WindowsDistance', 'scores']])

        def udf_compute_window_single_negative_window(sample_pd):
            if self.agg_method == "NAB":
                def nab_score(timestamp):
                    if timestamp > window_length:
                        score = 1
                    else:
                        score = 2 / (1+exp(-15*timestamp/window_length)) - 1
                    return(score)

                v_nab_score = np.vectorize(nab_score)
                sample_pd['NABWeight'] = v_nab_score(sample_pd['TimeDistance'])
                sample_pd['scores'] = sample_pd['NABWeight']*sample_pd['scores'] / np.sum(sample_pd['NABWeight'])
            sample_pd['WindowsDistance'] = sample_pd['TimeDistance'] > window_length
            return(sample_pd[['NextAlarm', 'WindowsDistance', 'scores']])

        def udf_compute_score(sample_pd):
            agg_score = self.agg_function(sample_pd['scores'])
            windows_distance = sample_pd['WindowsDistance'][0]
            next_alarm = sample_pd['NextAlarm'][0]
            score_by_window = pd.DataFrame({
                "NextAlarm": [next_alarm],
                "WindowsDistance": [windows_distance],
                "Score": [agg_score],
                "Label": [windows_distance == 0 and next_alarm != -1]
            })
            return score_by_window

        algorithm_results = algorithm_results.groupby('NextAlarm')

        if(single_negative_window):
            aggregated_results = algorithm_results\
                .applyInPandas(udf_compute_window_single_negative_window, schema_window)
        else:
            aggregated_results = algorithm_results\
                .applyInPandas(udf_compute_window, schema_window)

        aggregated_results = aggregated_results\
            .groupby('NextAlarm', 'WindowsDistance')\
            .applyInPandas(udf_compute_score, schema_score)\
            .toPandas()

        aggregated_results = aggregated_results.sort_values('Label')
        fpr, tpr, threshold = metrics.roc_curve(aggregated_results['Label'],
                                                aggregated_results['Score'])

        auc = metrics.auc(fpr, tpr)
        return(fpr, tpr, threshold, auc)

    def get_score(self, algorithm_results, maintenances, window_length=6,
                  single_negative_window=False):
        """
        Compute the score for algorithm results
        Parameters
        ----------
        algorithm_results:
            dataframe with unix, Label and Id features
        maintenances:
            numpy array containing the maintenances timestamp
        window_lengh:
            double Hours length of the window previous to a maintenance considered as a positive instance. Default 6 hours.
        single_negative_window: boolean
            Boolean that represents if one or many negative windows should be considered.
        Return
        ------
        list with fpr, tpr and thresholds to compute AUC
        auc
        """
        algorithm_results = self.__append_windows_to_next_alarm(
            algorithm_results, maintenances, window_length, single_negative_window
        )

        algorithm_results = self.__include_score(algorithm_results).toPandas()

        algorithm_results = algorithm_results.sort_values('Label')
        fpr, tpr, threshold = metrics.roc_curve(algorithm_results['Label'],
                                                algorithm_results['Score'])

        auc = metrics.auc(fpr, tpr)

        return fpr, tpr, threshold, auc

    def get_score_all_windows(self, algorithm_results, maintenances,
                              min_window_length=1, max_window_length=48,
                              num_windows=-1, single_negative_window=False):
        """
        Compute the score for algorithm results
        Parameters
        ----------
        algorithm_results:
            dataframe with unix, Label and Id features
        maintenances:
            dataframe with unix feature, containing the maintenances
        min_window_length:
            integer Number of minimum of hours previous to a maintenance to be considered an anomaly
        max_window_length:
            integer Number of maximum of hours previous to a maintenance to be considered an anomaly
        num_windows:
            integer Number of different window lengths to compute multiple AUCs
        single_negative_window: boolean
            Boolean that represents if one or many negative windows should be considered.
        Return
        ------
        List of lists with fpr, tpr and thresholds to compute AUC
        List of AUCs
        """

        spark = SparkSession.builder.getOrCreate()
        algorithm_results = self.__append_next_alarm(
            algorithm_results, maintenances
        )

        if hasattr(self, 'threshold'):
            self.threshold = algorithm_results\
                .selectExpr('percentile_approx(scores, 0.95)')\
                .collect()[0][0]

        if num_windows == -1:
            windows = np.array([1, 2, 3, 4, 5, 6, 12, 18, 24, 36, 48])
            num_windows = len(windows)
        else:
            max_window = min([algorithm_results.agg({"TimeDistance": "max"})\
                              .collect()[0]["max(TimeDistance)"],
                              max_window_length])
            min_window = max([algorithm_results.agg({"TimeDistance": "min"})\
                              .collect()[0]["min(TimeDistance)"],
                              min_window_length])
            windows = np.linspace(min_window, max_window, num=num_windows)

        scores_list = pd.DataFrame({'tpr': [], 'fpr': [],
                                    'threshold': [], 'window_length': []})
        auc_list = np.zeros(num_windows)

        for i, window in enumerate(windows):
            print(window)
            fpr, tpr, threshold, auc = self.__get_score_window(
                algorithm_results, window, single_negative_window
            )
            new_results = pd.DataFrame({
                'tpr': tpr, 'fpr': fpr,
                'threshold': threshold,
                'window_length': np.repeat(window, len(tpr))
            })
            scores_list = pd.concat([scores_list, new_results])
            auc_list[i] = auc
            spark.catalog.clearCache()

        return(scores_list, auc_list)

