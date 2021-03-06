import numpy as np
import pandas as pd

from sklearn import metrics
from math import exp


class MetricsROC:
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
            Algorithm results with unix, Label features.
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

        # def udf_append_next_alarm(sample_pd):
        #     sample_pd['NextAlarm'] = v_get_next_alarm(sample_pd['unix'])
        #     sample_pd['TimeDistance'] = (sample_pd['NextAlarm'] - sample_pd['unix']) / 3600.0
        #     algorithm_results['WindowsDistance'] = algorithm_results['TimeDistance'] // window_length
        #     return sample_pd

        # def udf_append_next_alarm_single_negative_window(sample_pd):
        #     sample_pd['NextAlarm'] = v_get_next_alarm(sample_pd['unix'])
        #     sample_pd['TimeDistance'] = (sample_pd['NextAlarm'] - sample_pd['unix']) / 3600.0
        #     sample_pd['WindowsDistance'] = sample_pd['TimeDistance'] > window_length
        #     return sample_pd
        def get_next_alarm(timestamp):
            posterior_alarms = maintenances[maintenances > timestamp]
            if len(posterior_alarms) > 0:
                next_alarm = posterior_alarms[0]
            else:
                next_alarm = -1
            return next_alarm

        v_get_next_alarm = np.vectorize(get_next_alarm)

        algorithm_results['NextAlarm'] = v_get_next_alarm(algorithm_results['unix'])
        algorithm_results['TimeDistance'] = (algorithm_results['NextAlarm'] -
                                             algorithm_results['unix']) / 3600.0
        if(single_negative_window):
            algorithm_results['WindowsDistance'] = algorithm_results['TimeDistance'] > window_length
        else:
            algorithm_results['WindowsDistance'] = algorithm_results['TimeDistance'] // window_length

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

        def get_next_alarm(timestamp):
            posterior_alarms = maintenances[maintenances > timestamp]
            if len(posterior_alarms) > 0:
                next_alarm = posterior_alarms[0]
            else:
                next_alarm = -1
            return next_alarm

        v_get_next_alarm = np.vectorize(get_next_alarm)

        algorithm_results['NextAlarm'] = v_get_next_alarm(algorithm_results['unix'])
        algorithm_results['TimeDistance'] = (algorithm_results['NextAlarm'] - algorithm_results['unix']) / 3600.0

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
        algorithm_results = pd.DataFrame({'Score': algorithm_results\
                                          .groupby(['NextAlarm', 'WindowsDistance'])\
                                          .scores\
                                          .apply(self.agg_function)})\
                              .reset_index()

        algorithm_results['Label'] = np.logical_and(algorithm_results['WindowsDistance'] == 0,
                                                    algorithm_results['NextAlarm'] != -1)

        return algorithm_results

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

        if self.agg_method == "NAB":
            def nab_score(timestamp):
                ratio = timestamp/window_length
                ratio = max(ratio, 0.01)
                score = 2 / (1+exp(-15*ratio)) - 1
                return(score)

            v_nab_score = np.vectorize(nab_score)
            algorithm_results['NABWeight'] = v_nab_score(algorithm_results['TimeDistance'])
            algorithm_results['scores'] = algorithm_results['NABWeight']*algorithm_results['scores'] / np.sum(algorithm_results['NABWeight'])


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

        aggregated_results = algorithm_results.groupby('NextAlarm')

        if(single_negative_window):
            algorithm_results['WindowsDistance'] = algorithm_results\
                .TimeDistance.apply(lambda x: x > window_length)
        else:
            algorithm_results['WindowsDistance'] = algorithm_results\
                .TimeDistance.apply(lambda x: x // window_length)

        algorithm_results = algorithm_results[['NextAlarm', 'WindowsDistance', 'scores']]

        # aggregated_results = algorithm_results\
        #     .groupby('NextAlarm', 'WindowsDistance')\
        #     .apply(udf_compute_score)

        algorithm_results = pd.DataFrame({'Score': algorithm_results\
                                          .groupby(['NextAlarm', 'WindowsDistance'])\
                                          .scores\
                                          .apply(self.agg_function)})\
                              .reset_index()

        algorithm_results['Label'] = np.logical_and(algorithm_results['WindowsDistance'] == 0,
                                                    algorithm_results['NextAlarm'] != -1)

        algorithm_results = algorithm_results.sort_values('Label')
        fpr, tpr, threshold = metrics.roc_curve(algorithm_results['Label'],
                                                algorithm_results['Score'])

        threshold = np.sort(threshold)
        threshold[len(threshold)-1] = threshold[len(threshold)-2]

        precision = [metrics.precision_score(algorithm_results['Label'],
                                             algorithm_results['Score'] >= thres) for thres in threshold]
        recall = [metrics.recall_score(algorithm_results['Label'],
                                       algorithm_results['Score'] >= thres) for thres in threshold]
        f1 = [metrics.f1_score(algorithm_results['Label'],
                               algorithm_results['Score'] >= thres) for thres in threshold]
        auc = metrics.auc(fpr, tpr)

        return(fpr, tpr, precision, recall, f1, threshold, auc)

    def get_score(self, algorithm_results, maintenances, window_length=6,
                  single_negative_window=False):
        """
        Compute the score for algorithm results
        Parameters
        ----------
        algorithm_results:
            dataframe with unix, Label features
        maintenances:
            numpy array containing the maintenances timestamp
        window_lengh:
            double Hours length of the window previous to a maintenance considered as a positive instance. Default 6 hours.
        single_negative_window: boolean
            Boolean that represents if one or many negative windows should be considered.
        Return
        ------
        list with fpr, tpr, recall, f1 and thresholds to compute AUC
        auc
        """
        algorithm_results = self.__append_windows_to_next_alarm(
            algorithm_results, maintenances, window_length, single_negative_window
        )

        algorithm_results = self.__include_score(algorithm_results)

        algorithm_results = algorithm_results.sort_values('Label')
        fpr, tpr, threshold = metrics.roc_curve(algorithm_results['Label'],
                                                algorithm_results['Score'])
        recall = [metrics.recall_score(algorithm_results['Label'],
                                       algorithm_results['Score'] >= thres) for thres in threshold]
        f1 = [metrics.f1_score(algorithm_results['Label'],
                               algorithm_results['Score'] >= thres) for thres in threshold]
        auc = metrics.auc(fpr, tpr)

        return fpr, tpr, recall, f1, threshold, auc

    def get_score_all_windows(self, algorithm_results, maintenances,
                              windows=np.array([1, 2, 3, 4, 5, 6, 12, 18, 24, 36, 48]),
                              single_negative_window=False):
        """
        Compute the score for algorithm results
        Parameters
        ----------
        algorithm_results:
            dataframe with unix, Label  features
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

        algorithm_results = self.__append_next_alarm(
            algorithm_results, maintenances
        )

        if hasattr(self, 'threshold'):
            self.threshold = np.percentile(algorithm_results['scores'], 0.95)

        num_windows = len(windows)
        scores_list = pd.DataFrame({'tpr': [], 'fpr': [],
                                    'threshold': [], 'window_length': []})
        auc_list = np.zeros(num_windows)

        for i, window in enumerate(windows):
            print(window)
            fpr, tpr, precision, recall, f1, threshold, auc = self.__get_score_window(
                algorithm_results, window, single_negative_window
            )
            new_results = pd.DataFrame({
                'tpr': tpr, 'fpr': fpr,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'threshold': threshold,
                'window_length': np.repeat(window, len(tpr))
            })
            scores_list = pd.concat([scores_list, new_results])
            auc_list[i] = auc

        print(scores_list)
        return(scores_list, auc_list)

