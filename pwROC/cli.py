#!/ucsr/bin/env python3
"""pwROC - Evaluation tools for Anomaly Detection Algorithms using the preceding window ROC
Usage:
    pwROC-cli filter <algorithm> [--header=header] [--format=format] [--filter_planned=filter_planned]
    pwROC-cli roc_curve <algorithm> [--window_size=window_size] [--header=header] [--format=format] [--agg_method=agg_method] [--filter_planned=filter_planned]
    pwROC-cli roc_surface <algorithm> [--num_windows=num_windows] [--header=header] [--format=format] [--agg_method=agg_method] [--filter_planned=filter_planned]
    pwROC-cli open_surface <algorithm> [--agg_method=agg_method] [--filter_planned=filter_planned]
    pwROC-cli summarise_surface <algorithm> [--agg_method=agg_method] [--filter_planned=filter_planned]
    pwROC-cli (-h | --help)

Options:
    <algorithm>                 Name of the folder with the results to evaluate.
    --window_size=window_size   Size of the window previous to a maintenance to be considered as an anomaly [default: 6].
    --header=header             Original files include the header [default: False].
    --format=format             Format of the results [default: us].
    --agg_method=agg_def        Aggregation method [default: mean].
    --filter_planned=filter_planned Filter planned maintenances [default: False].
    --num_windows=num_windows   Num windows [default: -1].
    -h --help                   Show this screen.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from docopt import docopt
from sklearn import metrics
from pwROC import MetricsROC, filter_maintenances
from matplotlib import cm
from matplotlib.ticker import MaxNLocator



def summarise_surface(alg_name, agg_method, filter_planned):
    results_path = "results/" + alg_name + "/surface-metrics-" + agg_method
    if filter_planned:
        results_path = results_path + "-FM"
        results = pd.read_csv(results_path + ".csv")
        results = results[['fpr', 'window_length', 'tpr']]

    results_by_window = results.groupby(['window_length'])
    summarise = pd.DataFrame({'Window': [], 'AUC': []})
    for window, frame in results_by_window:
        auc = metrics.auc(frame.fpr, frame.tpr)
        summarise = summarise.append(pd.DataFrame({
            'Window': [window], 'AUC': [auc]
        }))

    summarise.to_csv(results_path + "-summarise.csv", index=False)
def main():
    arguments = docopt(__doc__)

    alg_name = arguments['<algorithm>']
    alg_folder = "results/" + alg_name + "/"

    filter_planned = arguments['--filter_planned']
    maintenances = pd.read_csv('data/maintenances.csv')

    results = pd.read_csv('results/summary.csv', index_col=['Algorithm', 'WindowSize'])
    random_result = pd.DataFrame({'tpr': [0, 1], 'fpr': [0, 1], 'Algorithm': ['random', 'random']})

    if arguments['--format'] == "us":
        schema = StructType([StructField("unix", IntegerType()),
                             StructField("scores", DoubleType())])
    elif arguments['--format'] == "TLS":
        schema = StructType([StructField("Timestamp", IntegerType()),
                             StructField("Label", DoubleType()),
                             StructField("Score", DoubleType())])
    elif arguments['--format'] == "TL":
        schema = StructType([StructField("Timestamp", IntegerType()),
                             StructField("Label", DoubleType())])

    if not arguments['filter']:
        agg_method = arguments['--agg_method']
        window_size = float(arguments['--window_size'])
    if arguments['filter'] or arguments['roc_curve'] or arguments['roc_surface']:
        scores = pd.read_csv(alg_folder + "results.csv")

        if arguments['filter']:
            alg_folder = "results/" + alg_name + "-FM/"
            scores = filter_maintenances(scores, maintenances)
        else:
            roc_metric = MetricsROC(agg_method=agg_method)

    if filter_planned:
        maintenances = maintenances[maintenances['maint_id'] != 'M69']
    if arguments['roc_curve']:
        # Prepare result path
        results_path = "results/" + alg_name + "/metrics-" +\
            str(window_size) + agg_method
        fig_path = "./Informes/Anomalias y Deep Learning/img/" +\
            alg_name + "-" + str(window_size) + "-" +\
            agg_method

        if filter_planned:
            results_path = results_path + "-FM"
            fig_path = fig_path + "-FM"

        # Compute results
        fpr, tpr, threshold, auc = roc_metric.get_score(
            scores, np.array(maintenances.unix_start),
            window_length=window_size
        )
        alg_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': threshold})

        alg_roc.to_csv(results_path + ".csv", index=False)

        alg_roc['Algorithm'] = alg_name
        auc = metrics.auc(alg_roc.fpr, alg_roc.tpr)

        # Save results
        results.loc[(alg_name, window_size), 'AUC'] = auc
        results.to_csv("./results/summary.csv")

        # Generate ROC Curve
        alg_roc = alg_roc.append(random_result)
        roc_plot = sns.relplot(x="fpr", y="tpr", hue="Algorithm",
                               style="Algorithm", kind="line", data=alg_roc)
        roc_plot._legend.texts[0].set_text(alg_name + " AUC=" + str(round(auc, 2)))
        roc_plot.savefig(fig_path + "-roc_auc.png")
        plt.close()

    if arguments['roc_surface']:
        num_windows = int(arguments['--num_windows'])
        results_path = "results/" + alg_name + "/surface-metrics-" + agg_method
        if filter_planned:
            results_path = results_path + "-FM"

        scores_list, auc_list = roc_metric.get_score_all_windows(
            scores, np.array(maintenances.unix_start), num_windows=num_windows
        )

        # Save results
        print(auc_list)
        pd.DataFrame(scores_list)\
            .to_csv(results_path + ".csv", index=False)
    if arguments['open_surface']:
        # Read results
        results_path = "results/" + alg_name + "/surface-metrics-" + agg_method
        if filter_planned:
            results_path = results_path + "-FM"
        results = pd.read_csv(results_path + ".csv")
        results = results[['fpr', 'window_length', 'tpr']]

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_trisurf(results.fpr, results.window_length, results.tpr,
                               cmap=cm.jet, linewidth=0)
        fig.colorbar(surf)

        ax.set_xlabel('FPR', fontsize=20)
        ax.set_ylabel('Window', fontsize=20)
        ax.set_zlabel('TPR', fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.zaxis.set_major_locator(MaxNLocator(5))

        fig.tight_layout()

        plt.show()
    if arguments['summarise_surface']:
        summarise_surface(alg_name, agg_method, filter_planned)
