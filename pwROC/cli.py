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
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from pwROC import MetricsROC, filter_maintenances
import glob


def save_roc_figure(algorithm_roc, algorithm_name, auc, fig_path):
    roc_plot = sns.relplot(x="fpr", y="tpr", hue="algorithm",
                           style="algorithm", kind="line",
                           data=algorithm_roc)
    roc_plot._legend.texts[0].set_text(algorithm_name + " auc=" + str(round(auc, 2)))
    roc_plot.savefig(fig_path + "-roc_auc.png")
    plt.close()


def get_curve_metrics_path(algorithm_folder, window_size, filter_planned, agg_method):
    results_path = algorithm_folder + "metrics-" + str(window_size) +\
        agg_method
    fig_path = algorithm_folder + "img/curve-" +\
        str(window_size) + "-" + agg_method + "NoBD"

    if filter_planned:
        results_path = results_path + "-FM"
        fig_path = fig_path + "-FM"

    return results_path, fig_path


def get_surface_metrics_path(algorithm_folder, agg_method, filter_planned):
    results_path = algorithm_folder + "surface-metrics-" +\
        agg_method
    if filter_planned:
        results_path = results_path + "-FM"

    return results_path


def plot_surface(results):
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


def summarise_surface(algorithm_folder, agg_method, filter_planned):
    results_path = algorithm_folder + "surface-metrics-" +\
        agg_method
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

    algorithm_name = arguments['<algorithm>']
    algorithm_folder = "results/" + algorithm_name + "/"

    filter_planned = arguments['--filter_planned']
    maintenances = pd.read_csv('data/maintenances.csv')

    if not arguments['filter']:
        agg_method = arguments['--agg_method']
        window_size = float(arguments['--window_size'])
    if arguments['filter'] or arguments['roc_curve'] or arguments['roc_surface']:
        scores = pd.concat([pd.read_csv(f) for f in
                            glob.glob(algorithm_folder + "day*/*.csv")],
                           ignore_index=True)

        if arguments['filter']:
            # alg_folder = "results/" + algorithm_name + "-FM/"
            scores = filter_maintenances(scores, maintenances)
        else:
            roc_metric = MetricsROC(agg_method=agg_method)

    if filter_planned:
        maintenances = maintenances[maintenances['maint_id'] != 'M69']
    if arguments['roc_curve']:
        # Prepare result path
        results_path, fig_path = get_curve_metrics_path(
            algorithm_folder, window_size, filter_planned, agg_method
        )

        # Compute preceding window ROC curve
        fpr, tpr, threshold, auc = roc_metric.get_score(
            scores, np.array(maintenances.unix_start),
            window_length=window_size
        )
        algorithm_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': threshold})
        algorithm_roc.to_csv(results_path + ".csv", index=False)

        algorithm_roc['Algorithm'] = algorithm_name
        auc = metrics.auc(algorithm_roc.fpr, algorithm_roc.tpr)

        # Save summarised results
        summary_results = pd.read_csv('results/summary.csv',
                                      index_col=['Algorithm', 'WindowSize'])
        random_result = pd.DataFrame({'tpr': [0, 1], 'fpr': [0, 1],
                                      'Algorithm': ['random', 'random']})
        summary_results.loc[(algorithm_name, window_size), 'AUC'] = auc
        summary_results.to_csv("./results/summary.csv")

        # Generate ROC Curve
        algorithm_roc = algorithm_roc.append(random_result)
        save_roc_figure(algorithm_roc, algorithm_name, auc, fig_path)

    if arguments['roc_surface']:
        num_windows = int(arguments['--num_windows'])
        results_path = get_surface_metrics_path(algorithm_folder, agg_method,
                                                filter_planned)

        scores_list, auc_list = roc_metric.get_score_all_windows(
            scores, np.array(maintenances.unix_start), num_windows=num_windows
        )

        # Save results
        print(auc_list)
        pd.DataFrame(scores_list)\
            .to_csv(results_path + ".csv", index=False)
    if arguments['open_surface']:
        # Read results
        results_path = get_surface_metrics_path(algorithm_folder,
                                                agg_method, filter_planned)
        results = pd.read_csv(results_path + ".csv")
        results = results[['fpr', 'window_length', 'tpr']]
        plot_surface(results)

    if arguments['summarise_surface']:
        summarise_surface(algorithm_folder, agg_method, filter_planned)
