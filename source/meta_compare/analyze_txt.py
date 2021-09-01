import os
import pdb

import numpy as np
from fastestimator.summary.logs import parse_log_file
from scipy.stats import ttest_ind
from tabulate import tabulate


def get_best_step(objective, eval_steps, result, mode, train_history):
    obj_step = 0
    for idx, value in enumerate(result):
        if (mode == "max" and value >= objective) or (mode == "min"
                                                      and value <= objective):
            obj_step = eval_steps[idx]
            break
    upper_step = obj_step
    lower_step = eval_steps[idx - 1]
    min_loss = None
    min_train_step = None
    for train_step, train_loss in train_history.items():
        if train_step > lower_step and train_step <= upper_step:
            if min_loss is None:
                min_loss = train_loss
                min_train_step = train_step
            elif train_loss < min_loss:
                min_loss = train_loss
                min_train_step = train_step
    return min_train_step


def get_column_mean_std(all_data, best_mode, lr_schedules, arc_name="ARC"):
    if best_mode == "max":
        get_best = np.max
        get_worst = np.min
    elif best_mode == "min":
        get_best = np.min
        get_worst = np.max
    else:
        raise ValueError("best_mode needs to be one of ['max', 'min']")

    column_data = all_data
    best_numbers = []
    for lr_schedule in lr_schedules:
        lr_schedule_data = column_data[lr_schedule]
        for step, result, _ in lr_schedule_data:
            best_numbers.append(get_best(result))

    convergence_target = get_worst(best_numbers)

    br_dict, bs_dict = {}, {}
    for lr_schedule in lr_schedules:
        best_step, best_result = [], []
        lr_schedule_data = column_data[lr_schedule]
        for step, result, train_history in lr_schedule_data:
            best_result.append(get_best(result))
            best_step.append(
                get_best_step(convergence_target, step, result, best_mode,
                              train_history))

        br_dict[lr_schedule] = best_result
        bs_dict[lr_schedule] = best_step

    table = []
    for lr_schedule in lr_schedules:
        best_result = br_dict[lr_schedule]
        best_step = bs_dict[lr_schedule]

        br_display = f"{np.mean(best_result):.4f}"
        bs_display = f"{np.mean(best_step):.0f}"

        if np.mean(best_result) == get_best(
            [np.mean(x) for x in br_dict.values()]):
            br_display += "*"

        if np.mean(best_step) == min([np.mean(x) for x in bs_dict.values()]):
            bs_display += "*"

        if ttest_ind(br_dict[arc_name], br_dict[lr_schedule]).pvalue < 0.05:
            br_display += "#"

        if ttest_ind(bs_dict[arc_name], bs_dict[lr_schedule]).pvalue < 0.05:
            bs_display += "#"

        table.append([
            lr_schedule, br_display, f"{np.std(best_result):.4f}", bs_display,
            f"{np.std(best_step):.0f}"
        ])

    print(
        tabulate(table,
                 headers=[
                     "scheduler", "metric mean", "metric std", "step mean",
                     "step std"
                 ],
                 tablefmt="github"))


def get_column_median(all_data, best_mode, lr_schedules, arc_name="ARC"):
    if best_mode == "max":
        get_best = np.max
        get_worst = np.min
    elif best_mode == "min":
        get_best = np.min
        get_worst = np.max
    else:
        raise ValueError("best_mode needs to be one of ['max', 'min']")

    column_data = all_data
    best_numbers = []
    for lr_schedule in lr_schedules:
        lr_schedule_data = column_data[lr_schedule]
        for step, result, _ in lr_schedule_data:
            best_numbers.append(get_best(result))
    convergence_target = get_worst(best_numbers)

    br_dict, bs_dict = {}, {}
    for lr_schedule in lr_schedules:
        best_step, best_result = [], []
        lr_schedule_data = column_data[lr_schedule]
        for step, result, train_history in lr_schedule_data:
            best_result.append(get_best(result))
            best_step.append(
                get_best_step(convergence_target, step, result, best_mode,
                              train_history))

        br_dict[lr_schedule] = best_result
        bs_dict[lr_schedule] = best_step

    table = []
    for lr_schedule in lr_schedules:
        best_result = br_dict[lr_schedule]
        best_step = bs_dict[lr_schedule]

        br_display = f"{np.median(best_result):.4f}"
        bs_display = f"{np.median(best_step):.0f}"

        if np.median(best_result) == get_best(
            [np.median(x) for x in br_dict.values()]):
            br_display += "*"

        if np.median(best_step) == min(
            [np.median(x) for x in bs_dict.values()]):
            bs_display += "*"

        if ttest_ind(br_dict[arc_name], br_dict[lr_schedule]).pvalue < 0.05:
            br_display += "#"

        if ttest_ind(bs_dict[arc_name], bs_dict[lr_schedule]).pvalue < 0.05:
            bs_display += "#"

        table.append([
            lr_schedule,
            br_display,
            bs_display,
        ])

    print(
        tabulate(table,
                 headers=[
                     "scheduler",
                     "metric median",
                     "step median",
                 ],
                 tablefmt="github"))


def check_file_complete(folder_path):
    filenames = [
        fname for fname in os.listdir(folder_path) if fname.endswith(".txt")
    ]

    schedule_set = set()
    id_set = set()
    # get the set of lr, scheduler, id
    for filename in filenames:
        configs = os.path.splitext(filename)[0].split("_")
        lr_schedule_name, run_id = configs
        schedule_set.add(lr_schedule_name)
        id_set.add(run_id)

    # check all combinations exist
    for schedule in schedule_set:
        for run in id_set:
            filename = f"{schedule}_{run}.txt"
            assert os.path.exists(os.path.join(
                folder_path, filename)), f"{filename} is missing"


def print_table(folder_path, best_mode, metric_name, loss_name, mode):
    if mode == "mean_std":
        print_func = get_column_mean_std
    elif mode == "median":
        print_func = get_column_median
    else:
        raise ValueError("mode needs to be one of ['mean_std', 'median']")

    check_file_complete(folder_path)
    all_data = {}
    filenames = [
        fname for fname in os.listdir(folder_path) if fname.endswith(".txt")
    ]

    for filename in filenames:
        filepath = os.path.join(folder_path, filename)
        configs = os.path.splitext(filename)[0].split("_")
        lr_schedule_name, run_id = configs
        summary = parse_log_file(filepath, ".txt")
        result = np.array(
            [acc for acc in summary.history["eval"][metric_name].values()])
        steps = np.array(
            [acc for acc in summary.history["eval"][metric_name].keys()])
        train_history = summary.history["train"][loss_name]

        if lr_schedule_name not in all_data:
            all_data[lr_schedule_name] = []
        all_data[lr_schedule_name].append((steps, result, train_history))

    print_func(all_data,
               best_mode,
               lr_schedules=["sls", "superconvergence", "ARC"])


if __name__ == "__main__":
    print_table(
        mode="median",  # "median" or "mean_std"
        folder_path=
        "/mnt/c/Users/212770359/Downloads/ARC-master/iccv/logs/meta_comparison/language_modeling",  # path of the log dir
        best_mode="min",  # "max" or "min"
        metric_name="perplexity",  # evaluation metric
        loss_name="ce")  # loss key
