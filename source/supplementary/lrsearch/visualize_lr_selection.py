import numpy as np
from fastestimator.summary.logs import parse_log_file
from matplotlib import pyplot as plt


def visualize_image_classification(
        file_path="../../../logs/supplementary/lr_search/image_classification.txt",
        file_extension=".txt"):
    summary = parse_log_file(file_path, file_extension)
    acc_steps = [x[0] for x in summary.history["eval"]["accuracy"].items()]
    acc_values = [x[1] for x in summary.history["eval"]["accuracy"].items()]
    best_step, best_acc = max(summary.history["eval"]["accuracy"].items(),
                              key=lambda k: k[1])
    max_lr = summary.history["train"]["model_lr"][best_step]
    lr_values = [summary.history["train"]["model_lr"][x] for x in acc_steps]
    assert len(lr_values) == len(acc_values)
    plt.plot(lr_values, acc_values)
    plt.plot(max_lr,
             best_acc,
             'o',
             color='r',
             label="Best Acc={}, LR={}".format(best_acc, max_lr))
    plt.xlabel("Learning Rate")
    plt.ylabel("Evaluation Accuracy")
    plt.legend(loc='upper left', frameon=False)
    plt.title("Image Classification Learning Rate Search")
    # plt.show()
    plt.savefig("lr_search_ic.png", dpi=300)


def visualize_instance_detection(
        file_path="../../../logs/supplementary/lr_search/instance_detection.txt",
        file_extension=".txt"):
    summary = parse_log_file(file_path, file_extension)
    acc_steps = [x[0] for x in summary.history["eval"]["total_loss"].items()]
    loss_values = [x[1] for x in summary.history["eval"]["total_loss"].items()]
    best_step, best_acc = min(summary.history["eval"]["total_loss"].items(),
                              key=lambda k: k[1])
    max_lr = summary.history["train"]["model_lr"][best_step]
    lr_values = [summary.history["train"]["model_lr"][x] for x in acc_steps]
    assert len(lr_values) == len(loss_values)
    plt.plot(lr_values, loss_values)
    plt.plot(max_lr,
             best_acc,
             'o',
             color='r',
             label="Best Loss={}, LR={}".format(best_acc, max_lr))
    plt.xlim(0.0, 0.022)
    plt.ylim(1.7, 2.5)
    plt.xticks(np.arange(0.0, 0.022, 0.005))
    plt.xlabel("Learning Rate")
    plt.ylabel("Evaluation Loss")
    plt.legend(frameon=False)
    plt.title("Instance Detection Learning Rate Search")
    # plt.show()
    plt.savefig("lr_search_id.png", dpi=300)


def visualize_language_modeling(
        file_path="../../../logs/supplementary/lr_search/language_modeling.txt",
        file_extension=".txt"):
    summary = parse_log_file(file_path, file_extension)
    acc_steps = [x[0] for x in summary.history["eval"]["perplexity"].items()]
    perplexity_values = [
        x[1] for x in summary.history["eval"]["perplexity"].items()
    ]
    best_step, best_acc = min(summary.history["eval"]["perplexity"].items(),
                              key=lambda k: k[1])
    max_lr = summary.history["train"]["model_lr"][best_step]
    lr_values = [summary.history["train"]["model_lr"][x] for x in acc_steps]
    assert len(lr_values) == len(perplexity_values)
    plt.plot(lr_values, perplexity_values)
    plt.plot(max_lr,
             best_acc,
             'o',
             color='r',
             label="Best Perplexity={}, LR={}".format(best_acc, max_lr))
    plt.xlim(0.0, 46)
    plt.ylim(50, 1000)
    plt.xlabel("Learning Rate")
    plt.ylabel("Evaluation Perplexity")
    plt.legend(frameon=False)
    plt.title("Language Modeling Learning Rate Search")
    # plt.show()
    plt.savefig("lr_search_lm.png", dpi=300)


if __name__ == "__main__":
    visualize_language_modeling()
