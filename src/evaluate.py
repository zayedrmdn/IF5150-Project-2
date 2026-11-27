import numpy as np


def evaluate(true_labels, predicted_labels, fit_seconds, inference_seconds, classes):
    number_of_classes = len(classes)
    matrix = np.zeros((number_of_classes, number_of_classes), dtype=int)
    for i in range(len(true_labels)):
        row = classes.index(true_labels[i])
        column = classes.index(predicted_labels[i])
        matrix[row, column] = matrix[row, column] + 1

    correct = 0
    for i in range(number_of_classes):
        correct = correct + matrix[i, i]
    accuracy = correct / len(true_labels)

    per_class = []
    f1_total = 0.0
    for i in range(number_of_classes):
        true_positive = matrix[i, i]
        predicted_total = matrix[:, i].sum()
        actual_total = matrix[i, :].sum()

        precision = 0.0
        if predicted_total > 0:
            precision = true_positive / predicted_total

        recall = 0.0
        if actual_total > 0:
            recall = true_positive / actual_total

        f1 = 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)

        scores = {}
        scores["precision"] = float(precision)
        scores["recall"] = float(recall)
        scores["f1"] = float(f1)
        per_class.append(scores)
        f1_total = f1_total + f1

    results = {}
    results["classes"] = classes
    results["accuracy"] = float(accuracy)
    results["confusion_matrix"] = matrix
    results["per_class"] = per_class
    results["macro_f1"] = float(f1_total / number_of_classes)
    results["fit_seconds"] = float(fit_seconds)
    results["inference_seconds"] = float(inference_seconds)
    return results


def print_report(results):
    print("accuracy:", round(results["accuracy"], 4))
    print("macro F1:", round(results["macro_f1"], 4))
    print("fit seconds:", round(results["fit_seconds"], 2))
    print("inference seconds:", round(results["inference_seconds"], 3))
    print("class, precision, recall, f1")
    for i in range(len(results["classes"])):
        scores = results["per_class"][i]
        print(results["classes"][i], round(scores["precision"], 4), round(scores["recall"], 4), round(scores["f1"], 4))
    print("confusion matrix, rows true, columns predicted")
    print(results["confusion_matrix"])
