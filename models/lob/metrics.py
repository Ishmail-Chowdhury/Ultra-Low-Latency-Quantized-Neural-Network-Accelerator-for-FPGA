"""Fixed-class metrics for imbalanced three-way direction classification."""

from models.lob.data import CLASSES


def classification_metrics(labels, predictions) -> dict:
    if not labels or len(labels) != len(predictions):
        raise ValueError("metrics require equal nonempty label and prediction sequences")
    matrix = [[0] * 3 for _ in range(3)]
    for actual, predicted in zip(labels, predictions):
        if actual not in range(3) or predicted not in range(3):
            raise ValueError("class IDs must be 0, 1, or 2")
        matrix[actual][predicted] += 1
    f1, recalls = [], []
    for label in range(3):
        tp = matrix[label][label]
        actual = sum(matrix[label])
        predicted = sum(row[label] for row in matrix)
        f1.append(2 * tp / (actual + predicted) if actual + predicted else 0.0)
        recalls.append(tp / actual if actual else 0.0)
    return {"accuracy": sum(matrix[i][i] for i in range(3)) / len(labels),
            "macro_f1": sum(f1) / 3, "confusion_matrix": matrix,
            "class_order": CLASSES, "recall": dict(zip(CLASSES, recalls)),
            "support": dict(zip(CLASSES, map(sum, matrix)))}
