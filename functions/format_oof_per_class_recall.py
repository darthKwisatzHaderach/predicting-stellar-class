import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score


def format_oof_per_class_recall(y_true, y_pred, labels=None):
    """Форматирует OOF recall по классам для сохранения в лог."""
    labels = list(labels) if labels is not None else sorted(pd.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    recalls = recall_score(y_true, y_pred, labels=labels, average=None)
    support = cm.sum(axis=1)

    lines = []
    for label, rec, correct, sup in zip(labels, recalls, cm.diagonal(), support):
        lines.append(f'{label}: {rec:.4f} ({correct}/{sup})')

    lines.append(f'BalAcc: {balanced_accuracy_score(y_true, y_pred):.4f}')
    return '\n'.join(lines)
