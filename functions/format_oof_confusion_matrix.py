import pandas as pd
from sklearn.metrics import confusion_matrix

def format_oof_confusion_matrix(y_true, y_pred, labels=None):
    """Форматирует OOF confusion matrix для сохранения в лог."""
    labels = list(labels) if labels is not None else sorted(pd.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    header = 'actual \\ pred | ' + ' | '.join(str(l) for l in labels)
    lines = [header]
    for i, label in enumerate(labels):
        row_vals = ' | '.join(str(v) for v in cm[i])
        lines.append(f'{label} | {row_vals}')
    return '\n'.join(lines)