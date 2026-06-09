import pandas as pd
import platform

from datetime import datetime
from pathlib import Path

def log_experiment(model_name, params, k_fold, feature_importance_split, feature_importance_gain,
                   features_info="", oof_confusion_matrix="", oof_per_class_recall="",
                   log_file='experiments_log_v2.csv'):
    """
    Сохраняет результаты эксперимента в CSV файл
    """
    log_file = Path(log_file)
    
    result = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Kaggle Score': '',
        'model': model_name,
        'K-Fold': k_fold,
        'OOF confusion matrix': oof_confusion_matrix,
        'OOF per-class recall': oof_per_class_recall,
        'feature_importance_split': feature_importance_split,
        'feature_importance_gain': feature_importance_gain,
        'params': str(params),
        'features': features_info,
        'description': '',
        'os_full': platform.platform() 
    }
    
    if not log_file.exists():
        log_file.touch()
        pd.DataFrame(columns=result.keys()).to_csv(log_file, index=False)

    df_log = pd.read_csv(log_file)
    df_log = pd.concat([df_log, pd.DataFrame([result])], ignore_index=True)
    
    df_log.to_csv(log_file, index=False)
    
    print(f"Результат сохранён в {log_file}")
    print(df_log.sort_values('timestamp', ascending=False).to_string(index=False))
    
    return df_log