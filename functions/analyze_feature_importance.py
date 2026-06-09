import pandas as pd
import matplotlib.pyplot as plt

def analyze_feature_importance(model, X, top_n=75, importance_type='split'):
    """
    Анализирует важность признаков модели LightGBM.
    
    Args:
        model: обученная модель LightGBM
        X: данные (DataFrame) для получения названий признаков
        top_n: количество топ-признаков для отображения
        plot: строить ли график
        figsize: размер графика
    
    Returns:
        str: отформатированная строка с важностью признаков для логирования
    """
    # Для LightGBM можно получить разные типы важности
    if importance_type == 'gain':
        booster = model.booster_
        importances = booster.feature_importance(importance_type='gain')
    else:
        importances = model.feature_importances_

    feature_names = X.columns.tolist()

    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    print(f"Топ-{top_n} важных признаков:")
    print(fi_df.head(top_n).to_string(index=False))
    
    if importance_type == 'gain':
        total = fi_df['Importance'].sum()
        top_features_for_log = {}
        for _, row in fi_df.head(top_n).iterrows():
            pct = row['Importance'] / total * 100
            top_features_for_log[row['Feature']] = f"{fmt(row['Importance'])} ({pct:.1f}%)"
    else:
        top_features_for_log = fi_df.head(top_n).set_index('Feature')['Importance'].to_dict()
    
    return fi_df, top_features_for_log

def plot_feature_importance(fi_df, top_n=75, figsize=(10, 15), importance_type='split'):
    """Строит график важности признаков."""
    plt.figure(figsize=figsize)
    plt.barh(fi_df['Feature'][:top_n][::-1], fi_df['Importance'][:top_n][::-1])
    plt.xlabel('Importance')
    plt.title(f'Feature Importance (LightGBM, {importance_type})')
    plt.tight_layout()
    plt.show()

def fmt(num):
    if num >= 1e9: return f"{num/1e9:.2f}B"
    elif num >= 1e6: return f"{num/1e6:.2f}M"
    elif num >= 1e3: return f"{num/1e3:.2f}K"
    return f"{num:.2f}"