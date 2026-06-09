import pandas as pd
import matplotlib.pyplot as plt
import ast

from IPython.display import display

def visualize_experiments(log_file='experiments_log_v2.csv'):
    """
    Визуализирует все эксперименты из CSV файла.
    """
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"Файл '{log_file}' не найден!")
        return None
    
    if len(df) == 0:
        print("Файл пустой, нет экспериментов для визуализации.")
        return df
    
    # Парсим timestamp и параметры
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['exp_num'] = range(1, len(df) + 1)
    
    def parse_params(p):
        try:
            return ast.literal_eval(p) if pd.notna(p) else {}
        except:
            return {}
    
    df['params_dict'] = df['params'].fillna('').apply(parse_params)
    df['depth'] = df['params_dict'].apply(lambda x: x.get('depth'))
    df['learning_rate'] = df['params_dict'].apply(lambda x: x.get('learning_rate'))
    df['iterations'] = df['params_dict'].apply(lambda x: x.get('iterations'))
    df['l2_leaf_reg'] = df['params_dict'].apply(lambda x: x.get('l2_leaf_reg'))
    
    print(f"Загружено экспериментов: {len(df)}")
    if 'val_bal_acc' in df.columns:
        best_idx = df['val_bal_acc'].idxmax()
        print(f"Лучший val_bal_acc: {df.loc[best_idx, 'val_bal_acc']:.4f} (Эксперимент #{best_idx + 1})")
    print("=" * 70)
    
    # ГРАФИК: ПРОГРЕСС ВО ВРЕМЕНИ
    if 'train_bal_acc' in df.columns and 'val_bal_acc' in df.columns:
        fig, ax = plt.subplots()
        
        ax.plot(df['exp_num'], df['train_bal_acc'], 'o-', label='Train BalAcc', color='skyblue', linewidth=2.5, markersize=8)
        ax.plot(df['exp_num'], df['val_bal_acc'], 's-', label='Val BalAcc', color='coral', linewidth=2.5, markersize=8)
        
        # Отмечаем лучший результат
        best = df.loc[best_idx]
        ax.scatter(best['exp_num'], best['val_bal_acc'], s=300, color='red', marker='*', zorder=5, label=f'Best (#{int(best["exp_num"])})')
        ax.annotate(f"Val: {best['val_bal_acc']:.4f}", 
                    xy=(best['exp_num'], best['val_bal_acc']),
                    xytext=(20, 10), textcoords='offset points',
                    fontsize=11, color='red', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red'))
        
        ax.set_xlabel('Номер эксперимента', fontsize=13)
        ax.set_ylabel('Balanced Accuracy', fontsize=13)
        ax.set_title('Прогресс экспериментов во времени', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(df['exp_num'])
        plt.tight_layout()
        plt.show()
    
    # ГРАФИК: АНАЛИЗ ПЕРЕОБУЧЕНИЯ (GAP)
    if 'gap' in df.columns:
        fig, ax = plt.subplots()
        
        colors = ['green' if g < 0.01 else 'orange' if g < 0.03 else 'red' for g in df['gap']]
        bars = ax.bar(df['exp_num'], df['gap'], color=colors, alpha=0.7, edgecolor='black')
        
        ax.axhline(y=0.01, color='green', linestyle='--', alpha=0.7, label='Отлично (< 1%)')
        ax.axhline(y=0.03, color='orange', linestyle='--', alpha=0.7, label='Нормально (< 3%)')
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Переобучение (> 5%)')
        
        for bar, gap in zip(bars, df['gap']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(df['gap']) * 0.02,
                    f"{gap:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Номер эксперимента', fontsize=13)
        ax.set_ylabel('Gap (Train - Val)', fontsize=13)
        ax.set_title('Анализ переобучения (Gap)', fontsize=15, fontweight='bold')
        ax.set_xticks(df['exp_num'])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
    
    # СВОДНАЯ ТАБЛИЦА
    cols_to_show = ['exp_num', 'model', 'depth', 'learning_rate', 'iterations', 'train_bal_acc', 'val_bal_acc', 'gap', 'features']
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    
    summary = df[cols_to_show].copy()
    summary.columns = ['#', 'Model', 'Depth', 'LR', 'Iters', 'Train', 'Val', 'Gap', 'Features']
    
    print("\nСводная таблица экспериментов:")
    
    if 'val_bal_acc' in df.columns:
        # Форматируем числовые колонки для красивого вывода
        display_summary = summary.copy()
        for col in ['Train', 'Val', 'Gap']:
            if col in display_summary.columns:
                display_summary[col] = display_summary[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else ""
                )
        
        # Добавляем маркер лучшего эксперимента
        best_exp_num = df.loc[best_idx, 'exp_num']
        display_summary['Best'] = display_summary['#'].apply(
            lambda x: "⭐" if x == best_exp_num else ""
        )
        
        # Переупорядочиваем колонки
        cols_order = ['Best', '#', 'Model', 'Depth', 'LR', 'Iters', 'Train', 'Val', 'Gap', 'Features']
        display_summary = display_summary[[c for c in cols_order if c in display_summary.columns]]
        
        # Пытаемся использовать красивый стиль, если jinja2 доступен
        try:
            def highlight_row(row):
                if row['#'] == best_exp_num:
                    return ['background-color: gold; font-weight: bold'] * len(row)
                return [''] * len(row)
            
            styled = display_summary.style.apply(highlight_row, axis=1)
            display(styled)
        except (ImportError, AttributeError):
            # Если jinja2 не установлен, показываем обычную таблицу
            print("(Установите 'pip install jinja2' для красивого форматирования)")
            display(display_summary)
    
    return df