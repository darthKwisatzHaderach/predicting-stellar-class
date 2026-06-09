def format_params(params, keys=None):
    """
    Форматирует словарь параметров в читаемый вид.
    
    Args:
        params: словарь параметров
        keys: список ключей для сохранения (если None — все)
    """
    if keys is not None:
        params = {k: v for k, v in params.items() if k in keys}
    
    return "\n".join([f"{k} = {v}" for k, v in params.items()])