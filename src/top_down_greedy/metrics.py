# metrics.py
from typing import Dict, List

import pandas as pd


def extract_numeric_values(x):
    """Extrai valores numéricos de strings no formato '[a - b]'"""
    if isinstance(x, str) and '[' in x and '-' in x:
        try:
            values = x.strip('[]').split('-')
            return [float(v.strip()) for v in values]
        except ValueError:
            return None
    return x

def calculate_ncp_numeric(values: pd.Series) -> float:
    """Calcula NCP para atributos numéricos"""
    if pd.isna(values).all() or len(values) == 0:
        return 0
    
    numeric_values = []
    
    for val in values:
        if isinstance(val, str) and '[' in val:
            nums = extract_numeric_values(val)
            if nums:
                numeric_values.extend(nums)
        elif pd.notnull(val):
            try:
                numeric_values.append(float(val))
            except (ValueError, TypeError):
                continue
    
    if not numeric_values:
        return 0
    
    max_val = max(numeric_values)
    min_val = min(numeric_values)
    range_total = max(numeric_values) - min(numeric_values)
    
    return (max_val - min_val) / range_total if range_total != 0 else 0

def calculate_ncp_categorical(values: pd.Series) -> float:
    """Calcula NCP para atributos categóricos"""
    if pd.isna(values).all() or len(values) == 0:
        return 0
    
    values = values.dropna()
    
    if len(values) == 0:
        return 0
    
    unique_values = len(set(values))
    total_categories = len(set(values.explode() if isinstance(values.iloc[0], list) else values))
    
    return (unique_values - 1) / (total_categories - 1) if total_categories > 1 else 0

def calculate_ncp(df: pd.DataFrame, original_df: pd.DataFrame, qi_columns: List[str]) -> float:
    """
    Calcula o NCP (Normalized Certainty Penalty) total
    """
    total_ncp = 0
    n_attributes = len(qi_columns)
    
    for col in qi_columns:
        was_numeric = pd.api.types.is_numeric_dtype(original_df[col])
        if was_numeric:
            ncp = calculate_ncp_numeric(df[col])
        else:
            ncp = calculate_ncp_categorical(df[col])
        total_ncp += ncp
    
    return total_ncp / n_attributes if n_attributes > 0 else 0

def calculate_gcp(df: pd.DataFrame, original_df: pd.DataFrame, qi_columns: List[str]) -> float:
    """
    Calcula o Global Certainty Penalty (GCP)
    """
    ncp = calculate_ncp(df, original_df, qi_columns)
    return ncp * 100

def calculate_gen_il(df: pd.DataFrame, original_df: pd.DataFrame, qi_columns: List[str]) -> float:
    """
    Calcula o GenTotal_IL (Information Loss)
    """
    total_il = 0
    n_records = len(df)
    
    for col in qi_columns:
        was_numeric = pd.api.types.is_numeric_dtype(original_df[col])
        if was_numeric:
            il = calculate_ncp_numeric(df[col])
        else:
            il = calculate_ncp_categorical(df[col])
        total_il += il * n_records
    
    return (total_il / (n_records * len(qi_columns))) * 100

def get_equivalence_classes(df: pd.DataFrame, qi_columns: List[str]) -> List[pd.DataFrame]:
    """
    Identifica as classes de equivalência no dataset anonimizado
    """
    return [group for _, group in df.groupby(qi_columns)]

def calculate_cavg(df: pd.DataFrame, qi_columns: List[str], k: int) -> float:
    """
    Calcula o CAVG (Average Equivalence Class Size Metric)
    """
    eq_classes = get_equivalence_classes(df, qi_columns)
    total_records = len(df)
    return total_records / (len(eq_classes) * k)

def calculate_metrics(anonymized_df: pd.DataFrame, 
                     original_df: pd.DataFrame,
                     qi_columns: List[str],
                     k: int) -> Dict[str, float]:
    """
    Calcula todas as métricas de avaliação incluindo NCP
    """
    try:
        ncp_value = calculate_ncp(anonymized_df, original_df, qi_columns)
        metrics = {
            'k': k,
            'NCP': ncp_value,
            'GCP': calculate_gcp(anonymized_df, original_df, qi_columns),
            'GenIL': calculate_gen_il(anonymized_df, original_df, qi_columns),
            'CAVG': calculate_cavg(anonymized_df, qi_columns, k)
        }
    except Exception as e:
        print(f"Erro ao calcular métricas: {str(e)}")
        metrics = {
            'k': k,
            'NCP': None,
            'GCP': None,
            'GenIL': None,
            'CAVG': None
        }
    
    return metrics