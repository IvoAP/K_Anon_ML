import random
from typing import List, Tuple

import numpy as np
import pandas as pd


def get_ncp_score(values: pd.Series) -> float:
    """Calcula o NCP (Normalized Certainty Penalty) para uma série"""
    if pd.api.types.is_numeric_dtype(values):
        range_total = values.max() - values.min()
        return range_total / range_total if range_total != 0 else 0
    else:
        n_unique = values.nunique()
        return (n_unique - 1) / n_unique if n_unique > 1 else 0

def get_pair_distance(group: pd.DataFrame, record1_idx: int, record2_idx: int, qis: List[str]) -> float:
    """Versão vetorizada do cálculo de distância"""
    return sum(get_ncp_score(group.loc[[record1_idx, record2_idx], qi]) for qi in qis)

def find_split_pair(group: pd.DataFrame, qis: List[str], rounds: int = 3) -> Tuple[int, int]:
    """Encontra o par de registros mais distante para divisão"""
    n_records = len(group)
    if n_records < 2:
        return 0, 0
    
    print(f"\nBuscando par em grupo de {n_records} registros")
    
    u = random.randrange(n_records)
    indices = group.index.values
    distances = np.zeros(n_records)
    
    for _ in range(rounds):
        for i, idx in enumerate(indices):
            if idx != indices[u]:
                distances[i] = get_pair_distance(group, indices[u], idx, qis)
        
        max_index = np.argmax(distances)
        u = max_index
        
        print(f"Round {_ + 1}: Registro mais distante: {indices[max_index]}")
    
    return indices[u], indices[max_index]

def summarize_group(group: pd.DataFrame, qis: List[str]) -> pd.DataFrame:
    """Generaliza os valores do grupo"""
    result = group.copy()
    
    for qi in qis:
        if result[qi].iloc[0] != result[qi].iloc[-1]:
            if pd.api.types.is_numeric_dtype(result[qi]):
                min_val = result[qi].min()
                max_val = result[qi].max()
            else:
                min_val = result[qi].iloc[0]
                max_val = result[qi].iloc[-1]
            result[qi] = f"[{min_val} - {max_val}]"
    
    return result

def distribute_records(group: pd.DataFrame, u_idx: int, v_idx: int, qis: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Distribui os registros entre dois grupos"""
    print(f"\nDistribuindo {len(group)} registros")
    
    record_u = group.loc[u_idx]
    record_v = group.loc[v_idx]
    
    u_records = [record_u]
    v_records = [record_v]
    
    other_records = group.drop([u_idx, v_idx])
    
    for idx, record in other_records.iterrows():
        dist_u = sum(get_ncp_score(pd.concat([pd.Series([record_u[qi]]), pd.Series([record[qi]])], axis=0)) for qi in qis)
        dist_v = sum(get_ncp_score(pd.concat([pd.Series([record_v[qi]]), pd.Series([record[qi]])], axis=0)) for qi in qis)
        
        if dist_u <= dist_v:
            u_records.append(record)
        else:
            v_records.append(record)
    
    return pd.DataFrame(u_records), pd.DataFrame(v_records)

def anonymize_group(group: pd.DataFrame, qis: List[str], k: int, depth: int = 0) -> pd.DataFrame:
    """Função recursiva principal"""
    print(f"{'  ' * depth}Processando grupo de {len(group)} registros")
    
    if len(group) < 2 * k:
        return summarize_group(group, qis)
    
    u_idx, v_idx = find_split_pair(group, qis)
    group_u, group_v = distribute_records(group, u_idx, v_idx, qis)
    
    if len(group_u) < k or len(group_v) < k:
        return summarize_group(group, qis)
    
    return pd.concat([
        anonymize_group(group_u, qis, k, depth + 1),
        anonymize_group(group_v, qis, k, depth + 1)
    ])

def top_down_greedy(partition: pd.DataFrame, qis: List[str], k: int) -> pd.DataFrame:
    """Função principal do Top Down Greedy Anonymization"""
    print(f"Iniciando anonimização: {len(partition)} registros, k={k}")
    
    partition = partition.reset_index(drop=True)
    
    for qi in qis:
        if partition[qi].dtype == 'object':
            try:
                partition[qi] = pd.to_numeric(partition[qi])
            except:
                pass
    
    result = anonymize_group(partition, qis, k)
    print(f"Anonimização concluída: {len(result)} registros")
    
    return result