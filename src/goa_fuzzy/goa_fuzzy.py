# goa_fuzzy.py
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class FuzzyGOAnonymization:
    def __init__(self, n_gorillas: int = 40, max_iter: int = 75, n_clusters: int = 5,
                 m: float = 2.0, error: float = 1e-5, maxiter: int = 300):
        self.n_gorillas = n_gorillas
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.m = m
        self.error = error
        self.maxiter = maxiter
        self.scaler = MinMaxScaler()
        
    def calculate_ncp(self, cluster_data: pd.DataFrame, original_data: pd.DataFrame, qi: str) -> float:
        """NCP com preservação de padrões para ML"""
        if pd.api.types.is_numeric_dtype(original_data[qi]):
            range_total = original_data[qi].max() - original_data[qi].min()
            if range_total == 0:
                return 0
            
            cluster_range = cluster_data[qi].max() - cluster_data[qi].min()
            std_ratio = np.std(cluster_data[qi]) / np.std(original_data[qi])
            return (cluster_range / range_total) * std_ratio
        else:
            total_distinct = len(original_data[qi].unique())
            if total_distinct <= 1:
                return 0
                
            cluster_distinct = len(cluster_data[qi].unique())
            freq_ratio = len(cluster_data) / len(original_data)
            return ((cluster_distinct - 1) / (total_distinct - 1)) * freq_ratio

    def calculate_cluster_ncp(self, cluster_data: pd.DataFrame, original_data: pd.DataFrame, 
                            qis: List[str]) -> float:
        if len(cluster_data) == 0:
            return 0
            
        cluster_ncp = 0
        for qi in qis:
            qi_ncp = self.calculate_ncp(cluster_data, original_data, qi)
            if pd.api.types.is_numeric_dtype(original_data[qi]):
                qi_ncp *= 1.2
            cluster_ncp += qi_ncp
        
        return cluster_ncp / len(qis)
    
    def _preprocess_data(self, data: pd.DataFrame, qis: List[str]) -> pd.DataFrame:
        processed_data = data.copy()
        
        for qi in qis:
            if not pd.api.types.is_numeric_dtype(processed_data[qi]):
                processed_data[qi] = pd.Categorical(processed_data[qi]).codes
            
            processed_data[qi] = self.scaler.fit_transform(
                processed_data[qi].values.reshape(-1, 1)
            )
            
        return processed_data
    
    def _calculate_memberships(self, data: np.ndarray, centers: np.ndarray) -> np.ndarray:
        data_t = data.T
        centers_t = centers.T
        
        distances = np.zeros((centers.shape[0], data.shape[0]))
        for i in range(centers.shape[0]):
            distances[i] = np.sqrt(np.sum((data - centers[i])**2, axis=1))
        
        tmp = distances ** (-2/(self.m-1))
        memberships = tmp / np.sum(tmp, axis=0)
        
        return memberships
    
    def _calculate_fitness(self, gorilla: np.ndarray, data: pd.DataFrame, 
                         original_data: pd.DataFrame, qis: List[str], k: int) -> float:
        centers = gorilla.reshape(self.n_clusters, len(qis))
        memberships = self._calculate_memberships(data[qis].values, centers)
        cluster_labels = np.argmax(memberships.T, axis=1)
        
        total_ncp = 0
        information_loss = 0
        k_violation_penalty = 0
        pattern_preservation = 0
        
        for i in range(self.n_clusters):
            cluster_mask = cluster_labels == i
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size < k:
                k_violation_penalty += (k - cluster_size) * 100
            
            if cluster_size > 0:
                cluster_data = original_data.loc[cluster_mask]
                cluster_weight = (cluster_size / len(data)) ** 0.5
                total_ncp += self.calculate_cluster_ncp(cluster_data, original_data, qis) * cluster_weight
                
                for qi in qis:
                    original_cluster = original_data.loc[cluster_mask, qi]
                    if pd.api.types.is_numeric_dtype(original_data[qi]):
                        cluster_std = np.std(original_cluster)
                        total_std = np.std(original_data[qi])
                        if total_std > 0:
                            pattern_preservation += abs(1 - (cluster_std / total_std))
                            
                        total_range = original_data[qi].max() - original_data[qi].min()
                        if total_range > 0:
                            cluster_range = original_cluster.max() - original_cluster.min()
                            information_loss += cluster_range / total_range
                    else:
                        total_unique = len(original_data[qi].unique())
                        cluster_unique = len(original_cluster.unique())
                        if total_unique > 1:
                            information_loss += cluster_unique / total_unique
        
        w_ncp = 0.3
        w_info_loss = 0.2
        w_k_violation = 0.3
        w_pattern = 0.2
        
        combined_fitness = (w_ncp * total_ncp + 
                          w_info_loss * information_loss + 
                          w_k_violation * k_violation_penalty +
                          w_pattern * pattern_preservation)
        
        return combined_fitness
    
    def _initialize_population(self, n_features: int) -> np.ndarray:
        return np.random.rand(self.n_gorillas, self.n_clusters * n_features)
    
    def _update_position(self, gorilla: np.ndarray, best_gorilla: np.ndarray, 
                        exploration_rate: float) -> np.ndarray:
        social_force = best_gorilla - gorilla
        
        distance_to_best = np.linalg.norm(best_gorilla - gorilla)
        exploration_magnitude = 2 * np.exp(-distance_to_best)
        exploration_force = (np.random.rand(*gorilla.shape) - 0.5) * exploration_magnitude
        
        new_position = (gorilla + 
                       social_force * exploration_rate +
                       exploration_force * (1 - exploration_rate))
        
        return np.clip(new_position, 0, 1)
    
    def _generalize_values(self, data: pd.DataFrame, cluster_mask: np.ndarray, 
                          qi: str) -> str:
        cluster_data = data.loc[cluster_mask, qi]
        
        if pd.api.types.is_numeric_dtype(data[qi]):
            min_val = cluster_data.min()
            max_val = cluster_data.max()
            return f"[{min_val:.2f}-{max_val:.2f}]"
        else:
            values = sorted(cluster_data.unique())
            return f"[{values[0]}-{values[-1]}]"
    
    def anonymize(self, data: pd.DataFrame, qis: List[str], k: int) -> pd.DataFrame:
        processed_data = self._preprocess_data(data[qis], qis)
        population = self._initialize_population(len(qis))
        
        best_fitness = float('inf')
        best_gorilla = None
        
        for iteration in range(self.max_iter):
            exploration_rate = 1 - (iteration / self.max_iter)
            
            for i in range(self.n_gorillas):
                fitness = self._calculate_fitness(
                    population[i], processed_data, data, qis, k
                )
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_gorilla = population[i].copy()
            
            for i in range(self.n_gorillas):
                population[i] = self._update_position(
                    population[i], best_gorilla, exploration_rate
                )
        
        centers = best_gorilla.reshape(self.n_clusters, len(qis))
        memberships = self._calculate_memberships(processed_data.values, centers)
        cluster_labels = np.argmax(memberships.T, axis=1)
        
        result = data.copy()
        for i in range(self.n_clusters):
            cluster_mask = cluster_labels == i
            if np.sum(cluster_mask) >= k:
                for qi in qis:
                    generalized_value = self._generalize_values(data, cluster_mask, qi)
                    result.loc[cluster_mask, qi] = generalized_value
        
        return result

def goa_fuzzy(df: pd.DataFrame, qis: List[str], k: int) -> pd.DataFrame:
    print(f"\nIniciando anonimização GOA-Fuzzy com k={k}")
    
    anonymizer = FuzzyGOAnonymization(
        n_gorillas=40,
        max_iter=75,
        n_clusters=max(5, k),
        m=2.0
    )
    
    result = anonymizer.anonymize(df, qis, k)
    
    print(f"Anonimização concluída para k={k}")
    return result