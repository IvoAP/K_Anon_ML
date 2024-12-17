import os

import pandas as pd
from config.config_data import AdultsConfig
from config.constants import DatasetOptions

from goa_fuzzy import goa_fuzzy
from metrics import calculate_metrics

dataset_config = {
    DatasetOptions.ADULTS: AdultsConfig,
    # DatasetOptions.ATTACK: AdultsConfig,  # Temporário
    # DatasetOptions.DATA1: AdultsConfig,   # Temporário
    # DatasetOptions.DATA2: AdultsConfig    # Temporário
}
CONST_K = 3


def select_dataset():
    print("Datasets disponíveis:")
    for option in DatasetOptions:
        print(f"- {option.value}")
    
    choice = input("\nEscolha o dataset: ").lower()
    selected = DatasetOptions(choice)
    config = dataset_config[selected]
    df = pd.read_csv(config.path)

    print("\nDataset:", selected.value)
    print("QIs:", config.qi)
    print("Target:", config.target)
    print("\nPrimeiras linhas:")
    print(df.head())
    
    return df, config.qi, config.target

def main():
    df, qis, target = select_dataset()
    
    # Criar estrutura de pastas
    output_dir = f'datasets/goa_fuzzy/'
    metrics_dir = f'metrics/goa_fuzzy/'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    all_metrics = []
    
    mode = input("Escolha o modo (1 - Uma vez, 2 - Loop k=2 até 50): ")
   
    if mode == "1":
        k = int(input("Digite o valor de k: "))
        result = goa_fuzzy(df, qis, k=k)
        result.to_csv(f'{output_dir}anon_k_{k}.csv', index=False)
        metrics = calculate_metrics(result,df,qis,k)
        all_metrics.append(metrics)
    else:
        for k in range(2, 51):
            result = goa_fuzzy(df, qis, k=k)
            result.to_csv(f'{output_dir}anon_k_{k}.csv', index=False)
            metrics = calculate_metrics(result,df,qis,k)
            all_metrics.append(metrics)
            print(f"Completado k={k}")
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(f'{metrics_dir}goa_fuzzy_metrics.csv', index=False)
    print("\nMétricas salvas em:", f'{metrics_dir}goa_fuzzy_metrics.csv')



if __name__ == "__main__":
    main()