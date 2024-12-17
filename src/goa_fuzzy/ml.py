import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def process_interval_column(x):
    """Process columns containing intervals like [a - b]"""
    if pd.isna(x) or x == ' ?' or '?' in str(x):
        return np.nan
    if isinstance(x, str) and '[' in x and '-' in x:
        values = x.strip('[]').split('-')
        return float(values[0].strip())
    return x

def load_datasets(data_dir):
    datasets = {}
    label_encoders = {}
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            path = os.path.join(data_dir, filename)
            print(f"\nProcessing file: {filename}")
            df = pd.read_csv(path)
            
            # Separate features and target
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            # Process categorical columns
            for col in X.columns:
                if X[col].dtype == 'object':
                    if col not in label_encoders:
                        label_encoders[col] = LabelEncoder()
                    
                    X[col] = X[col].replace(' ?', X[col].mode()[0])
                    X[col] = label_encoders[col].fit_transform(X[col])
            
            # Normalize features
            scaler = MinMaxScaler()
            X_normalized = scaler.fit_transform(X)
            
            # Reconstruct dataframe
            df_normalized = pd.DataFrame(X_normalized, columns=X.columns)
            df_normalized['target'] = y
            
            k = int(filename.split('_')[2].split('.')[0])
            datasets[filename] = {'data': df_normalized, 'k': k}
            print(f"Shape: {df_normalized.shape}")
            print(f"Types: {df_normalized.dtypes}")
    
    return datasets

def create_pipelines():
    base_pipeline = [('scaler', MinMaxScaler())]
    
    models = {
        'MLP': MLPClassifier(
            hidden_layer_sizes=(100, 50),  
            activation='relu',             
            solver='adam',                 
            alpha=0.0001,                 
            learning_rate='adaptive',     
            learning_rate_init=0.001,      
            max_iter=500,                  
            early_stopping=True,           
            validation_fraction=0.1,
            verbose= True
        ),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=100,verbose=True),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Gaussian NB': GaussianNB(var_smoothing=1e-02),  
        'XGBoost': XGBClassifier(verbose=True),
    }
    
    return {name: Pipeline(base_pipeline + [('classifier', model)])
            for name, model in models.items()}

def evaluate_models(datasets, pipelines, cv=2):
    results_list = []
    scoring = ['accuracy', 'f1_weighted', 'recall_weighted', 'precision_weighted']
    
    os.makedirs('results/goa_fuzzy', exist_ok=True)
    
    for dataset_idx, (dataset_name, dataset_info) in enumerate(datasets.items(), 1):
        data = dataset_info['data']
        k = dataset_info['k']
        print(f"\n[{dataset_idx}/{len(datasets)}] Processing dataset k={k}")
        
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        model_results = {}
        
        for model_idx, (model_name, pipeline) in enumerate(pipelines.items(), 1):
            print(f"\n[Dataset {dataset_idx}/{len(datasets)}] [{model_idx}/{len(pipelines)}] Training {model_name}")
            try:
                scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
                result = {
                    'accuracy': scores['test_accuracy'].mean(),
                    'f1_score': scores['test_f1_weighted'].mean(),
                    'recall': scores['test_recall_weighted'].mean(),
                    'precision': scores['test_precision_weighted'].mean(),
                    'k': k,
                    'modelo_ML': model_name
                }
                results_list.append(result)
                model_results[model_name] = result
                
                # Save individual model results
                model_file = f'results/goa_fuzzy/{model_name.replace(" ", "_")}_results.csv'
                df = pd.DataFrame([result])
                if not os.path.exists(model_file):
                    df.to_csv(model_file, index=False)
                else:
                    df.to_csv(model_file, mode='a', header=False, index=False)
                
                print(f"Completed {model_name} - Accuracy: {scores['test_accuracy'].mean():.3f}")
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                
    return pd.DataFrame(results_list)

def main():
    data_dir = 'datasets/goa_fuzzy'
    print("Starting pipeline execution...")
    datasets = load_datasets(data_dir)
    pipelines = create_pipelines()
    results_df = evaluate_models(datasets, pipelines)
    
    # Save consolidated results
    results_df.to_csv('results/goa_fuzzy/all_results.csv', index=False)
    print("\nResults saved in results/goa_fuzzy/")

if __name__ == "__main__":
    main()