import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

def load_processed_data(file_path):
    """
    Carga el dataset procesado
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset procesado cargado correctamente. Forma: {df.shape}")
        return df
    except Exception as e:
        print(f"Error al cargar el dataset procesado: {e}")
        return None

def prepare_features_target(df):
    """
    Prepara las características (features) y la variable objetivo (target)
    """
    # Variable objetivo
    target = 'PROMEDIO CONSUMO ACUEDUCTO_log'
    
    # CRÍTICO: Excluir TODAS las versiones de la variable objetivo
    columns_to_exclude = [
        target,
        'PROMEDIO CONSUMO ACUEDUCTO',
        'PROMEDIO CONSUMO ACUEDUCTO_log',  # Por seguridad
        'MES'  # Si existe como texto
    ]
    
    # Definir características excluyendo todas las versiones del objetivo
    features = [col for col in df.columns if col not in columns_to_exclude]
    
    # Verificaciones de seguridad
    print(f"Variable objetivo: {target}")
    print(f"Total de características: {len(features)}")
    print(f"Columnas excluidas: {columns_to_exclude}")
    
    # VERIFICACIÓN CRÍTICA
    for col in features:
        if 'PROMEDIO' in col.upper() or 'CONSUMO' in col.upper():
            print(f"⚠️ ADVERTENCIA: Columna sospechosa en características: {col}")
            raise ValueError(f"La columna {col} parece contener información del objetivo")
    
    # Verificar tipos de características
    estrato_cols = [col for col in features if col.startswith('Estrato_')]
    municipio_cols = [col for col in features if col.startswith('Municipio_')]
    mes_cols = [col for col in features if col == 'MES_NUM']
    
    print(f"Encontradas {len(estrato_cols)} columnas de estrato")
    print(f"Encontradas {len(municipio_cols)} columnas de municipio")
    print(f"Encontradas {len(mes_cols)} columnas de mes")
    
    # Separar características y objetivo
    X = df[features]
    y = df[target]
    
    return X, y, features

def split_data(X, y):
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba
    """
    # Primer split: separar conjunto de prueba
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Segundo split: separar entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2 del total
    )
    
    print(f"Conjunto de entrenamiento: {X_train.shape[0]} registros")
    print(f"Conjunto de validación: {X_val.shape[0]} registros")
    print(f"Conjunto de prueba: {X_test.shape[0]} registros")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_preprocessing_pipeline():
    """
    Crea un pipeline de preprocesamiento para los datos
    """
    # Solo estandarizamos la variable MES_NUM, las variables one-hot las dejamos como están
    # Como estamos seleccionando las variables específicamente, no necesitamos un ColumnTransformer
    # y podemos usar directamente el StandardScaler
    return StandardScaler()

def train_models(X_train, y_train, preprocessor):
    """
    Entrena varios modelos y compara sus resultados
    """
    # Seleccionar solo la columna MES_NUM para estandarización
    X_mes_num = X_train[['MES_NUM']] if 'MES_NUM' in X_train.columns else pd.DataFrame()
    
    # Obtener columnas one-hot
    X_dummies = X_train.drop('MES_NUM', axis=1) if 'MES_NUM' in X_train.columns else X_train
    
    # Definir los modelos a evaluar
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42)
    }
    
    # Evaluar cada modelo manualmente
    results = {}
    for name, model in models.items():
        print(f"Entrenando modelo: {name}")
        try:
            # Estandarizar solo MES_NUM si existe
            if not X_mes_num.empty:
                X_mes_scaled = preprocessor.fit_transform(X_mes_num)
                # Convertir a DataFrame para preservar el nombre de columna
                X_mes_scaled_df = pd.DataFrame(X_mes_scaled, index=X_mes_num.index, columns=['MES_NUM'])
                # Concatenar con las variables one-hot
                X_processed = pd.concat([X_mes_scaled_df, X_dummies], axis=1)
            else:
                # Si no hay MES_NUM, usar directamente las variables one-hot
                X_processed = X_dummies
            
            # Crear y entrenar modelo
            model.fit(X_processed, y_train)
            
            # Validación cruzada manual para evitar problemas con pipeline
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            
            for train_idx, val_idx in kf.split(X_processed):
                X_train_cv, X_val_cv = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
                y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model_cv = models[name].__class__(**models[name].get_params())
                model_cv.fit(X_train_cv, y_train_cv)
                y_pred_cv = model_cv.predict(X_val_cv)
                
                mse = mean_squared_error(y_val_cv, y_pred_cv)
                cv_scores.append(-mse)  # Negativo para que sea consistente con sklearn
            
            # Convertir a RMSE
            rmse_scores = np.sqrt(-np.array(cv_scores))
            results[name] = {
                'mean_rmse': rmse_scores.mean(),
                'std_rmse': rmse_scores.std(),
                'model': model,
                'X_processed': X_processed  # Guardar X procesado para futuros pasos
            }
            
            print(f"{name} - RMSE medio: {rmse_scores.mean():.4f}, Desv. Estándar: {rmse_scores.std():.4f}")
        except Exception as e:
            print(f"Error al entrenar {name}: {str(e)}")
    
    # Identificar el mejor modelo
    if results:
        best_model = min(results.items(), key=lambda x: x[1]['mean_rmse'])
        print(f"Mejor modelo: {best_model[0]} con RMSE: {best_model[1]['mean_rmse']:.4f}")
        return results, best_model[0]
    else:
        raise ValueError("No se pudo entrenar ningún modelo correctamente")

def tune_hyperparameters(model_name, X_train, y_train, preprocessor, results):
    """
    Realiza ajuste de hiperparámetros para un modelo específico
    """
    # Usar X procesado
    X_processed = results[model_name]['X_processed']
    
    # Definir parámetros de búsqueda según el tipo de modelo
    if model_name == 'RandomForest':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_name == 'GradientBoosting':
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
    else:  # LinearRegression no tiene hiperparámetros para ajustar
        print("El modelo de regresión lineal no tiene hiperparámetros que ajustar.")
        return results[model_name]['model'], {}
    
    # Realizar búsqueda de hiperparámetros directamente sin pipeline
    print(f"Iniciando ajuste de hiperparámetros para {model_name}...")
    grid_search = GridSearchCV(
        model, param_grid, cv=5, 
        scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_processed, y_train)
    
    # Mostrar mejores parámetros
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def process_features(X, preprocessor, model_name, results):
    """
    Procesa las características de la misma manera que durante el entrenamiento
    """
    # Seleccionar solo la columna MES_NUM para estandarización
    X_mes_num = X[['MES_NUM']] if 'MES_NUM' in X.columns else pd.DataFrame()
    
    # Obtener columnas one-hot
    X_dummies = X.drop('MES_NUM', axis=1) if 'MES_NUM' in X.columns else X
    
    # Estandarizar solo MES_NUM si existe
    if not X_mes_num.empty:
        X_mes_scaled = preprocessor.transform(X_mes_num)
        # Convertir a DataFrame para preservar el nombre de columna
        X_mes_scaled_df = pd.DataFrame(X_mes_scaled, index=X_mes_num.index, columns=['MES_NUM'])
        # Concatenar con las variables one-hot
        X_processed = pd.concat([X_mes_scaled_df, X_dummies], axis=1)
    else:
        # Si no hay MES_NUM, usar directamente las variables one-hot
        X_processed = X_dummies
    
    return X_processed

def evaluate_model(model, X_val, y_val, X_test, y_test, preprocessor, model_name, results):
    """
    Evalúa el modelo en los conjuntos de validación y prueba
    """
    # Constante usada en la transformación logarítmica
    log_constant = 1
    
    # Procesar datos de validación
    X_val_processed = process_features(X_val, preprocessor, model_name, results)
    
    # Predecir en escala logarítmica
    y_val_pred_log = model.predict(X_val_processed)
    
    # Invertir transformación logarítmica para evaluación
    y_val_pred = np.exp(y_val_pred_log) - log_constant
    y_val_original = np.exp(y_val) - log_constant
    
    # Calcular métricas en escala original
    val_rmse = np.sqrt(mean_squared_error(y_val_original, y_val_pred))
    val_mae = mean_absolute_error(y_val_original, y_val_pred)
    val_r2 = r2_score(y_val_original, y_val_pred)
    
    print("Rendimiento en conjunto de validación:")
    print(f"RMSE: {val_rmse:.4f}")
    print(f"MAE: {val_mae:.4f}")
    print(f"R²: {val_r2:.4f}")
    
    # Procesar datos de prueba
    X_test_processed = process_features(X_test, preprocessor, model_name, results)
    
    # Predecir en escala logarítmica
    y_test_pred_log = model.predict(X_test_processed)
    
    # Invertir transformación logarítmica para evaluación
    y_test_pred = np.exp(y_test_pred_log) - log_constant
    y_test_original = np.exp(y_test) - log_constant
    
    # Calcular métricas en escala original
    test_rmse = np.sqrt(mean_squared_error(y_test_original, y_test_pred))
    test_mae = mean_absolute_error(y_test_original, y_test_pred)
    test_r2 = r2_score(y_test_original, y_test_pred)
    
    print("\nRendimiento en conjunto de prueba:")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"R²: {test_r2:.4f}")
    
    # Crear diccionario con métricas
    metrics = {
        'validation': {
            'rmse': val_rmse,
            'mae': val_mae,
            'r2': val_r2
        },
        'test': {
            'rmse': test_rmse,
            'mae': test_mae,
            'r2': test_r2
        }
    }
    
    # Retornar los valores en escala logarítmica (para mantener consistencia con el resto del código)
    # pero las métricas ya están calculadas en escala original
    return metrics, y_test, y_test_pred_log

def evaluate_all_models(model_results, X_val, y_val, X_test, y_test, X_train, y_train, preprocessor):
    """
    Evalúa todos los modelos entrenados en los conjuntos de validación y prueba
    """
    # Constante usada en la transformación logarítmica
    log_constant = 1
    
    all_models_metrics = {}
    all_models_predictions = {}
    
    for name, model_data in model_results.items():
        model = model_data['model']
        
        try:
            # Procesar datos de validación
            X_val_processed = process_features(X_val, preprocessor, name, model_results)
            
            # Predecir en escala logarítmica
            y_val_pred_log = model.predict(X_val_processed)
            
            # Invertir transformación logarítmica para evaluación
            y_val_pred = np.exp(y_val_pred_log) - log_constant
            y_val_original = np.exp(y_val) - log_constant
            
            # Calcular métricas en escala original
            val_rmse = np.sqrt(mean_squared_error(y_val_original, y_val_pred))
            val_mae = mean_absolute_error(y_val_original, y_val_pred)
            val_r2 = r2_score(y_val_original, y_val_pred)
            
            # Procesar datos de prueba
            X_test_processed = process_features(X_test, preprocessor, name, model_results)
            
            # Predecir en escala logarítmica
            y_test_pred_log = model.predict(X_test_processed)
            
            # Invertir transformación logarítmica para evaluación
            y_test_pred = np.exp(y_test_pred_log) - log_constant
            y_test_original = np.exp(y_test) - log_constant
            
            # Calcular métricas en escala original
            test_rmse = np.sqrt(mean_squared_error(y_test_original, y_test_pred))
            test_mae = mean_absolute_error(y_test_original, y_test_pred)
            test_r2 = r2_score(y_test_original, y_test_pred)
            
            # Guardar métricas (calculadas en escala original)
            all_models_metrics[name] = {
                'validation': {
                    'rmse': val_rmse,
                    'mae': val_mae,
                    'r2': val_r2
                },
                'test': {
                    'rmse': test_rmse,
                    'mae': test_mae,
                    'r2': test_r2
                }
            }
            
            # Guardar predicciones en escala logarítmica (para consistencia con el resto del código)
            all_models_predictions[name] = {
                'y_test': y_test,
                'y_pred': y_test_pred_log
            }
            
            print(f"\nMétricas para {name}:")
            print(f"Validación - RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
            print(f"Prueba - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
        except Exception as e:
            print(f"Error al evaluar {name}: {str(e)}")
    
    return all_models_metrics, all_models_predictions

def generate_model_visualizations(models_predictions, features, models, output_path, all_models_metrics):
    """
    Genera visualizaciones para evaluar todos los modelos
    """
    os.makedirs(output_path, exist_ok=True)
    # AÑADIR: Constante usada en la transformación
    log_constant = 1
    # Paleta de colores para distinguir los modelos
    model_colors = {
        'LinearRegression': 'blue',
        'RandomForest': 'green',
        'GradientBoosting': 'red'
    }

    # ———– SANITY CHECK: asegurar que y_test es el consumo promedio ———–
    print("\n=== VALORES ANTES DE INVERSIÓN LOGARÍTMICA ===")
    for model_name, pred in models_predictions.items():
        yt = pred['y_test']
        print(f"[ANTES] {model_name} y_test (log) min={yt.min():.4f}, max={yt.max():.4f}")
    
    # ———– INVERSIÓN DE LA TRANSFORMACIÓN LOGARÍTMICA ———–
    models_predictions_original = {}
    for model_name, pred in models_predictions.items():
        y_test_log = pred['y_test']
        y_pred_log = pred['y_pred']
        
        # Invertir transformación logarítmica
        y_test_original = np.exp(y_test_log) - log_constant
        y_pred_original = np.exp(y_pred_log) - log_constant
        
        models_predictions_original[model_name] = {
            'y_test': y_test_original,
            'y_pred': y_pred_original
        }
        
        print(f"[DESPUÉS] {model_name} y_test (original) min={y_test_original.min():.2f}, max={y_test_original.max():.2f}")

    # ———– 1. Comparación de predicciones vs valores reales ———–
    plt.figure(figsize=(12, 8))
    for model_name, pred in models_predictions_original.items():
        y_test = pred['y_test']
        y_pred = pred['y_pred']
        plt.scatter(
            y_test,
            y_pred,
            alpha=0.5,
            label=model_name,
            color=model_colors.get(model_name, 'gray')
        )

    # Definir límites del gráfico usando sólo y_test
    min_val = min(pred['y_test'].min() for pred in models_predictions_original.values())
    max_val = max(pred['y_test'].max() for pred in models_predictions_original.values())
    delta   = (max_val - min_val) * 0.10  # 10% de margen
    min_plot = max(0, min_val - delta)
    max_plot = max_val + delta

    # Línea de predicción perfecta
    plt.plot([min_plot, max_plot], [min_plot, max_plot], 'k--', linewidth=1.5)

    plt.xlim(min_plot, max_plot)
    plt.ylim(min_plot, max_plot)
    plt.title('Comparación de modelos: Valores reales vs. Predicciones de consumo promedio (m³)')
    plt.xlabel('Consumo promedio real (m³)')
    plt.ylabel('Consumo promedio predicho (m³)')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'comparacion_modelos_predicciones.png'))
    plt.close()

    # ———– 2. Histograma de residuos para cada modelo ———–
    plt.figure(figsize=(15, 10))
    for i, (model_name, pred) in enumerate(models_predictions_original.items()):
        plt.subplot(2, 2, i+1)
        residuos = pred['y_test'] - pred['y_pred']
        plt.hist(residuos, bins=30, alpha=0.7, color=model_colors.get(model_name, 'gray'))
        plt.axvline(0, color='red', linestyle='--')
        plt.title(f'Distribución de residuos - {model_name}')
        plt.xlabel('Residuo (real - predicción) en m³')
        plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'distribucion_residuos_comparativo.png'))
    plt.close()

    # ———– 3. Importancia de características ———–
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            try:
                importances    = model.feature_importances_
                feature_names  = features
                fi_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False).head(20)

                plt.figure(figsize=(12, 10))
                sns.barplot(x='importance', y='feature', data=fi_df)
                plt.title(f'Importancia de características - {model_name}')
                plt.tight_layout()
                plt.savefig(os.path.join(
                    output_path,
                    f'importancia_caracteristicas_{model_name}.png'
                ))
                plt.close()
            except Exception as e:
                print(f"Error al graficar importancias para {model_name}: {e}")

    # ———– 4. Comparación de métricas entre modelos ———–
    # NOTA: Las métricas ya deberían estar en escala original si se invirtió la transformación en la evaluación
    model_names = list(models_predictions.keys())
    rmse_test   = [all_models_metrics[m]['test']['rmse'] for m in model_names]
    mae_test    = [all_models_metrics[m]['test']['mae']  for m in model_names]
    r2_test     = [all_models_metrics[m]['test']['r2']   for m in model_names]

    x     = np.arange(len(model_names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(x - width/2, rmse_test, width, label='RMSE', color='darkblue')
    ax1.bar(x + width/2, mae_test,  width, label='MAE',  color='lightblue')
    ax1.set_xlabel('Modelo')
    ax1.set_ylabel('Error (m³)')
    ax1.set_title('Comparación de métricas de error entre modelos')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(x, r2_test, 'ro-', linewidth=2, label='R²')
    ax2.set_ylabel('R²')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'comparacion_metricas.png'))
    plt.close()

    # ———– 5. Dispersión del mejor modelo ———–
    best_model_name = max(
        all_models_metrics.items(),
        key=lambda x: x[1]['test']['r2']
    )[0]
    best_pred = models_predictions_original[best_model_name]
    y_test_b  = best_pred['y_test']
    y_pred_b  = best_pred['y_pred']

    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_b, y_pred_b, alpha=0.6, edgecolor='k', color='blue')

    # misma línea perfecta y mismos límites de eje
    plt.plot([min_plot, max_plot], [min_plot, max_plot], 'r--', linewidth=1.5)
    plt.xlim(min_plot, max_plot)
    plt.ylim(min_plot, max_plot)

    plt.title(f'Predicciones del mejor modelo ({best_model_name}): Consumo promedio de agua')
    plt.xlabel('Consumo promedio real (m³)')
    plt.ylabel('Consumo promedio predicho (m³)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'mejor_modelo_predicciones.png'))
    plt.close()

    print(f"Visualizaciones comparativas de modelos generadas en {output_path}")

def save_model_data(best_model, all_model_metrics, best_params, output_dir, best_model_name, preprocessor):
    """
    Guarda el mejor modelo y la información de todos los modelos
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar el mejor modelo
    model_data = {
        'model': best_model,
        'preprocessor': preprocessor
    }
    joblib.dump(model_data, os.path.join(output_dir, 'best_model.pkl'))
    
    # Preparar información de modelos para guardar
    model_info = {
        'name': best_model_name,
        'params': best_params,
        'metrics': all_model_metrics[best_model_name],
        'all_models': all_model_metrics
    }
    
    # Guardar información de modelos
    joblib.dump(model_info, os.path.join(output_dir, 'model_info.pkl'))
    
    print(f"Modelo e información guardados en {output_dir}")

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Elimina o suaviza outliers en una columna específica
    """
    df_clean = df.copy()
    
    if method == 'iqr':
        # Método del rango intercuartílico
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
    elif method == 'std':
        # Método de la desviación estándar
        mean = df[column].mean()
        std = df[column].std()
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
    
    # Identificar outliers
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    print(f"Identificados {outliers.sum()} outliers de {len(df)} registros ({outliers.sum()/len(df)*100:.2f}%)")
    
    # Opción: Suavizar outliers (winsorización)
    df_clean.loc[df_clean[column] > upper_bound, column] = upper_bound
    df_clean.loc[df_clean[column] < lower_bound, column] = lower_bound
    
    return df_clean

def apply_log_transform(df, column, add_constant=1):
    """
    Aplica transformación logarítmica a una columna
    """
    df_transformed = df.copy()
    
    # Crear nueva columna con transformación log
    df_transformed[f'{column}_log'] = np.log(df_transformed[column] + add_constant)
    
    print(f"Transformación logarítmica aplicada a {column}")
    print(f"Asimetría original: {df[column].skew():.4f}")
    print(f"Asimetría después de transformación: {df_transformed[f'{column}_log'].skew():.4f}")
    
    return df_transformed

def main():
    """
    Función principal que ejecuta todo el proceso de ingeniería del modelo
    """
    try:
        
        # Definir rutas
        data_file = 'static/data/data_processed.csv'
        model_output_dir = 'static/data/models'
        viz_path = 'static/img/model_viz'
        
        # Asegurar que los directorios existan
        os.makedirs(model_output_dir, exist_ok=True)
        os.makedirs(viz_path, exist_ok=True)
        
        # Cargar datos procesados
        df = load_processed_data(data_file)
        if df is None:
            return False

        # Remover o suavizar outliers en la variable objetivo
        df = remove_outliers(df, 'PROMEDIO CONSUMO ACUEDUCTO', method='iqr', threshold=1.5)
        

        # Aplicar transformación logarítmica al consumo
        df = apply_log_transform(df, 'PROMEDIO CONSUMO ACUEDUCTO', add_constant=1)
        
        # IMPORTANTE: Guardar los parámetros de transformación para invertir después
        log_constant = 1
        
        # Preparar características y variable objetivo
        # MODIFICAR: Usar la columna transformada como objetivo
        X, y, features = prepare_features_target(df)
        
        # Verificación adicional de seguridad
        print("\n=== VERIFICACIÓN DE INTEGRIDAD DE DATOS ===")
        print(f"Columnas en DataFrame original: {df.columns.tolist()}")
        print(f"Características seleccionadas: {features[:10]}...")  # Primeras 10
        print(f"Variable objetivo: {y.name}")

        # Asegurar que no hay fuga de datos
        if 'PROMEDIO CONSUMO ACUEDUCTO' in features:
            raise ValueError("¡FUGA DE DATOS DETECTADA! La variable objetivo está en las características")
        # Dividir datos
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        # Crear preprocesador
        preprocessor = create_preprocessing_pipeline()
        
        # Entrenar y comparar modelos
        model_results, best_model_name = train_models(X_train, y_train, preprocessor)
        
        # Ajustar hiperparámetros del mejor modelo
        best_model, best_params = tune_hyperparameters(best_model_name, X_train, y_train, preprocessor, model_results)
        
        # Evaluar todos los modelos
        all_models_metrics, all_models_predictions = evaluate_all_models(
            model_results, X_val, y_val, X_test, y_test, X_train, y_train, preprocessor
        )
        
        # Extraer los modelos entrenados
        trained_models = {name: model_data['model'] for name, model_data in model_results.items()}
        
        # Generar visualizaciones comparativas
        generate_model_visualizations(all_models_predictions, features, trained_models, viz_path, all_models_metrics)
        
        # Guardar modelo e información
        save_model_data(best_model, all_models_metrics, best_params, model_output_dir, best_model_name, preprocessor)
        
        print("Proceso de ingeniería del modelo completado con éxito")
        return True
    except Exception as e:
        import traceback
        print(f"Error en el proceso de ingeniería del modelo: {str(e)}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    main()