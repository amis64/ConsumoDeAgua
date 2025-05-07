import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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
    # Definimos las características a usar
    features = ['AÑO', 'MES_NUM', 'MUNICIPIO', 'ESTRATO', 'No. SUSCRIPTORES ACUEDUCTO']
    
    # Variable objetivo: consumo promedio de agua por suscriptor
    target = 'PROMEDIO CONSUMO ACUEDUCTO'
    
    # Separamos características y objetivo
    X = df[features]
    y = df[target]
    
    print(f"Features utilizados: {features}")
    print(f"Variable objetivo: {target}")
    
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

def create_preprocessing_pipeline(features):
    """
    Crea un pipeline de preprocesamiento para los datos
    """
    # Identificar tipos de columnas
    categorical_features = ['MUNICIPIO']
    numerical_features = [col for col in features if col not in categorical_features]
    
    # Transformadores para cada tipo de columna
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combinar transformadores en un ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def train_models(X_train, y_train, preprocessor):
    """
    Entrena varios modelos y compara sus resultados
    """
    # Definir los modelos a evaluar
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42)
    }
    
    # Evaluar cada modelo con validación cruzada
    results = {}
    for name, model in models.items():
        # Crear pipeline con preprocesamiento y modelo
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Evaluar con validación cruzada
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, 
            cv=5, scoring='neg_mean_squared_error'
        )
        
        # Convertir a RMSE y almacenar resultados
        rmse_scores = np.sqrt(-cv_scores)
        results[name] = {
            'mean_rmse': rmse_scores.mean(),
            'std_rmse': rmse_scores.std(),
            'pipeline': pipeline
        }
        
        print(f"{name} - RMSE medio: {rmse_scores.mean():.4f}, Desv. Estándar: {rmse_scores.std():.4f}")
    
    # Identificar el mejor modelo
    best_model = min(results.items(), key=lambda x: x[1]['mean_rmse'])
    print(f"Mejor modelo: {best_model[0]} con RMSE: {best_model[1]['mean_rmse']:.4f}")
    
    return results, best_model[0]

def tune_hyperparameters(model_name, X_train, y_train, preprocessor, results):
    """
    Realiza ajuste de hiperparámetros para un modelo específico
    """
    # Definir parámetros de búsqueda según el tipo de modelo
    if model_name == 'RandomForest':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
    elif model_name == 'GradientBoosting':
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
            'model__min_samples_split': [2, 5, 10]
        }
    else:  # LinearRegression no tiene hiperparámetros para ajustar
        print("El modelo de regresión lineal no tiene hiperparámetros que ajustar.")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])
        pipeline.fit(X_train, y_train)
        return pipeline, {}
    
    # Crear pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Realizar búsqueda de hiperparámetros
    print(f"Iniciando ajuste de hiperparámetros para {model_name}...")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, 
        scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Mostrar mejores parámetros
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X_val, y_val, X_test, y_test):
    """
    Evalúa el modelo en los conjuntos de validación y prueba
    """
    # Evaluar en conjunto de validación
    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print("Rendimiento en conjunto de validación:")
    print(f"RMSE: {val_rmse:.4f}")
    print(f"MAE: {val_mae:.4f}")
    print(f"R²: {val_r2:.4f}")
    
    # Evaluar en conjunto de prueba
    y_test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
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
    
    return metrics, y_test, y_test_pred

def evaluate_all_models(model_results, X_val, y_val, X_test, y_test, X_train, y_train):
    """
    Evalúa todos los modelos entrenados en los conjuntos de validación y prueba
    """
    all_models_metrics = {}
    all_models_predictions = {}
    
    for name, model_data in model_results.items():
        pipeline = model_data['pipeline']
        
        # Entrenar el modelo con los datos de entrenamiento
        pipeline.fit(X_train, y_train)
        
        # Evaluar en conjunto de validación
        y_val_pred = pipeline.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        # Evaluar en conjunto de prueba
        y_test_pred = pipeline.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Guardar métricas
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
        
        # Guardar predicciones
        all_models_predictions[name] = {
            'y_test': y_test,
            'y_pred': y_test_pred
        }
        
        print(f"\nMétricas para {name}:")
        print(f"Validación - RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
        print(f"Prueba - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    
    return all_models_metrics, all_models_predictions

def generate_model_visualizations(models_predictions, features, models, output_path, all_models_metrics):
    """
    Genera visualizaciones para evaluar todos los modelos
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Paleta de colores para distinguir los modelos
    model_colors = {
        'LinearRegression': 'blue',
        'RandomForest': 'green',
        'GradientBoosting': 'red'
    }
    
    # 1. Gráfico comparativo de predicciones vs valores reales para todos los modelos
    plt.figure(figsize=(12, 8))
    
    for model_name, predictions in models_predictions.items():
        y_test = predictions['y_test']
        y_pred = predictions['y_pred']
        
        plt.scatter(y_test, y_pred, alpha=0.5, 
                   label=model_name, 
                   color=model_colors.get(model_name, 'gray'))
    
    # Línea de predicción perfecta
    max_val = max([max(pred['y_test'].max(), pred['y_pred'].max()) for pred in models_predictions.values()])
    min_val = min([min(pred['y_test'].min(), pred['y_pred'].min()) for pred in models_predictions.values()])
    
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    plt.title('Comparación de modelos: Valores reales vs. Predicciones de consumo promedio (m³)')
    plt.xlabel('Consumo promedio real (m³)')
    plt.ylabel('Consumo promedio predicho (m³)')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'comparacion_modelos_predicciones.png'))
    plt.close()
    
    # 2. Histograma de residuos para cada modelo
    plt.figure(figsize=(15, 10))
    
    for i, (model_name, predictions) in enumerate(models_predictions.items()):
        plt.subplot(2, 2, i+1)
        residuos = predictions['y_test'] - predictions['y_pred']
        plt.hist(residuos, bins=30, alpha=0.7, color=model_colors.get(model_name, 'gray'))
        plt.axvline(0, color='red', linestyle='--')
        plt.title(f'Distribución de residuos - {model_name}')
        plt.xlabel('Residuo (real - predicción) en m³')
        plt.ylabel('Frecuencia')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'distribucion_residuos_comparativo.png'))
    plt.close()
    
    # 3. Gráfico de importancia de características para modelos que lo permiten
    for model_name, model in models.items():
        if hasattr(model, 'named_steps') and hasattr(model.named_steps.get('model', None), 'feature_importances_'):
            try:
                # Para modelos basados en árboles con feature_importances_
                
                # Obtener el preprocesador
                preprocessor = model.named_steps['preprocessor']
                
                # Obtener los nombres de las características después de la transformación
                categorical_features = ['MUNICIPIO']
                numerical_features = [col for col in features if col not in categorical_features]
                
                # Para características numéricas, mantener nombres originales
                feature_names = numerical_features.copy()
                
                # Para características categóricas, tratar de reconstruir nombres
                if hasattr(preprocessor, 'transformers_'):
                    for name, transformer, cols in preprocessor.transformers_:
                        if name == 'cat':
                            if hasattr(transformer, 'named_steps') and 'onehot' in transformer.named_steps:
                                onehot = transformer.named_steps['onehot']
                                if hasattr(onehot, 'get_feature_names_out'):
                                    cat_features = onehot.get_feature_names_out(cols)
                                    feature_names.extend(cat_features)
                
                # Si hay desajuste en las dimensiones, usar nombres genéricos
                importances = model.named_steps['model'].feature_importances_
                if len(importances) != len(feature_names):
                    feature_names = [f'Feature {i}' for i in range(len(importances))]
                
                feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
                feature_importance = feature_importance.sort_values('importance', ascending=False)
                
                # Graficar importancias
                plt.figure(figsize=(12, 8))
                sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
                plt.title(f'Importancia de características en la predicción del consumo promedio - {model_name}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, f'importancia_caracteristicas_{model_name}.png'))
                plt.close()
            except Exception as e:
                print(f"Error al generar visualización de importancia para {model_name}: {e}")
    
    # 4. Comparación de métricas entre modelos
    plt.figure(figsize=(12, 6))
    
    # Preparar datos para gráfico de barras
    model_names = list(models_predictions.keys())
    rmse_test = [all_models_metrics[name]['test']['rmse'] for name in model_names]
    mae_test = [all_models_metrics[name]['test']['mae'] for name in model_names]
    r2_test = [all_models_metrics[name]['test']['r2'] for name in model_names]
    
    # Crear gráfico de barras para RMSE y MAE
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(x - width/2, rmse_test, width, label='RMSE', color='darkblue')
    ax1.bar(x + width/2, mae_test, width, label='MAE', color='lightblue')
    ax1.set_xlabel('Modelo')
    ax1.set_ylabel('Error (m³)')
    ax1.set_title('Comparación de métricas de error entre modelos')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.legend(loc='upper left')
    
    # Añadir R² en eje secundario
    ax2 = ax1.twinx()
    ax2.plot(x, r2_test, 'ro-', linewidth=2, label='R²')
    ax2.set_ylabel('R²')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'comparacion_metricas.png'))
    plt.close()
    
    print(f"Visualizaciones comparativas de modelos generadas en {output_path}")

def save_model_data(best_model, all_model_metrics, best_params, output_dir, best_model_name):
    """
    Guarda el mejor modelo y la información de todos los modelos
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar el mejor modelo
    joblib.dump(best_model, os.path.join(output_dir, 'best_model.pkl'))
    
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
        
        # Preparar características y variable objetivo
        X, y, features = prepare_features_target(df)
        
        # Dividir datos
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        # Crear pipeline de preprocesamiento
        preprocessor = create_preprocessing_pipeline(features)
        
        # Entrenar y comparar modelos
        model_results, best_model_name = train_models(X_train, y_train, preprocessor)
        
        # Ajustar hiperparámetros del mejor modelo
        best_model, best_params = tune_hyperparameters(best_model_name, X_train, y_train, preprocessor, model_results)
        
        # Evaluar todos los modelos
        all_models_metrics, all_models_predictions = evaluate_all_models(
            model_results, X_val, y_val, X_test, y_test, X_train, y_train
        )
        
        # Extraer los modelos entrenados de los resultados
        trained_models = {}
        for name, model_data in model_results.items():
            pipeline = model_data['pipeline']
            pipeline.fit(X_train, y_train)
            trained_models[name] = pipeline
        
        # Generar visualizaciones comparativas
        generate_model_visualizations(all_models_predictions, features, trained_models, viz_path, all_models_metrics)
        
        # Guardar modelo e información
        save_model_data(best_model, all_models_metrics, best_params, model_output_dir, best_model_name)
        
        print("Proceso de ingeniería del modelo completado con éxito")
        return True
    except Exception as e:
        import traceback
        print(f"Error en el proceso de ingeniería del modelo: {str(e)}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    main()