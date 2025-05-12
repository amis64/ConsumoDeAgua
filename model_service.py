"""
ModelService - Servicio para cargar y utilizar el modelo entrenado

Este módulo proporciona funcionalidad para:
- Cargar el modelo RandomForest entrenado y su preprocesador
- Preparar datos de entrada para predicción
- Realizar predicciones individuales y en lote
- Obtener información sobre el modelo y valores válidos
"""

import joblib
import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Union, Any, Optional

class ModelService:
    """
    Servicio para gestionar predicciones con el modelo de consumo hídrico
    """
    
    def __init__(self, model_path: str = 'static/data/models/'):
        """
        Inicializa el servicio cargando el modelo y metadatos necesarios
        
        Args:
            model_path: Ruta donde se almacena el modelo entrenado
        """
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.model_info = None
        self.feature_columns = []
        self.target_column = 'PROMEDIO CONSUMO ACUEDUCTO'
        
        # Definir mapeo de meses a número
        self.month_map = {
            'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4,
            'MAYO': 5, 'JUNIO': 6, 'JULIO': 7, 'AGOSTO': 8,
            'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
        }
        
        # Log constant for inverse transformation if needed
        self.log_constant = 1
        
        # Cargar el modelo y metadatos
        self._load_model()
        self._load_feature_columns()
    
    def _load_model(self) -> None:
        """Carga el modelo entrenado y su información"""
        try:
            # Cargar modelo y preprocesador
            model_data = joblib.load(os.path.join(self.model_path, 'best_model.pkl'))
            self.model = model_data['model']
            self.preprocessor = model_data.get('preprocessor')
            
            # Cargar información del modelo
            self.model_info = joblib.load(os.path.join(self.model_path, 'model_info.pkl'))
            print(f"Modelo '{self.model_info['name']}' cargado correctamente")
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            raise RuntimeError(f"No se pudo cargar el modelo: {str(e)}")
    
    def _load_feature_columns(self) -> None:
        """Carga la estructura de columnas del dataset procesado"""
        try:
            # Cargar un registro del dataset procesado para obtener estructura
            sample_df = pd.read_csv('static/data/data_processed.csv', nrows=1)
            self.feature_columns = [col for col in sample_df.columns 
                                  if col != self.target_column and
                                     col != f"{self.target_column}_log"]
            print(f"Estructura de features cargada: {len(self.feature_columns)} columnas")
        except Exception as e:
            print(f"Error al cargar estructura de features: {str(e)}")
            # Intentamos continuar con columnas mínimas
            self.feature_columns = ['MES', 'MES_NUM']
    
    def prepare_features(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepara un dataframe con formato adecuado para predicción a partir de datos de entrada
        
        Args:
            input_data: Diccionario con datos de entrada (MES, MUNICIPIO, ESTRATO)
            
        Returns:
            DataFrame con formato adecuado para predicción
        """
        # Verificar entradas requeridas
        required_fields = ['MES', 'MUNICIPIO', 'ESTRATO']
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Campo requerido no encontrado: {field}")
        
        # Crear DataFrame con una fila
        df = pd.DataFrame([input_data])
        
        # Convertir MES a número
        if 'MES' in df.columns:
            mes = df['MES'].iloc[0]
            if mes in self.month_map:
                df['MES_NUM'] = self.month_map[mes]
            else:
                raise ValueError(f"Mes no válido: {mes}")
        
        # Aplicar one-hot encoding para ESTRATO y MUNICIPIO
        # Crear columnas de estrato (solo activar la correspondiente)
        for i in range(1, 7):  # Estratos del 1 al 6
            estrato_col = f'Estrato_{i}'
            if estrato_col in self.feature_columns:
                df[estrato_col] = (df['ESTRATO'] == i)
        
        # Crear columnas de municipio (solo activar el correspondiente)
        for col in self.feature_columns:
            if col.startswith('Municipio_'):
                municipio = col.replace('Municipio_', '')
                df[col] = (df['MUNICIPIO'] == municipio)
        
        # Filtrar solo las columnas necesarias para el modelo
        model_columns = [col for col in self.feature_columns if col in df.columns]
        
        # Verificar si faltan columnas
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            print(f"Columnas faltantes: {missing_cols}")
            # Agregar columnas faltantes con valor False (para one-hot encoding)
            for col in missing_cols:
                df[col] = False
        
        # Asegurar que solo usamos las columnas del modelo y en el orden correcto
        result_df = df[self.feature_columns].copy()
        
        return result_df
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Realiza una predicción individual
        
        Args:
            input_data: Diccionario con datos de entrada (MES, MUNICIPIO, ESTRATO)
            
        Returns:
            Diccionario con resultado de predicción
        """
        try:
            # Preparar features
            X = self.prepare_features(input_data)
            
            # Aplicar preprocesamiento si existe
            if self.preprocessor:
                # Aquí asumimos que el preprocesador ya sabe qué columnas transformar
                X_processed = X.copy()
                # Solo aplicamos a MES_NUM si existe en el preprocesador
                if 'MES_NUM' in X.columns:
                    X_mes = X[['MES_NUM']]
                    X_mes_scaled = self.preprocessor.transform(X_mes)
                    X_processed['MES_NUM'] = X_mes_scaled.flatten()
            else:
                X_processed = X
            
            # Realizar predicción
            prediction = self.model.predict(X_processed)[0]
            
            # Si se aplicó transformación logarítmica, revertirla
            # Verificamos si el modelo usa transformación log viendo las métricas
            uses_log = False
            for metric_key in self.model_info['metrics'].keys():
                if '_log' in metric_key:
                    uses_log = True
                    break
            
            if uses_log:
                prediction = np.exp(prediction) - self.log_constant
            
            # Construir respuesta
            result = {
                'prediction': float(prediction),
                'unit': 'm³/suscriptor',
                'input_data': input_data,
                'model_used': self.model_info['name'],
                'confidence': {
                    'r2_test': float(self.model_info['metrics']['test']['r2']),
                    'rmse_test': float(self.model_info['metrics']['test']['rmse'])
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error en predicción: {str(e)}")
            raise ValueError(f"Error al realizar predicción: {str(e)}")
    
    def predict_batch(self, input_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Realiza predicciones en lote
        
        Args:
            input_data_list: Lista de diccionarios con datos de entrada
            
        Returns:
            Lista de resultados de predicción
        """
        results = []
        errors = []
        
        for i, data in enumerate(input_data_list):
            try:
                result = self.predict(data)
                results.append(result)
            except Exception as e:
                # Registrar error pero continuar con otras predicciones
                error_info = {
                    'index': i,
                    'input_data': data,
                    'error': str(e)
                }
                errors.append(error_info)
                print(f"Error en predicción {i}: {str(e)}")
        
        # Si todas las predicciones fallaron, lanzar excepción
        if len(errors) == len(input_data_list):
            raise ValueError(f"Todas las predicciones fallaron: {len(errors)} errores")
        
        # Si algunas fallaron, incluir información de errores
        if errors:
            return {
                'predictions': results,
                'errors': errors,
                'success_rate': f"{len(results)}/{len(input_data_list)}"
            }
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna información sobre el modelo
        
        Returns:
            Diccionario con información del modelo
        """
        return {
            'name': self.model_info['name'],
            'metrics': self.model_info['metrics'],
            'parameters': self.model_info.get('params', {}),
            'feature_count': len(self.feature_columns)
        }
    
    def get_valid_values(self) -> Dict[str, List]:
        """
        Retorna valores válidos para inputs de predicción
        
        Returns:
            Diccionario con valores válidos para cada campo
        """
        # Extraer municipios de las columnas
        municipios = [col.replace('Municipio_', '') 
                     for col in self.feature_columns 
                     if col.startswith('Municipio_')]
        
        # Extraer estratos de las columnas
        estratos = []
        for col in self.feature_columns:
            if col.startswith('Estrato_'):
                try:
                    estrato = int(col.replace('Estrato_', ''))
                    estratos.append(estrato)
                except ValueError:
                    pass
        
        return {
            'meses': list(self.month_map.keys()),
            'municipios': sorted(municipios),
            'estratos': sorted(estratos)
        }
