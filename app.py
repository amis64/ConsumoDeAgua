"""
Aplicación Flask para predicción de consumo hídrico
- Interfaz web para visualización y evaluación del modelo
- API REST para realizar predicciones
"""

from flask import Flask, render_template, jsonify, request
from model_service import ModelService
import pandas as pd
import os
import matplotlib
import joblib
import json
import traceback
from werkzeug.serving import run_simple
matplotlib.use('Agg')  # Configuración para usar matplotlib sin interfaz gráfica

app = Flask(__name__)

# Inicializar servicio del modelo (se carga en la primera solicitud)
model_service = None

def init_model_service():
    """Inicializa el servicio del modelo si el modelo está disponible"""
    global model_service
    if model_service is None:
        try:
            if os.path.exists('static/data/models/best_model.pkl'):
                model_service = ModelService()
                print("Servicio de modelo inicializado correctamente")
            else:
                print("Modelo no encontrado en static/data/models/best_model.pkl")
        except Exception as e:
            print(f"Error al inicializar servicio del modelo: {str(e)}")
            traceback.print_exc()

# Rutas existentes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/entendimiento-negocio')
def business_understanding():
    return render_template('business_understanding.html')

@app.route('/ingenieria-datos')
def data_engineering():
    # Verificar si existen los archivos procesados
    processed_data_exists = os.path.exists('static/data/data_processed.csv')
    visualizations_exist = os.path.exists('static/img/data_viz')
   
    # Obtener estadísticas si el archivo existe
    stats = {}
    if processed_data_exists:
        try:
            df = pd.read_csv('static/data/data_processed.csv')
            
            # Calcular número de municipios - pueden estar en formato one-hot
            num_municipalities = 0
            if 'MUNICIPIO' in df.columns:
                num_municipalities = df['MUNICIPIO'].nunique()
            else:
                # Buscar columnas de municipio en formato one-hot
                municipio_cols = [col for col in df.columns if col.startswith('Municipio_')]
                num_municipalities = len(municipio_cols)
            
            # Calcular consumo promedio - ahora es nuestra variable objetivo
            avg_consumption = 0
            if 'PROMEDIO CONSUMO ACUEDUCTO' in df.columns:
                avg_consumption = df['PROMEDIO CONSUMO ACUEDUCTO'].mean()
            
            # Contar número total de features
            num_features = len(df.columns)
            
            stats = {
                'num_records': len(df),
                'num_municipalities': num_municipalities,
                'avg_consumption': avg_consumption,
                'num_features': num_features
            }
        except Exception as e:
            stats = {'error': str(e)}
   
    return render_template(
        'data_engineering.html',
        processed_data_exists=processed_data_exists,
        visualizations_exist=visualizations_exist,
        stats=stats
    )

@app.route('/ingenieria-modelo')
def model_engineering():
    # Verificar si existe el modelo entrenado
    model_exists = os.path.exists('static/data/models/best_model.pkl')
    model_visualizations_exist = os.path.exists('static/img/model_viz')
    
    # Obtener información del modelo si existe
    model_info = {}
    if model_exists and os.path.exists('static/data/models/model_info.pkl'):
        try:
            model_info = joblib.load('static/data/models/model_info.pkl')
        except Exception as e:
            model_info = {'error': str(e)}
    
    return render_template(
        'model_engineering.html',
        model_exists=model_exists,
        model_visualizations_exist=model_visualizations_exist,
        model_info=model_info
    )

@app.route('/train-model')
def train_model():
    try:
        from model_engineering import main
        result = main()
        return jsonify({'success': result, 'message': 'Modelo entrenado correctamente' if result else 'Error al entrenar el modelo'})
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({'success': False, 'message': str(e), 'details': error_details})

@app.route('/process-data')
def process_data():
    try:
        print("Iniciando procesamiento de datos...")
        # Verificar que el archivo existe
        import os
        input_file = 'static/data/HISTORICO_CONSUMO_POR_ESTRATO_20250506.csv'
        if not os.path.exists(input_file):
            error_msg = f"El archivo {input_file} no existe"
            print(error_msg)
            return jsonify({'success': False, 'message': error_msg})
        
        # Crear directorios necesarios
        os.makedirs('static/data', exist_ok=True)
        os.makedirs('static/img/data_viz', exist_ok=True)
        
        print("Importando módulo de procesamiento...")
        from data_processing import main
        print("Ejecutando procesamiento...")
        main()
        print("Procesamiento completado con éxito")
        return jsonify({'success': True, 'message': 'Datos procesados correctamente'})
    except Exception as e:
        import traceback
        error_msg = f"Error en el procesamiento: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'success': False, 'message': str(e)})

# ==== NUEVAS RUTAS PARA DESPLIEGUE ====

# Nueva interfaz de predicción
@app.route('/prediccion')
def prediction_interface():
    """Interfaz web para realizar predicciones"""
    # Inicializar servicio si no existe
    init_model_service()
    
    # Verificar si el modelo está cargado
    model_loaded = model_service is not None
    
    # Obtener valores válidos si el modelo está disponible
    valid_values = None
    if model_loaded:
        valid_values = model_service.get_valid_values()
    
    return render_template(
        'prediction.html',
        model_loaded=model_loaded,
        valid_values=valid_values
    )

# Documentación de API
@app.route('/api_documentacion')
def api_documentation():
    """Documentación de la API"""
    # Inicializar servicio si no existe
    init_model_service()
    
    # Verificar si el modelo está cargado
    model_loaded = model_service is not None
    
    # Obtener valores válidos si el modelo está disponible
    valid_values = None
    if model_loaded:
        valid_values = model_service.get_valid_values()
    
    # Obtener base URL 
    host = request.host_url.rstrip('/')
    
    return render_template(
        'api_documentation.html',
        model_loaded=model_loaded,
        valid_values=valid_values,
        base_url=f"{host}/api/v1"
    )

# ==== API ENDPOINTS ====

@app.route('/api/v1/predict', methods=['POST'])
def api_predict():
    """Endpoint de API para predicción individual"""
    # Inicializar servicio si no existe
    init_model_service()
    
    # Verificar si el modelo está disponible
    if model_service is None:
        return jsonify({
            'success': False,
            'error': 'Modelo no disponible. Entrene el modelo primero.'
        }), 503  # Service Unavailable
    
    try:
        # Obtener datos de la solicitud
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No se recibieron datos JSON válidos'
            }), 400  # Bad Request
        
        # Validar campos requeridos
        required_fields = ['MES', 'MUNICIPIO', 'ESTRATO']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Campos requeridos no encontrados: {", ".join(missing_fields)}'
            }), 400  # Bad Request
        
        # Convertir ESTRATO a entero si es string
        if isinstance(data['ESTRATO'], str) and data['ESTRATO'].isdigit():
            data['ESTRATO'] = int(data['ESTRATO'])
        
        # Realizar predicción
        result = model_service.predict(data)
        
        # Agregar campo de éxito
        result['success'] = True
        
        return jsonify(result)
    
    except ValueError as e:
        # Error en datos de entrada
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400  # Bad Request
    
    except Exception as e:
        # Error inesperado
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error interno: {str(e)}'
        }), 500  # Internal Server Error

@app.route('/api/v1/predict_batch', methods=['POST'])
def api_predict_batch():
    """Endpoint de API para predicciones en lote"""
    # Inicializar servicio si no existe
    init_model_service()
    
    # Verificar si el modelo está disponible
    if model_service is None:
        return jsonify({
            'success': False,
            'error': 'Modelo no disponible. Entrene el modelo primero.'
        }), 503  # Service Unavailable
    
    try:
        # Obtener datos de la solicitud
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No se recibieron datos JSON válidos'
            }), 400  # Bad Request
        
        # Verificar que sea una lista
        if not isinstance(data, list):
            return jsonify({
                'success': False,
                'error': 'Se esperaba una lista de objetos para predicción'
            }), 400  # Bad Request
        
        # Limitar número máximo de predicciones
        if len(data) > 100:
            return jsonify({
                'success': False,
                'error': 'Máximo 100 predicciones por solicitud'
            }), 400  # Bad Request
        
        # Convertir ESTRATO a entero si es string
        for item in data:
            if isinstance(item.get('ESTRATO'), str) and item['ESTRATO'].isdigit():
                item['ESTRATO'] = int(item['ESTRATO'])
        
        # Realizar predicciones
        results = model_service.predict_batch(data)
        
        # Si es un diccionario con errores, mantener estructura
        if isinstance(results, dict) and 'predictions' in results:
            results['success'] = True
            return jsonify(results)
        
        # Si es lista directa de resultados, formatear respuesta
        return jsonify({
            'success': True,
            'predictions': results
        })
    
    except ValueError as e:
        # Error en datos de entrada
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400  # Bad Request
    
    except Exception as e:
        # Error inesperado
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error interno: {str(e)}'
        }), 500  # Internal Server Error

@app.route('/api/v1/model_info', methods=['GET'])
def api_model_info():
    """Endpoint de API para obtener información del modelo"""
    # Inicializar servicio si no existe
    init_model_service()
    
    # Verificar si el modelo está disponible
    if model_service is None:
        return jsonify({
            'success': False,
            'error': 'Modelo no disponible. Entrene el modelo primero.'
        }), 503  # Service Unavailable
    
    try:
        # Obtener información del modelo
        model_info = model_service.get_model_info()
        
        # Obtener valores válidos para entrada
        valid_values = model_service.get_valid_values()
        
        # Combinar información
        result = {
            'success': True,
            'model_name': model_info['name'],
            'metrics': model_info['metrics'],
            'parameters': model_info['parameters'],
            'valid_values': valid_values
        }
        
        return jsonify(result)
    
    except Exception as e:
        # Error inesperado
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error interno: {str(e)}'
        }), 500  # Internal Server Error

@app.route('/api/v1/health', methods=['GET'])
def api_health():
    """Endpoint de API para verificar estado del servicio"""
    # Inicializar servicio si no existe
    init_model_service()
    
    # Verificar estado
    model_loaded = model_service is not None
    model_name = model_service.model_info['name'] if model_loaded else None
    
    return jsonify({
        'success': True,
        'status': 'healthy' if model_loaded else 'model_not_loaded',
        'model_loaded': model_loaded,
        'model_name': model_name
    })

# Antiguo endpoint de predicción (mantener por compatibilidad)
@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint antiguo de predicción (redireccionando al nuevo)"""
    try:
        # Inicializar servicio si no existe
        init_model_service()
        
        # Verificar si el modelo está disponible
        if model_service is None:
            return jsonify({
                'success': False,
                'message': 'Modelo no disponible. Entrene el modelo primero.'
            }), 503  # Service Unavailable
        
        # Obtener datos de la solicitud
        data = request.get_json()
        
        # Realizar predicción con el nuevo servicio
        result = model_service.predict(data)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'unit': result['unit']
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    # Inicializar servicio del modelo al arrancar
    init_model_service()
    
    # Ejecutar aplicación Flask
    app.run(debug=True, host='0.0.0.0', port=5000)
