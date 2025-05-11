from flask import Flask, render_template, jsonify
import pandas as pd
import os
import matplotlib
import joblib
import json
matplotlib.use('Agg')  # Configuración para usar matplotlib sin interfaz gráfica


app = Flask(__name__)

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

# Añadir ruta para ejecutar el proceso de modelado
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


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Cargar el modelo
        model = joblib.load('static/data/models/best_model.pkl')
        
        # Obtener datos de la solicitud
        data = json.request.get_json()
        
        # Preparar los datos para la predicción
        # (Esto dependerá de cómo se estructuren los datos de entrada)
        
        # Realizar predicción
        # prediction = model.predict([input_data])[0]
        
        return jsonify({'success': True, 'prediction': 'Implementación pendiente'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

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

if __name__ == '__main__':
    app.run(debug=True)