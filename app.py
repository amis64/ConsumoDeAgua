from flask import Flask, render_template, jsonify
import pandas as pd
import os
import matplotlib
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
            stats = {
                'num_records': len(df),
                'num_municipalities': df['MUNICIPIO'].nunique(),
                'years_range': f"{df['AÑO'].min()} - {df['AÑO'].max()}",
                # Usar los valores sin modificar
                'total_subscribers': df['No. SUSCRIPTORES ACUEDUCTO'].sum(),
                'total_consumption': df['CONSUMO M3 ACUEDUCTO'].sum()
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
    return render_template('model_engineering.html')

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