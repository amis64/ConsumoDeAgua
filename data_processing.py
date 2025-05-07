import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

def load_data(file_path):
    """
    Carga el dataset desde la ruta especificada con manejo de diferentes codificaciones
    """
    try:
        # Intentar diferentes codificaciones
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='latin-1')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='ISO-8859-1')
        
        # Limpiar caracteres BOM de los nombres de columnas
        df.columns = [col.replace('ï»¿', '') if 'ï»¿' in col else col for col in df.columns]
        
        print(f"Dataset cargado correctamente. Forma: {df.shape}")
        print(f"Columnas detectadas: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return None

def select_columns(df):
    """
    Selecciona solo las columnas relevantes para el análisis
    """
    # Mapeo de columnas necesarias
    column_mapping = {
        'MES': 'MES',
        'MUNICIPIO': 'MUNICIPIO', 
        'ESTRATO': 'ESTRATO',
        'PROMEDIO CONSUMO ACUEDUCTO': 'PROMEDIO CONSUMO ACUEDUCTO'
    }
    
    try:
        # Seleccionar solo las columnas necesarias según el nuevo enfoque
        cols_to_keep = [column_mapping[col] for col in ['MES', 'MUNICIPIO', 'ESTRATO', 
                                                       'PROMEDIO CONSUMO ACUEDUCTO']]
        
        df_selected = df[cols_to_keep].copy()  # Usar .copy() para evitar SettingWithCopyWarning
        
        print(f"Columnas seleccionadas. Nuevas dimensiones: {df_selected.shape}")
        return df_selected
    except Exception as e:
        print(f"Error al seleccionar columnas: {e}")
        print("Columnas disponibles:", df.columns.tolist())
        return None
    

def clean_data(df):
    """
    Limpia los datos, enfocándose solo en el consumo promedio
    """
    # Asegurar que trabajamos con una copia
    df = df.copy()
    
    print(f"Registros iniciales: {len(df)}")
    
    # Filtrar solo los registros residenciales (Estrato1 a Estrato6)
    df = df[df['ESTRATO'].str.contains('Estrato', case=False, na=False)]
    print(f"Registros después de filtrar solo estratos residenciales: {len(df)}")
    
    # Convertir el estrato a número
    df['ESTRATO_NUM'] = df['ESTRATO'].str.extract(r'Estrato(\d+)', expand=False)
    df = df.dropna(subset=['ESTRATO_NUM'])  # Eliminar filas donde no se pudo extraer un número
    print(f"Registros después de limpiar estratos: {len(df)}")
    
    # Asegurar que el consumo promedio sea numérico
    df['PROMEDIO_NUM'] = pd.to_numeric(df['PROMEDIO CONSUMO ACUEDUCTO'], errors='coerce')
    
    # Eliminar registros con valores inválidos o cero
    df = df.dropna(subset=['PROMEDIO_NUM'])
    df = df[df['PROMEDIO_NUM'] > 0]  # Eliminar registros con consumo promedio cero o negativo
    
    print(f"Registros después de limpieza numérica: {len(df)}")
    
    # Construir el DataFrame final con las columnas clave según el nuevo enfoque
    df_final = pd.DataFrame({
        'MES': df['MES'],
        'MUNICIPIO': df['MUNICIPIO'],
        'ESTRATO': df['ESTRATO_NUM'].astype(int),
        'PROMEDIO CONSUMO ACUEDUCTO': df['PROMEDIO_NUM']
    })
    
    print(f"\nRegistros finales después de limpieza completa: {len(df_final)}")
    
    return df_final

def analyze_municipality_balance(df):
    """
    Analiza el balance de datos por municipio y regresa un dataframe equilibrado si es necesario
    """
    # Contar registros por municipio
    municipio_counts = df['MUNICIPIO'].value_counts()
    total_registros = len(df)
    
    print("\n=== Balance de datos por municipio ===")
    print(municipio_counts)
    
    # Calcular porcentajes
    municipio_pct = (municipio_counts / total_registros * 100).round(2)
    print("\nPorcentaje de registros por municipio:")
    for municipio, pct in municipio_pct.items():
        print(f"{municipio}: {pct}%")
    
    # Identificar municipios con muy pocos datos (menos del 1%)
    municipios_escasos = municipio_pct[municipio_pct < 1].index.tolist()
    if municipios_escasos:
        print(f"\nMunicipios con menos del 1% de los datos: {municipios_escasos}")
        
        # Preguntar si se eliminan (en este caso lo dejamos a criterio automático)
        if len(municipios_escasos) > len(municipio_pct) * 0.3:  # Si son más del 30% de los municipios
            print("Hay muchos municipios con pocos datos. Se mantendrán para preservar la diversidad geográfica.")
        else:
            print(f"Se eliminarán {len(municipios_escasos)} municipios con pocos datos para mejorar el balance.")
            df = df[~df['MUNICIPIO'].isin(municipios_escasos)]
            
    # Identificar municipios sobre-representados (más del 20%)
    municipios_sobrerep = municipio_pct[municipio_pct > 20].index.tolist()
    if municipios_sobrerep:
        print(f"\nMunicipios sobre-representados (>20%): {municipios_sobrerep}")
        
        # Determinar si es necesario equilibrar
        if max(municipio_pct) > 30:  # Si algún municipio tiene más del 30%
            print("Equilibrando dataset para reducir sesgo geográfico...")
            
            # Encontrar el número objetivo de registros por municipio (50% del municipio más representado)
            target_count = int(municipio_counts[municipios_sobrerep[0]] * 0.5)
            
            # Crear dataframe equilibrado
            df_balanced = pd.DataFrame()
            
            for municipio in df['MUNICIPIO'].unique():
                df_muni = df[df['MUNICIPIO'] == municipio]
                
                if len(df_muni) > target_count:
                    # Muestrear aleatoriamente
                    df_muni = df_muni.sample(target_count, random_state=42)
                
                df_balanced = pd.concat([df_balanced, df_muni])
            
            print(f"Dataset equilibrado. Registros originales: {len(df)}, Registros equilibrados: {len(df_balanced)}")
            return df_balanced
    
    return df

def transform_data(df):
    """
    Realiza transformaciones adicionales en los datos
    """
    # Convertir MES a número
    meses = {
        'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
        'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
    }
    df['MES_NUM'] = df['MES'].map(meses)
    
    # Normalizar nombres de municipios (capitalizar)
    df['MUNICIPIO'] = df['MUNICIPIO'].str.title()
    
    # Aplicar one-hot encoding para ESTRATO y MUNICIPIO
    df_dummies = pd.get_dummies(df, columns=['ESTRATO', 'MUNICIPIO'], prefix=['Estrato', 'Municipio'])
    
    print("Transformaciones completadas")
    print(f"Dimensiones después de one-hot encoding: {df_dummies.shape}")
    
    return df_dummies

def generate_visualizations(df, output_path):
    """
    Genera visualizaciones para entender los datos
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Crear una versión del dataframe sin one-hot encoding para visualizaciones
    df_viz = df.copy()
    if 'ESTRATO' not in df.columns:
        # Reconstruir la columna ESTRATO a partir de columnas dummies
        estrato_cols = [col for col in df.columns if col.startswith('Estrato_')]
        if estrato_cols:
            for col in estrato_cols:
                estrato_num = col.split('_')[1]
                df_viz.loc[df[col] == 1, 'ESTRATO'] = int(estrato_num)
    
    # 1. Distribución de consumo promedio por estrato
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='ESTRATO', y='PROMEDIO CONSUMO ACUEDUCTO', data=df_viz)
    plt.title('Distribución del consumo promedio por estrato')
    plt.xlabel('Estrato socioeconómico')
    plt.ylabel('Consumo promedio (m³)')
    plt.savefig(os.path.join(output_path, 'consumo_por_estrato.png'))
    plt.close()  # Cerrar la figura para liberar memoria
    
    # 2. Tendencia mensual de consumo por estrato
    plt.figure(figsize=(12, 6))
    
    for estrato in sorted(df_viz['ESTRATO'].unique()):
        datos_estrato = df_viz[df_viz['ESTRATO'] == estrato]
        consumo_mensual = datos_estrato.groupby('MES_NUM')['PROMEDIO CONSUMO ACUEDUCTO'].mean()
        plt.plot(consumo_mensual.index, consumo_mensual.values, label=f'Estrato {estrato}')
    
    plt.title('Tendencia de consumo promedio por estrato a lo largo del año')
    plt.xlabel('Mes')
    plt.ylabel('Consumo promedio (m³)')
    plt.xticks(range(1, 13), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'tendencia_consumo.png'))
    plt.close()
    
    # 3. Mapa de calor de consumo por municipio y estrato
    # Reconstruir dataframe para el mapa de calor
    if 'MUNICIPIO' not in df_viz.columns or 'ESTRATO' not in df_viz.columns:
        # Recrear un dataframe con municipio y estrato
        df_heatmap = pd.DataFrame()
        
        # Reconstruir municipio
        municipio_cols = [col for col in df.columns if col.startswith('Municipio_')]
        for col in municipio_cols:
            municipio_name = col.replace('Municipio_', '')
            mask = df[col] == 1
            if mask.any():
                temp_df = df[mask].copy()
                temp_df['MUNICIPIO'] = municipio_name
                df_heatmap = pd.concat([df_heatmap, temp_df])
        
        # Reconstruir estrato
        estrato_cols = [col for col in df.columns if col.startswith('Estrato_')]
        for col in estrato_cols:
            estrato_num = col.replace('Estrato_', '')
            mask = df[col] == 1
            if mask.any():
                df_heatmap.loc[mask, 'ESTRATO'] = int(estrato_num)
    else:
        df_heatmap = df_viz.copy()
    
    # Crear pivot table
    consumo_municipio_estrato = df_heatmap.pivot_table(
        values='PROMEDIO CONSUMO ACUEDUCTO', 
        index='MUNICIPIO', 
        columns='ESTRATO', 
        aggfunc='mean'
    ).fillna(0)
    
    # Limitar a un número manejable de municipios si hay muchos
    if len(consumo_municipio_estrato) > 20:
        # Ordenar por consumo total y tomar los top 20
        total_consumo = consumo_municipio_estrato.sum(axis=1)
        top_municipios = total_consumo.sort_values(ascending=False).head(20).index
        consumo_municipio_estrato = consumo_municipio_estrato.loc[top_municipios]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(consumo_municipio_estrato, cmap='YlGnBu', annot=True, fmt='.1f')
    plt.title('Consumo promedio por municipio y estrato')
    plt.xlabel('Estrato socioeconómico')
    plt.ylabel('Municipio')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'mapa_calor_consumo.png'))
    plt.close()
    
    # 4. Distribución del consumo promedio
    plt.figure(figsize=(10, 6))
    sns.histplot(df['PROMEDIO CONSUMO ACUEDUCTO'], kde=True, bins=30)
    plt.title('Distribución del consumo promedio de agua')
    plt.xlabel('Consumo promedio (m³)')
    plt.ylabel('Frecuencia')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_path, 'distribucion_consumo_promedio.png'))
    plt.close()
    
    # 5. Comparación de consumo por municipio
    plt.figure(figsize=(14, 8))
    
    # Agrupar por municipio y calcular promedio
    consumo_por_municipio = df_heatmap.groupby('MUNICIPIO')['PROMEDIO CONSUMO ACUEDUCTO'].mean().sort_values(ascending=False)
    
    # Limitar a 15 municipios para mejor visualización
    if len(consumo_por_municipio) > 15:
        consumo_por_municipio = consumo_por_municipio.head(15)
    
    # Crear gráfico de barras
    sns.barplot(x=consumo_por_municipio.values, y=consumo_por_municipio.index)
    plt.title('Consumo promedio por municipio')
    plt.xlabel('Consumo promedio (m³)')
    plt.ylabel('Municipio')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'consumo_por_municipio.png'))
    plt.close()
    
    print("Visualizaciones generadas en", output_path)

def main():
    """
    Función principal que ejecuta todo el proceso de ingeniería de datos
    """
    try:
        # Definir rutas
        input_file = 'static/data/HISTORICO_CONSUMO_POR_ESTRATO_20250506.csv'
        output_file = 'static/data/data_processed.csv'
        viz_path = 'static/img/data_viz'
        
        # Asegurar que los directorios existan
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        os.makedirs(viz_path, exist_ok=True)
        
        # Verificar existencia del archivo
        if not os.path.exists(input_file):
            print(f"ERROR: El archivo {input_file} no existe")
            return False
        
        # Cargar datos
        df = load_data(input_file)
        if df is None:
            return False
        
        # Seleccionar columnas
        df = select_columns(df)
        if df is None:
            return False
        
        # Limpiar datos
        df = clean_data(df)
        
        # Analizar balance de datos por municipio
        df = analyze_municipality_balance(df)
        
        # Transformar datos
        df = transform_data(df)
        
        # Guardar datos procesados
        df.to_csv(output_file, index=False)
        print(f"Datos procesados guardados en {output_file}")
        
        # Generar visualizaciones
        generate_visualizations(df, viz_path)
        
        print("Proceso de ingeniería de datos completado con éxito")
        return True
    except Exception as e:
        import traceback
        print(f"Error en el proceso de ingeniería de datos: {str(e)}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    main()