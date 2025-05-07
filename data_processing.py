import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    # Detectar la columna de año - que puede aparecer como AÑO o AÃ'O
    year_column = [col for col in df.columns if 'A' in col and ('Ñ' in col or 'Ã' in col)][0]
    print(f"Columna de año detectada como: {year_column}")
    
    # Mapeo de columnas necesarias con sus posibles variantes
    column_mapping = {
        'AÑO': year_column,
        'MES': 'MES',
        'MUNICIPIO': 'MUNICIPIO', 
        'ESTRATO': 'ESTRATO',
        'No. SUSCRIPTORES ACUEDUCTO': 'No. SUSCRIPTORES ACUEDUCTO',
        'CONSUMO M3 ACUEDUCTO': 'CONSUMO M3 ACUEDUCTO',
        'PROMEDIO CONSUMO ACUEDUCTO': 'PROMEDIO CONSUMO ACUEDUCTO'
    }
    
    try:
        # Seleccionar columnas usando el mapeo
        cols_to_keep = [column_mapping[col] for col in ['AÑO', 'MES', 'MUNICIPIO', 'ESTRATO', 
                                                       'No. SUSCRIPTORES ACUEDUCTO', 
                                                       'CONSUMO M3 ACUEDUCTO', 
                                                       'PROMEDIO CONSUMO ACUEDUCTO']]
        
        df_selected = df[cols_to_keep].copy()  # Usar .copy() para evitar SettingWithCopyWarning
        
        # Renombrar la columna de año para consistencia
        df_selected = df_selected.rename(columns={year_column: 'AÑO'})
        
        print(f"Columnas seleccionadas. Nuevas dimensiones: {df_selected.shape}")
        return df_selected
    except Exception as e:
        print(f"Error al seleccionar columnas: {e}")
        print("Columnas disponibles:", df.columns.tolist())
        return None
    

def clean_data(df):
    """
    Limpia los datos tratando todo como strings hasta la conversión final,
    preservando valores enteros que aparecen como flotantes con ".0"
    """
    # Asegurar que trabajamos con una copia
    df = df.copy()
    
    print(f"Registros iniciales: {len(df)}")
    
    # Filtrar solo los registros residenciales (Estrato1 a Estrato6)
    df = df[df['ESTRATO'].str.contains('Estrato', case=False, na=False)]
    print(f"Registros después de filtrar solo estratos residenciales: {len(df)}")
    
    # Convertir el estrato a número manteniendo los valores originales como strings
    df['ESTRATO_NUM'] = df['ESTRATO'].str.extract(r'Estrato(\d+)', expand=False)
    df = df.dropna(subset=['ESTRATO_NUM'])  # Eliminar filas donde no se pudo extraer un número
    print(f"Registros después de limpiar estratos: {len(df)}")
    
    # PASO 1: Convertir todo a strings para evitar conversiones automáticas
    cols_to_clean = ['No. SUSCRIPTORES ACUEDUCTO', 'CONSUMO M3 ACUEDUCTO', 'PROMEDIO CONSUMO ACUEDUCTO']
    for col in cols_to_clean:
        df[f"{col}_STR"] = df[col].astype(str)
    
    # Mostrar algunos ejemplos para diagnóstico
    print("\nEjemplos de valores originales:")
    for i in range(min(10, len(df))):
        print(f"Fila {i+1}: Suscriptores: {df['No. SUSCRIPTORES ACUEDUCTO_STR'].iloc[i]}, " +
              f"Consumo: {df['CONSUMO M3 ACUEDUCTO_STR'].iloc[i]}, " +
              f"Promedio: {df['PROMEDIO CONSUMO ACUEDUCTO_STR'].iloc[i]}")
    
    # PASO 2: Identificar valores problemáticos pero EXCLUYENDO aquellos que terminan en ".0"
    # Patrón que excluye específicamente ".0" al final
    patron_problematico = r'\.\d{1,2}$'
    patron_float_entero = r'\.0$'
    
    # Usar funciones lambda para aplicar la lógica específica
    def es_problematico(valor):
        if pd.isna(valor):
            return False
        valor_str = str(valor)
        # Si termina en ".0", NO es problemático (es un entero representado como float)
        if re.search(patron_float_entero, valor_str):
            return False
        # Si tiene un punto seguido de 1-2 dígitos (y no es ".0"), SÍ es problemático
        return bool(re.search(patron_problematico, valor_str))
    
    # Aplicar las funciones a las columnas
    import re
    problematicos_suscriptores = df['No. SUSCRIPTORES ACUEDUCTO_STR'].apply(es_problematico)
    problematicos_consumo = df['CONSUMO M3 ACUEDUCTO_STR'].apply(es_problematico)
    
    # Mostrar ejemplos de valores problemáticos
    if problematicos_suscriptores.any() or problematicos_consumo.any():
        print("\nEjemplos de valores con formato problemático:")
        if problematicos_suscriptores.any():
            print("Suscriptores problemáticos:", df.loc[problematicos_suscriptores, 'No. SUSCRIPTORES ACUEDUCTO_STR'].head(3).tolist())
        if problematicos_consumo.any():
            print("Consumo problemático:", df.loc[problematicos_consumo, 'CONSUMO M3 ACUEDUCTO_STR'].head(3).tolist())
    
    # PASO 3: Eliminar solo los registros realmente problemáticos
    registros_problematicos = problematicos_suscriptores | problematicos_consumo
    df = df[~registros_problematicos]
    print(f"Registros eliminados por formato decimal incorrecto: {registros_problematicos.sum()}")
    print(f"Registros restantes: {len(df)}")
    
    # PASO 4: Limpiar los datos para conversión numérica
    # Para valores que terminan en ".0", simplemente eliminar el ".0"
    def limpiar_valor(valor):
        valor_str = str(valor)
        # Si termina en ".0", eliminar el ".0"
        if re.search(r'\.0$', valor_str):
            return valor_str.replace('.0', '')
        # Si tiene otros puntos (separadores de miles), eliminarlos
        return valor_str.replace('.', '')
    
    df['SUSCRIPTORES_LIMPIO'] = df['No. SUSCRIPTORES ACUEDUCTO_STR'].apply(limpiar_valor)
    df['CONSUMO_LIMPIO'] = df['CONSUMO M3 ACUEDUCTO_STR'].apply(limpiar_valor)
    
    # Para el promedio, reemplazar comas por puntos (es un decimal)
    df['PROMEDIO_LIMPIO'] = df['PROMEDIO CONSUMO ACUEDUCTO_STR'].str.replace(',', '.')
    
    # Mostrar ejemplos de valores limpios
    print("\nEjemplos de valores después de limpieza:")
    for i in range(min(10, len(df))):
        print(f"Fila {i+1}: Suscriptores: {df['SUSCRIPTORES_LIMPIO'].iloc[i]}, " +
              f"Consumo: {df['CONSUMO_LIMPIO'].iloc[i]}, " +
              f"Promedio: {df['PROMEDIO_LIMPIO'].iloc[i]}")
    
    # PASO 6: Convertir a numéricos para los cálculos finales
    df['SUSCRIPTORES_NUM'] = pd.to_numeric(df['SUSCRIPTORES_LIMPIO'], errors='coerce')
    df['CONSUMO_NUM'] = pd.to_numeric(df['CONSUMO_LIMPIO'], errors='coerce')
    df['PROMEDIO_NUM'] = pd.to_numeric(df['PROMEDIO_LIMPIO'], errors='coerce')
    
    # Eliminar registros con valores inválidos
    df = df.dropna(subset=['SUSCRIPTORES_NUM', 'CONSUMO_NUM', 'PROMEDIO_NUM'])
    df = df[df['SUSCRIPTORES_NUM'] > 0]  # Eliminar registros sin suscriptores
    
    print(f"Registros después de conversión numérica: {len(df)}")
    
    # PASO 7: Comprobar coherencia entre consumo y promedio
    df['PROMEDIO_CALCULADO'] = df['CONSUMO_NUM'] / df['SUSCRIPTORES_NUM']
    df['DIFF_ABSOLUTA'] = abs(df['PROMEDIO_CALCULADO'] - df['PROMEDIO_NUM'])
    
    # Mostrar sólo registros donde la diferencia es grande para diagnóstico
    grandes_diferencias = df['DIFF_ABSOLUTA'] > 1
    if grandes_diferencias.any():
        print("\nRegistros con diferencias grandes entre promedio calculado y reportado:")
        for i, (idx, row) in enumerate(df[grandes_diferencias].head(5).iterrows()):
            print(f"Fila {i+1}: Suscriptores: {row['SUSCRIPTORES_NUM']}, " +
                  f"Consumo: {row['CONSUMO_NUM']}, " +
                  f"Promedio reportado: {row['PROMEDIO_NUM']}, " +
                  f"Promedio calculado: {row['PROMEDIO_CALCULADO']:.2f}")
    
    # PASO 8: Construir el DataFrame final con las columnas correctas
    df_final = pd.DataFrame({
        'AÑO': df['AÑO'],
        'MES': df['MES'],
        'MUNICIPIO': df['MUNICIPIO'],
        'ESTRATO': df['ESTRATO_NUM'].astype(int),
        'No. SUSCRIPTORES ACUEDUCTO': df['SUSCRIPTORES_NUM'].astype(int),
        'CONSUMO M3 ACUEDUCTO': df['CONSUMO_NUM'].astype(int),
        'PROMEDIO CONSUMO ACUEDUCTO': df['PROMEDIO_NUM']
    })
    
    print(f"\nRegistros finales después de limpieza completa: {len(df_final)}")
    
    return df_final

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
    
    # Convertir AÑO a formato correcto si contiene comas o puntos
    if df['AÑO'].dtype == object:
        df['AÑO'] = df['AÑO'].astype(str).str.replace(',', '').str.replace('.', '')
        df['AÑO'] = pd.to_numeric(df['AÑO'], errors='coerce').astype(int)
    
    # Normalizar nombres de municipios (capitalizar)
    df['MUNICIPIO'] = df['MUNICIPIO'].str.title()
    
    print("Transformaciones completadas")
    return df

def generate_visualizations(df, output_path):
    """
    Genera visualizaciones para entender los datos
    """
    os.makedirs(output_path, exist_ok=True)
    
    # 1. Distribución de consumo por estrato
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='ESTRATO', y='PROMEDIO CONSUMO ACUEDUCTO', data=df)
    plt.title('Distribución del consumo promedio por estrato')
    plt.xlabel('Estrato socioeconómico')
    plt.ylabel('Consumo promedio (m³)')
    plt.savefig(os.path.join(output_path, 'consumo_por_estrato.png'))
    plt.close()  # Cerrar la figura para liberar memoria
    
    # 2. Tendencia temporal de consumo (usando AÑO y MES_NUM en lugar de FECHA)
    # Crear una columna temporal para ordenar
    df['PERIODO'] = df['AÑO'].astype(str) + '-' + df['MES_NUM'].astype(str).str.zfill(2)
    
    # Agrupar por período y estrato
    consumo_mensual = df.groupby(['PERIODO', 'ESTRATO'])['CONSUMO M3 ACUEDUCTO'].sum().reset_index()
    
    # Ordenar por período para asegurar la correcta visualización temporal
    consumo_mensual = consumo_mensual.sort_values('PERIODO')
    
    plt.figure(figsize=(12, 6))
    for estrato in sorted(df['ESTRATO'].unique()):
        datos_estrato = consumo_mensual[consumo_mensual['ESTRATO'] == estrato]
        plt.plot(datos_estrato['PERIODO'], datos_estrato['CONSUMO M3 ACUEDUCTO'], label=f'Estrato {estrato}')
    
    # Configurar el eje X para mostrar solo algunos períodos (para evitar sobrecarga)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Mostrar aproximadamente 10 etiquetas
    
    plt.title('Tendencia de consumo total por estrato a lo largo del tiempo')
    plt.xlabel('Período (Año-Mes)')
    plt.ylabel('Consumo total (m³)')
    plt.legend()
    plt.tight_layout()  # Ajustar layout para evitar recortes
    plt.savefig(os.path.join(output_path, 'tendencia_consumo.png'))
    plt.close()  # Cerrar la figura para liberar memoria
    
    # 3. Mapa de calor de consumo por municipio y estrato
    consumo_municipio_estrato = df.pivot_table(
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
    plt.tight_layout()  # Ajustar layout para evitar recortes
    plt.savefig(os.path.join(output_path, 'mapa_calor_consumo.png'))
    plt.close()  # Cerrar la figura para liberar memoria
    
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