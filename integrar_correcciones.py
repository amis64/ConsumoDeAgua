
import pandas as pd
import os

# Cargar dataset original
original_file = 'static/data/HISTORICO_CONSUMO_POR_ESTRATO_20250506.csv'
df_original = pd.read_csv(original_file, encoding='latin-1')

# Cargar correcciones manuales
corrected_file = 'static/data/registros_sospechosos_corregidos.csv'
df_corrected = pd.read_csv(corrected_file)

# Preparar identificadores únicos para matching
df_original['ID'] = df_original['AÑO'].astype(str) + '_' + df_original['MES'] + '_' + df_original['MUNICIPIO'] + '_' + df_original['ESTRATO']
df_corrected['ID'] = df_corrected['AÑO'].astype(str) + '_' + df_corrected['MES'] + '_' + df_corrected['MUNICIPIO'] + '_' + df_corrected['ESTRATO']

# Crear un diccionario de correcciones
corrections = {}
for idx, row in df_corrected.iterrows():
    corrections[row['ID']] = {
        'SUSCRIPTORES': row['SUSCRIPTORES_CORREGIDO'],
        'CONSUMO': row['CONSUMO_CORREGIDO']
    }

# Aplicar correcciones
modified_count = 0
for idx, row in df_original.iterrows():
    if row['ID'] in corrections:
        df_original.at[idx, 'No. SUSCRIPTORES ACUEDUCTO'] = corrections[row['ID']]['SUSCRIPTORES']
        df_original.at[idx, 'CONSUMO M3 ACUEDUCTO'] = corrections[row['ID']]['CONSUMO']
        modified_count += 1

print(f"Se aplicaron {modified_count} correcciones al dataset")

# Eliminar columna ID auxiliar
df_original = df_original.drop('ID', axis=1)

# Guardar dataset corregido
output_file = 'static/data/HISTORICO_CONSUMO_POR_ESTRATO_FINAL.csv'
df_original.to_csv(output_file, index=False)
print(f"Dataset corregido guardado en {output_file}")
