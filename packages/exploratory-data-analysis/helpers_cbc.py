import numpy as np
import pandas as pd

def normalize_column_values(df, cols_to_normalize):
    """Normalize values in specified columns of the DataFrame."""
    df[cols_to_normalize] = df[cols_to_normalize].apply(lambda col: col.astype(str).str.strip().str.upper().replace('NAN', np.nan) if col.dtype == 'object' else col)
    return df

def delete_duplicates(df):
    """Delete duplicate rows from the DataFrame."""
    initial_shape = df.shape
    df = df.drop_duplicates(keep='first')
    df.reset_index(drop=True, inplace=True)
    final_shape = df.shape
    print(f"Deleted {initial_shape[0] - final_shape[0]} duplicate rows.")
    return df

def replace_zero_with_nan(df, cols_to_replace):
    df[cols_to_replace] = df[cols_to_replace].replace('0', np.nan)
    
    return df


def get_data(local_path):
    df_calificaciones = pd.read_excel(local_path, sheet_name='Calificaciones')
    df_carreras = pd.read_excel(local_path, sheet_name='Materias Grilla Carreras')

    # Normalize column values
    df_carreras = normalize_column_values(df_carreras, ['Carrera', 'Materia'])
    df_calificaciones = normalize_column_values(df_calificaciones, ['Carrera', 'Dirección', 'Localidad', 'dominio email', 'Materia', 'Nota', 'UBA XXI', 'Es materia FCEN?'])
    df_calificaciones = replace_zero_with_nan(df_calificaciones, ['Localidad', 'Dirección'])

    # Caso especial que encontramos
    df_calificaciones['Localidad'] = df_calificaciones['Localidad'].replace('NO RESIDENTE', np.nan)

    # Delete duplicates
    df_calificaciones = delete_duplicates(df_calificaciones)
    return df_calificaciones, df_carreras
