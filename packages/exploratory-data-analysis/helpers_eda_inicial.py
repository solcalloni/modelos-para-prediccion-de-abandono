import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
import io
import tabulate

def dataset_profiling(df, name='df'):
    # Generar el perfil del dataset
    profile = ProfileReport(df, explorative=True)

    # Guardar el informe en HTML
    profile.to_file(f"dataset_profiling_{name}.html")

    print(f"El reporte ha sido generado: dataset_profiling_{name}.html")

def print_header(df_name="DataFrame", df=None):
    """Print the header section of the report."""
    print("="*80)
    print(f"DataFrame Report for {df_name}")
    print("="*80)
    print(f"\nThe DataFrame has {df.shape[0]} rows and {df.shape[1]} columns.")
    print("Below you will find detailed information about the DataFrame, including column data types,")
    print("summary statistics for both numerical and categorical data, missing values information,")
    print("and additional metrics that help understand the structure and quality of the dataset.\n")

def display_general_info(df):
    """Display general information about the DataFrame."""
    print("### General Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    print(info_str)

def display_numerical_statistics(df):
    """Display summary statistics for numerical columns."""
    print("### Summary Statistics (Numerical Columns)")
    print("The following table shows count, mean, standard deviation, min, and max values for each numerical column.\n")
    print(df.describe().to_markdown())

def display_all_column_statistics(df):
    """Display summary statistics for all columns."""
    print("\n### Summary Statistics (All Columns)")
    print("This table includes descriptive statistics for all columns, including categorical variables.\n")
    print(df.describe(include='all').to_markdown())

def display_missing_values(df):
    """Display information about missing values in each column."""
    print("\n### Missing Values")
    missing = df.isnull().sum().to_frame(name="Missing Values")
    missing["Missing Percentage (%)"] = (missing["Missing Values"] / len(df)) * 100
    print("The following table details the number and percentage of missing values per column.\n")
    print(missing.to_markdown())

def display_data_types(df):
    """Display data types for each column."""
    print("\n### Data Types")
    dtypes = df.dtypes.to_frame(name="Data Type")
    print("This table lists the data type for each column, which is helpful in determining appropriate data cleaning and preprocessing steps.\n")
    print(dtypes.to_markdown())

def display_unique_values(df):
    """Display count of unique values for each column."""
    print("\n### Unique Values per Column")
    unique = df.nunique().to_frame(name="Unique Values")
    print("The table below shows the count of unique values for each column. This is useful for understanding categorical diversity.\n")
    print(unique.to_markdown())

def display_duplicate_rows(df):
    """Display information about duplicate rows."""
    print("\n### Duplicate Rows")
    duplicate_count = df.duplicated().sum()
    print(f"The DataFrame contains {duplicate_count} duplicate rows.")
    if duplicate_count > 0:
        print("Consider removing duplicates to ensure data integrity.")
        print(df[df.duplicated(keep=False)])

# Configuración general para visualizaciones
plt.style.use('seaborn-v0_8')  # Estilo de gráficos
sns.set_palette("muted")

# === FUNCIÓN PARA ANÁLISIS DE DATOS FALTANTES ===
def analizar_datos_faltantes(df):
    """
    Analiza y visualiza los datos faltantes en el DataFrame.
    Imprime un resumen y genera un gráfico de barras para las columnas con valores nulos.
    """
    print("\n=== Análisis de Datos Faltantes ===")
    print("-" * 50)
    
    # Calcular número y porcentaje de valores faltantes
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Valores Faltantes': missing_data,
        'Porcentaje (%)': missing_percentage
    })
    missing_df = missing_df[missing_df['Valores Faltantes'] > 0].sort_values(by='Porcentaje (%)', ascending=False)
    
    # Resultados descriptivos
    if not missing_df.empty:
        print(f"Se encontraron datos faltantes en {len(missing_df)} columnas de un total de {df.shape[1]}.")
        print("\nResumen de columnas con valores faltantes:")
        print(missing_df)
        print("\nInterpretación:")
        for col, row in missing_df.iterrows():
            perc = row['Porcentaje (%)']
            if perc > 50:
                print(f"- '{col}': {perc:.2f}% de datos faltantes. Alta proporción, podría requerir eliminación o análisis específico.")
            elif perc > 10:
                print(f"- '{col}': {perc:.2f}% de datos faltantes. Considerar imputación o exclusión según el contexto.")
            else:
                print(f"- '{col}': {perc:.2f}% de datos faltantes. Impacto bajo, imputación viable.")
        
        # Visualización
        plt.figure(figsize=(12, 6))
        sns.barplot(x=missing_df.index, y=missing_df['Porcentaje (%)'])
        plt.xticks(rotation=45, ha='right')
        plt.title('Porcentaje de Datos Faltantes por Columna', fontsize=14)
        plt.xlabel('Columnas', fontsize=12)
        plt.ylabel('Porcentaje de Faltantes (%)', fontsize=12)
        plt.tight_layout()
        plt.show()
    else:
        print("No se encontraron datos faltantes en el DataFrame.")

# === FUNCIÓN PARA ANÁLISIS DE VALORES ATÍPICOS ===
def analizar_valores_atipicos(df, columnas_numericas):
    """
    Identifica y visualiza valores atípicos en columnas numéricas usando el método IQR.
    Imprime estadísticas y genera boxplots para cada variable.
    """
    print("\n=== Análisis de Valores Atípicos ===")
    print("-" * 50)
    print(f"Analizando {len(columnas_numericas)} columnas numéricas.")
    
    for col in columnas_numericas:
        print(f"\n--- Variable: {col} ---")
        
        # Calcular IQR y límites
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identificar outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        num_outliers = len(outliers)
        perc_outliers = (num_outliers / len(df)) * 100
        
        # Estadísticas descriptivas
        print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        print(f"Límite inferior: {lower_bound:.2f}, Límite superior: {upper_bound:.2f}")
        print(f"Número de valores atípicos: {num_outliers}")
        print(f"Porcentaje de valores atípicos: {perc_outliers:.2f}%")
        
        # Interpretación
        if perc_outliers > 10:
            print(f"Interpretación: Alta presencia de outliers ({perc_outliers:.2f}%). Revisar si son errores o patrones legítimos.")
        elif perc_outliers > 1:
            print(f"Interpretación: Moderada presencia de outliers ({perc_outliers:.2f}%). Considerar su impacto en modelos predictivos.")
        else:
            print(f"Interpretación: Baja presencia de outliers ({perc_outliers:.2f}%). Probablemente no afecten significativamente.")
        
        # Visualización con boxplot
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=df[col], color='lightblue')
        plt.title(f'Boxplot de {col}', fontsize=14)
        plt.xlabel(f'{col}', fontsize=12)
        plt.axvline(lower_bound, color='red', linestyle='--', label='Límite inferior')
        plt.axvline(upper_bound, color='red', linestyle='--', label='Límite superior')
        plt.legend()
        plt.tight_layout()
        plt.show()

# === FUNCIÓN PRINCIPAL PARA EJECUTAR EL ANÁLISIS ===
def missing_values_and_outliers(df):
    """
    Ejecuta el análisis completo de datos faltantes y valores atípicos.
    """
    print("=== Inicio del Análisis Exploratorio de Datos (EDA) ===")
    print(f"Dimensiones del DataFrame: {df.shape[0]} filas, {df.shape[1]} columnas")
    print("-" * 50)
    
    # 1. Análisis de datos faltantes
    analizar_datos_faltantes(df)
    
    # 2. Seleccionar columnas numéricas para análisis de valores atípicos
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nColumnas numéricas identificadas para análisis de outliers: {columnas_numericas}")
    analizar_valores_atipicos(df, columnas_numericas)
    
    print("\n=== Fin del Análisis ===")

def initial_eda(df, name='df'):
    """
    Realiza un análisis exploratorio inicial del DataFrame.
    Genera estadísticas descriptivas, información general y visualizaciones.
    """
    print_header(name, df)
    display_general_info(df)
    display_data_types(df)
    display_numerical_statistics(df)
    display_all_column_statistics(df)
    display_missing_values(df)
    display_unique_values(df)
    display_duplicate_rows(df)
    missing_values_and_outliers(df)
