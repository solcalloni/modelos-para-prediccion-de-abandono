import unicodedata
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

_DEFAULT_PERSONAS = (
    Path(__file__).parent
    / "../../assets/bronze/FCEN/FCEN_oficial_2005_2025/reporte_personas_desde_2005.csv"
)
_DEFAULT_ACTAS = (
    Path(__file__).parent
    / "../../assets/bronze/FCEN/FCEN_oficial_2005_2025/reporte_actas_desde_2005.csv"
)


def _normalizar(texto: str) -> str:
    """Convierte a mayúsculas y elimina tildes."""
    return (
        unicodedata.normalize("NFKD", texto)
        .encode("ASCII", "ignore")
        .decode("ASCII")
        .upper()
        .strip()
    )


def get_porcentaje_aprobadas(
    carreras: List[str],
    anio: int,
    path_yaml: str,
    path_personas: Optional[str] = None,
    path_actas: Optional[str] = None,
    todos_los_planes: bool = True,
) -> pd.DataFrame:
    """Calcula el porcentaje de materias aprobadas por estudiante para el año dado.

    Para cada estudiante inscripto en alguna de las 'carreras' indicadas,
    considera únicamente aquellos cuya última materia aprobada del listado
    pertenezca al año 'anio'. Devuelve el porcentaje de materias aprobadas
    sobre el total del listado de materias del YAML.

    Parámetros
    ----------
    carreras : list[str]
        Valores a filtrar en la columna 'carrera_principal' de personas.
    anio : int
        Año al que debe pertenecer la última materia aprobada del estudiante.
    path_yaml : str
        Ruta al archivo YAML con el listado de materias (clave 'materias').
    path_personas : str | None
        Ruta a reporte_personas_desde_2005.csv. Si es None usa el default de FCEN.
    path_actas : str | None
        Ruta a reporte_actas_desde_2005.csv. Si es None usa el default de FCEN.
    todos_los_planes : bool
        Si True, usa todos los planes del YAML. Si False, usa solo el primero (default: True).

    Retorna
    -------
    pd.DataFrame
        Columnas: dni, año_inscripcion_facultad, porcentaje_materias_aprobadas,
        carrera_principal. Un registro por DNI.
    """
    path_personas = Path(path_personas) if path_personas else _DEFAULT_PERSONAS
    path_actas = Path(path_actas) if path_actas else _DEFAULT_ACTAS

    # --- Materias ---
    with open(path_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        planes = config["planes"] if todos_los_planes else config["planes"][:1]
        materias = [_normalizar(m) for plan in planes for m in plan["materias"]]

    # --- Personas ---
    personas = pd.read_csv(
        path_personas,
        usecols=["dni", "carrera_principal", "año_inscripcion_facultad"],
        dtype={"dni": str},
    )
    personas = personas[personas["carrera_principal"].isin(carreras)].copy()

    # --- Actas ---
    actas = pd.read_csv(path_actas, encoding="latin-1", dtype={"dni": str})
    actas["materia"] = actas["materia"].apply(lambda x: _normalizar(str(x)))
    actas["fecha"] = pd.to_datetime(actas["fecha"], format="%Y-%m-%d", errors="coerce")

    actas = actas[
        actas["dni"].isin(personas["dni"])
        & (actas["tipo_acta"] == "Acta de Examen")
        & (actas["resultado"] == "Aprobado")
        & (actas["materia"].isin(materias))
    ].copy()

    # Deduplicar: quedarse con el registro más reciente por (dni, materia)
    actas = actas.sort_values("fecha").drop_duplicates(
        subset=["dni", "materia"], keep="last"
    )

    # Filtrar: solo DNIs cuya última aprobación pertenezca al año indicado
    ultima_fecha_por_dni = actas.groupby("dni")["fecha"].max()
    dnis_anio = ultima_fecha_por_dni[ultima_fecha_por_dni.dt.year == anio].index
    actas = actas[actas["dni"].isin(dnis_anio)].copy()

    # Porcentaje de materias aprobadas
    actas["porcentaje_materias_aprobadas"] = (
        actas.groupby("dni")["materia"].transform("count") / len(materias)
    )

    # cantidad de materias aprobadas
    actas["cantidad_materias_aprobadas"] = (
        actas.groupby("dni")["materia"].transform("count")
    )    

    # Agregar datos de personas
    resultado = actas.merge(
        personas[["dni", "año_inscripcion_facultad", "carrera_principal"]],
        on="dni",
        how="left",
    )

    resultado = (
        resultado[
            ["dni", "año_inscripcion_facultad", "porcentaje_materias_aprobadas", "cantidad_materias_aprobadas", "carrera_principal"]
        ]
        .drop_duplicates(subset=["dni"])
        .reset_index(drop=True)
    )

    return resultado


def plot_porcentaje_aprobadas_por_anio(df: pd.DataFrame, anio: int) -> None:
    """Genera dos barplots por año de inscripción para estudiantes con el 90% o más de materias aprobadas:
    uno con la cantidad absoluta y otro con la proporción sobre el total con ≥90%.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame devuelto por get_porcentaje_aprobadas, con columnas
        'año_inscripcion_facultad', 'dni' y 'porcentaje_materias_aprobadas'.
    """
    resumen = (
        df[df["porcentaje_materias_aprobadas"] >= 0.9]
        .groupby("año_inscripcion_facultad")["dni"]
        .count()
        .reset_index()
        .rename(columns={"dni": "cantidad_estudiantes"})
        .sort_values("año_inscripcion_facultad")
    )
    resumen["proporcion"] = resumen["cantidad_estudiantes"] / resumen["cantidad_estudiantes"].sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    orden = resumen["año_inscripcion_facultad"]

    for ax, col, ylabel, title in [
        (ax1, "cantidad_estudiantes", "Cantidad de estudiantes", "Cantidad con ≥90% de materias aprobadas"),
        (ax2, "proporcion", "Proporción sobre el total con ≥90%", "Distribución por cohorte de estudiantes con ≥90% aprobadas"),
    ]:
        sns.barplot(data=resumen, x="año_inscripcion_facultad", y=col, order=orden, ax=ax)
        ax.set_xlabel("Año de inscripción a la facultad")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} en el año {anio}")
        ax.tick_params(axis="x", rotation=45)

    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.suptitle("Estudiantes con el 90% o más de materias aprobadas por cohorte")
    plt.tight_layout()
    plt.show()


def get_materias_fuera_de_carrera(
    df_porcentaje: pd.DataFrame,
    path_yaml: str,
    path_actas: Optional[str] = None,
    min_materias_aprobadas: int = 15,
    todos_los_planes: bool = True,
) -> pd.DataFrame:
    """Retorna registros de actas con materias fuera del listado de la carrera para estudiantes con muchas aprobadas.

    Filtra los DNIs del DataFrame de get_porcentaje_aprobadas con al menos
    'min_materias_aprobadas' materias aprobadas y devuelve los registros de
    actas de esos DNIs cuya materia NO se encuentre en el listado del YAML.

    Parámetros
    ----------
    df_porcentaje : pd.DataFrame
        DataFrame devuelto por get_porcentaje_aprobadas, con columnas
        'dni' y 'cantidad_materias_aprobadas'.
    path_yaml : str
        Ruta al archivo YAML con el listado de materias (clave 'materias').
    path_actas : str | None
        Ruta a reporte_actas_desde_2005.csv. Si es None usa el default de FCEN.
    min_materias_aprobadas : int
        Umbral mínimo de materias aprobadas para considerar un estudiante (default: 15).
    todos_los_planes : bool
        Si True, usa todos los planes del YAML. Si False, usa solo el primero (default: True).

    Retorna
    -------
    pd.DataFrame
        Columnas: dni, cantidad_optativas_aprobadas. Un registro por DNI con la
        cantidad de materias distintas aprobadas fuera del listado de la carrera,
        para los DNIs con al menos 'min_materias_aprobadas' materias aprobadas.
    """
    path_actas = Path(path_actas) if path_actas else _DEFAULT_ACTAS

    # DNIs con suficientes materias aprobadas
    dnis_filtrados = df_porcentaje.loc[
        df_porcentaje["cantidad_materias_aprobadas"] >= min_materias_aprobadas, "dni"
    ]

    # Materias de la carrera (normalizadas)
    with open(path_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        planes = config["planes"] if todos_los_planes else config["planes"][:1]
        materias = {_normalizar(m) for plan in planes for m in plan["materias"]}

    # Actas filtradas
    actas = pd.read_csv(path_actas, encoding="latin-1", dtype={"dni": str})
    actas["materia"] = actas["materia"].apply(lambda x: _normalizar(str(x)))
    actas["fecha"] = pd.to_datetime(actas["fecha"], format="%Y-%m-%d", errors="coerce")

    actas_filtradas = actas[
        actas["dni"].isin(dnis_filtrados)
        & (actas["tipo_acta"] == "Acta de Examen")
        & (actas["resultado"] == "Aprobado")
        & (~actas["materia"].isin(materias))
    ].copy()

    resultado = (
        actas_filtradas.groupby("dni")["materia"]
        .nunique()
        .reset_index()
        .rename(columns={"materia": "cantidad_optativas_aprobadas"})
    )

    return resultado.reset_index(drop=True)


def plot_optativas_por_anio(
    df_optativas: pd.DataFrame,
    df_porcentaje: pd.DataFrame,
    min_optativas: int = 3,
    anio: int = 2014,
) -> None:
    """Barplot de la cantidad de estudiantes con al menos 'min_optativas' materias optativas aprobadas por cohorte.
    A su vez, ya sabemos que si tienen calculo de optativas, es porque aprobaron al menos 'min_materias_aprobadas' materias obligatorias, por lo que este gráfico se enfoca en los estudiantes que terminaron la carrera.

    Parámetros
    ----------
    df_optativas : pd.DataFrame
        DataFrame devuelto por get_materias_fuera_de_carrera, con columnas
        'dni' y 'cantidad_optativas_aprobadas'.
    df_porcentaje : pd.DataFrame
        DataFrame devuelto por get_porcentaje_aprobadas, con columnas
        'dni' y 'año_inscripcion_facultad'.
    min_optativas : int
        Umbral mínimo de materias optativas aprobadas (default: 3).
    """
    df = df_optativas.merge(
        df_porcentaje[["dni", "año_inscripcion_facultad"]],
        on="dni",
        how="left",
    )

    resumen = (
        df[df["cantidad_optativas_aprobadas"] >= min_optativas]
        .groupby("año_inscripcion_facultad")["dni"]
        .count()
        .reset_index()
        .rename(columns={"dni": "cantidad_estudiantes"})
        .sort_values("año_inscripcion_facultad")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=resumen, x="año_inscripcion_facultad", y="cantidad_estudiantes", order=resumen["año_inscripcion_facultad"], ax=ax)
    ax.set_xlabel("Año de inscripción a la facultad")
    ax.set_ylabel("Cantidad de estudiantes")
    ax.set_title(f"Cantidad de estudiantes egresados en el año {anio}")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()


def get_egresados(
    carreras: List[str],
    anio: int,
    path_yaml: str,
    path_personas: Optional[str] = None,
    path_actas: Optional[str] = None,
    min_materias_obligatorias: int = 15,
    min_materias_optativas: int = 3,
    todos_los_planes: bool = True,
) -> pd.DataFrame:
    """Identifica estudiantes que terminaron la carrera según dos criterios.

    Condición 1: el DNI tiene "TESIS DE LICENCIATURA" aprobada en el año indicado.
    Condición 2: el DNI no tiene "TESIS DE LICENCIATURA" pero aprobó al menos
    'min_materias_obligatorias' materias del listado Y al menos 'min_materias_optativas'
    materias fuera del listado.

    Parámetros
    ----------
    carreras : list[str]
        Valores a filtrar en la columna 'carrera_principal' de personas.
    anio : int
        Año al que debe pertenecer la aprobación de la tesis (condición 1).
    path_yaml : str
        Ruta al archivo YAML con el listado de materias (clave 'materias').
    path_personas : str | None
        Ruta a reporte_personas_desde_2005.csv. Si es None usa el default de FCEN.
    path_actas : str | None
        Ruta a reporte_actas_desde_2005.csv. Si es None usa el default de FCEN.
    min_materias_obligatorias : int
        Mínimo de materias del plan aprobadas para condición 2 (default: 15).
    min_materias_optativas : int
        Mínimo de materias fuera del plan aprobadas para condición 2 (default: 3).
    todos_los_planes : bool
        Si True, usa todos los planes del YAML. Si False, usa solo el primero (default: True).

    Retorna
    -------
    pd.DataFrame
        Columnas: dni, año_inscripcion_facultad, carrera_principal.
        Un registro por DNI que cumple condición 1 o condición 2.
    """
    path_personas = Path(path_personas) if path_personas else _DEFAULT_PERSONAS
    path_actas = Path(path_actas) if path_actas else _DEFAULT_ACTAS

    # --- Materias del plan (normalizadas) ---
    with open(path_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        planes = config["planes"] if todos_los_planes else config["planes"][:1]
        materias = {_normalizar(m) for plan in planes for m in plan["materias"]}

    # --- Personas ---
    personas = pd.read_csv(
        path_personas,
        usecols=["dni", "carrera_principal", "año_inscripcion_facultad"],
        dtype={"dni": str},
    )
    personas = personas[personas["carrera_principal"].isin(carreras)].copy()

    # --- Actas: aprobadas en Acta de Examen para los DNIs de personas ---
    actas_raw = pd.read_csv(path_actas, encoding="latin-1", dtype={"dni": str})
    actas_raw["materia"] = actas_raw["materia"].apply(lambda x: _normalizar(str(x)))
    actas_raw["fecha"] = pd.to_datetime(actas_raw["fecha"], format="%Y-%m-%d", errors="coerce")

    actas_raw = actas_raw[
        actas_raw["dni"].isin(personas["dni"])
        & (actas_raw["tipo_acta"] == "Acta de Examen")
        & (actas_raw["resultado"] == "Aprobado")
    ].copy()

    # Deduplicar por (dni, materia): quedarse con el registro más reciente
    actas_raw = actas_raw.sort_values("fecha").drop_duplicates(
        subset=["dni", "materia"], keep="last"
    )

    # Subconjunto dentro del plan
    actas_en_listado = actas_raw[actas_raw["materia"].isin(materias)].copy()

    # --- Condición 1: TESIS DE LICENCIATURA aprobada en el año indicado ---
    TESIS = "TESIS DE LICENCIATURA"
    dnis_cond1 = set(
        actas_en_listado.loc[
            (actas_en_listado["materia"] == TESIS)
            & (actas_en_listado["fecha"].dt.year == anio),
            "dni",
        ].unique()
    )

    # --- Condición 2: sin tesis, ≥min_materias_obligatorias del plan, ≥min_materias_optativas fuera
    #                  y última materia obligatoria aprobada en el año indicado ---
    dnis_con_tesis = set(actas_en_listado.loc[actas_en_listado["materia"] == TESIS, "dni"].unique())

    actas_obligatorias_sin_tesis = actas_en_listado[~actas_en_listado["dni"].isin(dnis_con_tesis)]

    # Filtrar: solo DNIs cuya última obligatoria aprobada pertenezca al año indicado
    ultima_obligatoria_por_dni = actas_obligatorias_sin_tesis.groupby("dni")["fecha"].max()
    dnis_ultima_en_anio = set(ultima_obligatoria_por_dni[ultima_obligatoria_por_dni.dt.year == anio].index)

    # Materias del plan por DNI (excluidos los que tienen tesis y cuya última obligatoria es del año)
    conteo_obligatorias = (
        actas_obligatorias_sin_tesis[actas_obligatorias_sin_tesis["dni"].isin(dnis_ultima_en_anio)]
        .groupby("dni")["materia"]
        .count()
    )
    dnis_suficientes_obligatorias = set(
        conteo_obligatorias[conteo_obligatorias >= min_materias_obligatorias].index
    )

    # Materias fuera del plan para esos DNIs
    conteo_optativas = (
        actas_raw[
            ~actas_raw["materia"].isin(materias)
            & actas_raw["dni"].isin(dnis_suficientes_obligatorias)
        ]
        .groupby("dni")["materia"]
        .count()
    )
    dnis_suficientes_optativas = set(
        conteo_optativas[conteo_optativas >= min_materias_optativas].index
    )

    dnis_cond2 = dnis_suficientes_obligatorias & dnis_suficientes_optativas

    # --- Resultado ---
    dnis_egresados = dnis_cond1 | dnis_cond2

    return (
        personas[personas["dni"].isin(dnis_egresados)][
            ["dni", "año_inscripcion_facultad", "carrera_principal"]
        ]
        .drop_duplicates(subset=["dni"])
        .reset_index(drop=True)
    )


def plot_egresados_por_anio(df: pd.DataFrame, anio: int) -> None:
    """Genera dos barplots de egresados por año de inscripción: cantidad absoluta y proporción.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame devuelto por get_egresados, con columnas 'dni' y
        'año_inscripcion_facultad'.
    """
    resumen = (
        df.groupby("año_inscripcion_facultad")["dni"]
        .count()
        .reset_index()
        .rename(columns={"dni": "cantidad_egresados"})
        .sort_values("año_inscripcion_facultad")
    )
    resumen["proporcion"] = resumen["cantidad_egresados"] / resumen["cantidad_egresados"].sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    orden = resumen["año_inscripcion_facultad"]

    for ax, col, ylabel, title in [
        (ax1, "cantidad_egresados", "Cantidad de egresados", f"Cantidad de egresados de {anio} por cohorte"),
        (ax2, "proporcion", "Proporción sobre el total de egresados", f"Distribución de egresados de {anio} por cohorte"),
    ]:
        sns.barplot(data=resumen, x="año_inscripcion_facultad", y=col, order=orden, ax=ax)
        ax.set_xlabel("Año de inscripción a la facultad")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)

    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.suptitle("Egresados por cohorte de ingreso")
    plt.tight_layout()
    plt.show()


def get_anio_mayor_proporcion(df: pd.DataFrame) -> pd.Series:
    """Devuelve el año de inscripción con la mayor proporción de estudiantes con ≥90% de materias aprobadas.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame devuelto por get_porcentaje_aprobadas, con columnas
        'año_inscripcion_facultad', 'dni' y 'porcentaje_materias_aprobadas'.

    Retorna
    -------
    pd.Series
        Fila con 'año_inscripcion_facultad', 'cantidad_estudiantes' y 'proporcion'.
    """
    total_por_anio = df.groupby("año_inscripcion_facultad")["dni"].count().rename("total")

    aprobados_por_anio = (
        df[df["porcentaje_materias_aprobadas"] >= 0.9]
        .groupby("año_inscripcion_facultad")["dni"]
        .count()
        .rename("cantidad_estudiantes")
    )

    resumen = pd.concat([total_por_anio, aprobados_por_anio], axis=1).fillna({"cantidad_estudiantes": 0})
    resumen["proporcion"] = resumen["cantidad_estudiantes"] / resumen["cantidad_estudiantes"].sum()

    return resumen["proporcion"].idxmax(), resumen["proporcion"].max()

