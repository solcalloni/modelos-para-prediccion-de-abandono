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
        materias = [_normalizar(m) for plan in config["planes"] for m in plan["materias"]]

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

    # Agregar datos de personas
    resultado = actas.merge(
        personas[["dni", "año_inscripcion_facultad", "carrera_principal"]],
        on="dni",
        how="left",
    )

    resultado = (
        resultado[
            ["dni", "año_inscripcion_facultad", "porcentaje_materias_aprobadas", "carrera_principal"]
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

