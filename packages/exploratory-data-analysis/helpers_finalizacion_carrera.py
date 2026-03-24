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

_DEFAULT_DATOS_AGRUPADOS = (Path(__file__).parent / "../../assets/datos_agrupados")


def _normalizar(texto: str) -> str:
    """Convierte a mayúsculas y elimina tildes."""
    return (
        unicodedata.normalize("NFKD", texto)
        .encode("ASCII", "ignore")
        .decode("ASCII")
        .upper()
        .strip()
    )


def get_egresados_computacion(
    carreras: List[str],
    path_yaml: str,
    path_personas: Optional[str] = None,
    path_actas: Optional[str] = None,
    min_materias_obligatorias: int = 15,
    min_materias_optativas: int = 3,
    todos_los_planes: bool = True,
) -> pd.DataFrame:
    """Identifica estudiantes que terminaron la carrera según dos criterios.

    Condición 1: el DNI tiene "TESIS DE LICENCIATURA" aprobada.
        → anio_egreso = año de aprobación de la tesis.
    Condición 2: el DNI no tiene "TESIS DE LICENCIATURA" pero aprobó al menos
    'min_materias_obligatorias' materias del listado Y al menos 'min_materias_optativas'
    materias fuera del listado.
        → anio_egreso = año de la última materia obligatoria aprobada.

    Parámetros
    ----------
    carreras : list[str]
        Valores a filtrar en la columna 'carrera_principal' de personas.
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
        Columnas: dni, año_inscripcion_facultad, carrera_principal, anio_egreso.
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

    # --- Condición 1: TESIS DE LICENCIATURA aprobada ---
    TESIS = "TESIS DE LICENCIATURA"
    actas_tesis = actas_en_listado[actas_en_listado["materia"] == TESIS][["dni", "fecha"]].copy()
    actas_tesis["anio_egreso"] = actas_tesis["fecha"].dt.year

    dnis_con_tesis = set(actas_tesis["dni"].unique())

    # --- Condición 2: sin tesis, ≥min_materias_obligatorias del plan, ≥min_materias_optativas fuera ---
    actas_obligatorias_sin_tesis = actas_en_listado[~actas_en_listado["dni"].isin(dnis_con_tesis)]

    conteo_obligatorias = actas_obligatorias_sin_tesis.groupby("dni")["materia"].count()
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

    # anio_egreso cond2: año de la última materia obligatoria aprobada
    egresados_cond2 = (
        actas_obligatorias_sin_tesis[actas_obligatorias_sin_tesis["dni"].isin(dnis_cond2)]
        .groupby("dni")["fecha"]
        .max()
        .dt.year
        .reset_index()
        .rename(columns={"fecha": "anio_egreso"})
    )

    # --- Resultado ---
    egresados = pd.concat(
        [actas_tesis[["dni", "anio_egreso"]], egresados_cond2],
        ignore_index=True,
    )

    resultado = egresados.merge(
        personas[["dni", "año_inscripcion_facultad", "carrera_principal"]],
        on="dni",
        how="left",
    )

    return (
        resultado[["dni", "año_inscripcion_facultad", "carrera_principal", "anio_egreso"]]
        .drop_duplicates(subset=["dni"])
        .reset_index(drop=True)
    )

# Materias troncales (obligatorias) de la Licenciatura en Biología, unión de ambos planes.
# Materias troncales (obligatorias) de la Licenciatura en Ciencias Biológicas.
# Lista unificada usada para ambos planes (1984 y 2019).
_MATERIAS_OBLIGATORIAS_BIOLOGIA = [
    "Biometría",
    "Ecología General",
    "Fisica II",
    "Genética I",
    "Introducción a la Biología Molecular y Celular",
    "Introducción a la Botánica",
    "Introducción a la Zoología",
    "Análisis Matemático I",
    "Elementos de Cálculo Numérico",
    "Física I",
    "Química Biológica",
    "Química General e Inorgánica I",
    "Química Orgánica",
    "Electromagnetismo y Óptica",
    "Evolución",
    "Genética",
    "Matemática I",
    "Matemática II",
    "Mecánica y Termodinámica",
]


def get_egresados_biologia(
    carreras: List[str],
    path_yaml: str,
    path_personas: Optional[str] = None,
    path_actas: Optional[str] = None,
    min_obligatorias_1984: int = 13,
    min_ciclo_superior_1984: int = 8,
    min_obligatorias_2019: int = 14,
    min_ciclo_superior_2019: int = 6,
) -> pd.DataFrame:
    """Identifica egresados de la Licenciatura en Biología con su año de egreso.

    Condición 1: el DNI tiene "TESIS DE LICENCIATURA" aprobada.
        → anio_egreso = año de aprobación de la tesis.

    Condición 2: el DNI no tiene tesis. El plan se determina por "EVOLUCION":
      - Plan 2019: aprobó "EVOLUCION" en 2019 o posterior.
          Debe tener ≥min_obligatorias_2019 materias del plan 2019 Y
          ≥min_ciclo_superior_2019 materias del ciclo superior del plan 2019.
      - Plan 1984: no cumple la condición anterior.
          Debe tener ≥min_obligatorias_1984 materias del plan 1984 Y
          ≥min_ciclo_superior_1984 materias del ciclo superior del plan 1984.
      → anio_egreso = año de la última materia aprobada entre las
        obligatorias y ciclo superior consideradas.

    Parámetros
    ----------
    carreras : list[str]
        Valores a filtrar en la columna 'carrera_principal' de personas.
    path_yaml : str
        Ruta al archivo YAML con el listado de materias por plan.
    path_personas : str | None
        Ruta a reporte_personas_desde_2005.csv. Si es None usa el default de FCEN.
    path_actas : str | None
        Ruta a reporte_actas_desde_2005.csv. Si es None usa el default de FCEN.
    min_obligatorias_1984 : int
        Mínimo de materias troncales del plan 1984 (default: 13).
    min_ciclo_superior_1984 : int
        Mínimo de materias del ciclo superior del plan 1984 (default: 8).
    min_obligatorias_2019 : int
        Mínimo de materias troncales del plan 2019 (default: 14).
    min_ciclo_superior_2019 : int
        Mínimo de materias del ciclo superior del plan 2019 (default: 6).

    Retorna
    -------
    pd.DataFrame
        Columnas: dni, año_inscripcion_facultad, carrera_principal, anio_egreso.
        Un registro por DNI que cumple condición 1 o condición 2.
    """
    path_personas = Path(path_personas) if path_personas else _DEFAULT_PERSONAS
    path_actas = Path(path_actas) if path_actas else _DEFAULT_ACTAS

    # --- Materias obligatorias compartidas (normalizadas) ---
    obligatorias = {_normalizar(m) for m in _MATERIAS_OBLIGATORIAS_BIOLOGIA}
    # le saco a la lista obligatorias_1984 a la materia "Evolucion" porque en el plan 1984 no es obligatoria, sino del ciclo superior
    obligatorias_1984 = obligatorias - {_normalizar("Evolución")}
    obligatorias_2019 = obligatorias

    # --- Ciclo superior por plan: materias del YAML que no son obligatorias ---
    with open(path_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        materias_plan_1984 = {_normalizar(m) for m in config["planes"][0]["materias"]}
        materias_plan_2019 = {_normalizar(m) for m in config["planes"][1]["materias"]}

    TESIS = _normalizar("TESIS DE LICENCIATURA")
    EVOLUCION = _normalizar("Evolución")

    cs_1984 = materias_plan_1984 - obligatorias_1984
    cs_2019 = materias_plan_2019 - obligatorias_2019

    # --- Personas ---
    personas = pd.read_csv(
        path_personas,
        usecols=["dni", "carrera_principal", "año_inscripcion_facultad"],
        dtype={"dni": str},
    )
    personas = personas[personas["carrera_principal"].isin(carreras)].copy()

    # --- Actas: Acta de Examen + Aprobado para los DNIs relevantes ---
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

    # -----------------------------------------------------------------------
    # Condición 1: tiene TESIS DE LICENCIATURA aprobada
    # -----------------------------------------------------------------------
    actas_tesis = actas_raw[actas_raw["materia"] == TESIS][["dni", "fecha"]].copy()
    actas_tesis = actas_tesis.rename(columns={"fecha": "fecha_egreso"})
    actas_tesis["anio_egreso"] = actas_tesis["fecha_egreso"].dt.year

    dnis_con_tesis = set(actas_tesis["dni"].unique())

    # -----------------------------------------------------------------------
    # Condición 2: sin tesis — detectar plan y aplicar umbrales correspondientes
    # -----------------------------------------------------------------------
    actas_sin_tesis = actas_raw[~actas_raw["dni"].isin(dnis_con_tesis)].copy()

    # DNIs del plan 2019: aprobaron EVOLUCION en 2019 o posterior
    dnis_plan_2019 = set(
        actas_sin_tesis.loc[
            (actas_sin_tesis["materia"] == EVOLUCION)
            & (actas_sin_tesis["fecha"].dt.year >= 2019),
            "dni",
        ].unique()
    )
    dnis_plan_1984 = set(actas_sin_tesis["dni"].unique()) - dnis_plan_2019

    def _egresados_por_plan(dnis, obligatorias, ciclo_superior, min_oblig, min_cs):
        """Devuelve DNIs que cumplen los umbrales y su año de egreso."""
        actas = actas_sin_tesis[actas_sin_tesis["dni"].isin(dnis)]

        conteo_oblig = actas[actas["materia"].isin(obligatorias)].groupby("dni")["materia"].count()
        dnis_oblig_ok = set(conteo_oblig[conteo_oblig >= min_oblig].index)

        conteo_cs = (
            actas[actas["materia"].isin(ciclo_superior) & actas["dni"].isin(dnis_oblig_ok)]
            .groupby("dni")["materia"]
            .count()
        )
        dnis_cs_ok = set(conteo_cs[conteo_cs >= min_cs].index)

        dnis_egresados = dnis_oblig_ok & dnis_cs_ok

        actas_relevantes = actas[
            actas["dni"].isin(dnis_egresados)
            & (actas["materia"].isin(obligatorias) | actas["materia"].isin(ciclo_superior))
        ]
        return (
            actas_relevantes.groupby("dni")["fecha"]
            .max()
            .dt.year
            .reset_index()
            .rename(columns={"fecha": "anio_egreso"})
        )

    egresados_1984 = _egresados_por_plan(
        dnis_plan_1984, obligatorias_1984, cs_1984, min_obligatorias_1984, min_ciclo_superior_1984
    )
    egresados_2019 = _egresados_por_plan(
        dnis_plan_2019, obligatorias_2019, cs_2019, min_obligatorias_2019, min_ciclo_superior_2019
    )

    # -----------------------------------------------------------------------
    # Resultado final
    # -----------------------------------------------------------------------
    egresados = pd.concat(
        [actas_tesis[["dni", "anio_egreso"]], egresados_1984, egresados_2019],
        ignore_index=True,
    )

    resultado = egresados.merge(
        personas[["dni", "año_inscripcion_facultad", "carrera_principal"]],
        on="dni",
        how="left",
    )

    return (
        resultado[["dni", "año_inscripcion_facultad", "carrera_principal", "anio_egreso"]]
        .drop_duplicates(subset=["dni"])
        .reset_index(drop=True)
    )


def get_egresados_fisica(
    carreras: List[str],
    path_yaml: str,
    path_personas: Optional[str] = None,
    path_actas: Optional[str] = None,
    min_materias_sin_tesis: int = 22,
    min_materias_fuera_de_plan: int = 2,
) -> pd.DataFrame:
    """Identifica egresados de la Licenciatura en Física con su año de egreso.

    Condición 1: el DNI tiene "TESIS DE LICENCIATURA" aprobada.
        → anio_egreso = año de aprobación de la tesis.

    Condición 2: el DNI no tiene "TESIS DE LICENCIATURA" pero aprobó al menos
    'min_materias_sin_tesis' de las 24 materias restantes del plan Y al menos
    'min_materias_fuera_de_plan' materias fuera del temario de física.
        → anio_egreso = año de la última materia obligatoria aprobada del listado (sin tesis).

    El umbral por defecto (22) equivale al 90% de las 24 materias sin tesis.

    Parámetros
    ----------
    carreras : list[str]
        Valores a filtrar en la columna 'carrera_principal' de personas.
    path_yaml : str
        Ruta al archivo YAML con el listado de materias (clave 'planes').
    path_personas : str | None
        Ruta a reporte_personas_desde_2005.csv. Si es None usa el default de FCEN.
    path_actas : str | None
        Ruta a reporte_actas_desde_2005.csv. Si es None usa el default de FCEN.
    min_materias_sin_tesis : int
        Mínimo de materias del plan (sin tesis) aprobadas para condición 2 (default: 22).
    min_materias_fuera_de_plan : int
        Mínimo de materias fuera del temario aprobadas para condición 2 (default: 2).

    Retorna
    -------
    pd.DataFrame
        Columnas: dni, año_inscripcion_facultad, carrera_principal, anio_egreso.
        Un registro por DNI que cumple condición 1 o condición 2.
    """
    path_personas = Path(path_personas) if path_personas else _DEFAULT_PERSONAS
    path_actas = Path(path_actas) if path_actas else _DEFAULT_ACTAS

    TESIS = _normalizar("Tesis de Licenciatura")

    # --- Materias del plan (normalizadas), separando tesis ---
    with open(path_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        todas_las_materias = {
            _normalizar(m) for plan in config["planes"] for m in plan["materias"]
        }
    materias_sin_tesis = todas_las_materias - {TESIS}

    # --- Personas ---
    personas = pd.read_csv(
        path_personas,
        usecols=["dni", "carrera_principal", "año_inscripcion_facultad"],
        dtype={"dni": str},
    )
    personas = personas[personas["carrera_principal"].isin(carreras)].copy()

    # --- Actas: Acta de Examen + Aprobado para los DNIs relevantes ---
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

    # -----------------------------------------------------------------------
    # Condición 1: tiene TESIS DE LICENCIATURA aprobada
    # -----------------------------------------------------------------------
    actas_tesis = actas_raw[actas_raw["materia"] == TESIS][["dni", "fecha"]].copy()
    actas_tesis["anio_egreso"] = actas_tesis["fecha"].dt.year

    dnis_con_tesis = set(actas_tesis["dni"].unique())

    # -----------------------------------------------------------------------
    # Condición 2: sin tesis — al menos min_materias_sin_tesis del plan aprobadas
    # -----------------------------------------------------------------------
    actas_sin_tesis = actas_raw[
        ~actas_raw["dni"].isin(dnis_con_tesis)
        & actas_raw["materia"].isin(materias_sin_tesis)
    ].copy()

    conteo = actas_sin_tesis.groupby("dni")["materia"].count()
    dnis_cond2 = set(conteo[conteo >= min_materias_sin_tesis].index)

    # Materias fuera del plan para esos DNIs
    conteo_fuera = (
        actas_raw[
            ~actas_raw["materia"].isin(todas_las_materias)
            & actas_raw["dni"].isin(dnis_cond2)
        ]
        .groupby("dni")["materia"]
        .count()
    )
    dnis_cond2 = dnis_cond2 & set(conteo_fuera[conteo_fuera >= min_materias_fuera_de_plan].index)

    egresados_cond2 = (
        actas_sin_tesis[actas_sin_tesis["dni"].isin(dnis_cond2)]
        .groupby("dni")["fecha"]
        .max()
        .dt.year
        .reset_index()
        .rename(columns={"fecha": "anio_egreso"})
    )

    # -----------------------------------------------------------------------
    # Resultado final
    # -----------------------------------------------------------------------
    egresados = pd.concat(
        [actas_tesis[["dni", "anio_egreso"]], egresados_cond2],
        ignore_index=True,
    )

    resultado = egresados.merge(
        personas[["dni", "año_inscripcion_facultad", "carrera_principal"]],
        on="dni",
        how="left",
    )

    return (
        resultado[["dni", "año_inscripcion_facultad", "carrera_principal", "anio_egreso"]]
        .drop_duplicates(subset=["dni"])
        .reset_index(drop=True)
    )

def resumir_egresados_por_anio(df: pd.DataFrame, anio: int) -> pd.DataFrame:
    """Agrupa egresados por cohorte de ingreso y calcula cantidad absoluta y proporción.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas 'dni' y 'año_inscripcion_facultad'.
    anio : int
        Año de egreso al que corresponden los datos.

    Retorna
    -------
    pd.DataFrame
        Columnas: anio_egreso, año_inscripcion_facultad, cantidad_egresados, proporcion.
    """
    resumen = (
        df.groupby("año_inscripcion_facultad")["dni"]
        .count()
        .reset_index()
        .rename(columns={"dni": "cantidad_egresados"})
        .sort_values("año_inscripcion_facultad")
    )
    resumen["proporcion"] = resumen["cantidad_egresados"] / resumen["cantidad_egresados"].sum()
    resumen["anio_egreso"] = anio
    return resumen.reset_index(drop=True)


def plot_egresados_por_anio(resumen: pd.DataFrame, anio: int, carrera: str) -> None:
    """Genera dos barplots de egresados por año de inscripción: cantidad absoluta y proporción.

    Parámetros
    ----------
    resumen : pd.DataFrame
        DataFrame devuelto por resumir_egresados_por_anio, con columnas
        'año_inscripcion_facultad', 'cantidad_egresados' y 'proporcion'.
    anio : int
        Año de egreso que aparece en los títulos de los gráficos.
    carrera : str
        Nombre de la carrera para el título del gráfico.
    """
    sns.set_theme(style="darkgrid")
    orden = resumen["año_inscripcion_facultad"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

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
    plt.suptitle(f"Egresados por cohorte de ingreso: {carrera}")
    plt.tight_layout()
    plt.show()

def plot_todos_anios_egreso(df: pd.DataFrame, carrera: str) -> None:
    """Genera un par de barplots por cada año de egreso con la distribución de cohortes.

    Delega en plot_egresados_por_anio para cada valor distinto de 'anio_egreso'
    presente en el DataFrame.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame devuelto por get_egresados_fisica, con columnas
        'dni', 'año_inscripcion_facultad' y 'anio_egreso'.
    """
    carrera = _normalizar(carrera)
    carrera = carrera.lower()
    for anio, grupo in df.groupby("anio_egreso"):
        resumen = resumir_egresados_por_anio(grupo, anio)
        resumen.to_csv(f"{_DEFAULT_DATOS_AGRUPADOS}/{carrera}/resumen_egresados_{carrera}_{anio}.csv", index=False)
        plot_egresados_por_anio(resumen, anio, carrera)


def plot_egresados_por_anio_cantidad_y_k(resumen: pd.DataFrame, anio: int, carrera: str) -> None:
    """Genera un barplot de egresados por años de cursada.

    Parámetros
    ----------
    resumen : pd.DataFrame
        DataFrame devuelto por resumir_egresados_por_anio, con columnas
        'año_inscripcion_facultad', 'cantidad_egresados' y 'proporcion'.
    anio : int
        Año de egreso que aparece en los títulos de los gráficos.
    carrera : str
        Nombre de la carrera para el título del gráfico.
    """
    sns.set_theme(style="darkgrid")
    resumen_plot = resumen.copy()
    resumen_plot["anios_cursada"] = anio - resumen_plot["año_inscripcion_facultad"] + 1
    resumen_plot = resumen_plot.sort_values("anios_cursada").reset_index(drop=True)

    orden = resumen_plot["anios_cursada"]
    cantidades = resumen_plot["cantidad_egresados"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(data=resumen_plot, x="anios_cursada", y="cantidad_egresados", order=orden, ax=ax)
    ax.set_xlabel("Años de cursada", fontsize=15)
    ax.set_ylabel("Cantidad de egresados", fontsize=15)
    ax.set_title(f"Cantidad de egresados en {anio} por años de cursada. Carrera: {carrera}", fontsize=15)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13)

    idx_max = resumen_plot["cantidad_egresados"].idxmax()
    cant_max = resumen_plot.loc[idx_max, "cantidad_egresados"]
    k = resumen_plot.loc[idx_max, "anios_cursada"]

    # Selecciona el tramo consecutivo que contiene la barra de k y se aproxima al 90%.
    pos_k = idx_max
    total = sum(cantidades)
    objetivo = 0.9 * total
    n_barras = len(cantidades)
    k_en_borde = pos_k == 0 or pos_k == n_barras - 1

    mejor_tramo = None
    for izq in range(0, pos_k + 1):
        for der in range(pos_k, n_barras):
            if not k_en_borde and not (izq < pos_k and der > pos_k):
                continue
            suma_tramo = sum(cantidades[izq : der + 1])
            distancia = abs(suma_tramo - objetivo)
            usa_ambos_lados = izq < pos_k and der > pos_k
            criterio = (distancia, 0 if usa_ambos_lados else 1, der - izq)
            if mejor_tramo is None or criterio < mejor_tramo[0]:
                mejor_tramo = (criterio, izq, der)

    if mejor_tramo is not None:
        print(f"Mejor tramo para k={k}: {mejor_tramo[1]} a {mejor_tramo[2]}, suma={sum(cantidades[mejor_tramo[1] : mejor_tramo[2] + 1])} (objetivo={objetivo:.1f})")
        _, izq_sombreado, der_sombreado = mejor_tramo
        ax.axvspan(
            izq_sombreado - 0.5,
            der_sombreado + 0.5,
            color="#f7c6d9",
            alpha=0.35,
            zorder=0,
        )
        x_centro = (izq_sombreado + der_sombreado) / 2 + 1
        y_texto = ax.get_ylim()[1] * 0.93
        ax.text(
            x_centro,
            y_texto,
            "90%",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#8a2d55",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 2},
        )

    pos_x = pos_k
    ax.text(pos_x, cant_max, f"k = {k}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.show()

def plot_todos_anios_egreso_solo_cantidad_y_k(anio_minimo, anio_maximo, carrera: str) -> None:
    """Genera un barplot por cada año de egreso con la distribución de cohortes.

    Delega en plot_egresados_por_anio_cantidad_y_k para cada valor distinto de 'anio_egreso'
    presente en el DataFrame.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame devuelto por get_egresados_fisica, con columnas
        'dni', 'año_inscripcion_facultad' y 'anio_egreso'.
    """
    carrera_titulo = carrera
    carrera = _normalizar(carrera)
    carrera = carrera.lower()
    for anio in range(anio_minimo, anio_maximo + 1):
        path_csv = (
            _DEFAULT_DATOS_AGRUPADOS
            / carrera
            / f"resumen_egresados_{carrera}_{anio}.csv"
        )
        if not path_csv.exists():
            continue
        resumen = pd.read_csv(path_csv)
        plot_egresados_por_anio_cantidad_y_k(resumen, anio, carrera_titulo)

def plot_tiempo_cursando(
    df: pd.DataFrame,
    carrera: str,
    anio_desde: int,
    anio_hasta: int,
    mostrar_rectas_verticales: bool = False,
) -> None:
    """Genera dos gráficos sobre el tiempo de cursada de los egresados en un rango de años.

    El primero muestra la cantidad total de egresados por años cursando.
    El segundo desagrega por año de egreso mediante un gráfico de barras apiladas.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame pre-agregado con columnas 'cantidad_egresados', 'proporcion',
        'anio_egreso' y 'año_inscripcion_facultad' (concatenación de archivos
        resumen_egresados_*.csv de assets/datos_agrupados).
    carrera : str
        Nombre de la carrera (usado en los títulos).
    anio_desde : int
        Año de egreso mínimo (inclusive).
    anio_hasta : int
        Año de egreso máximo (inclusive).
    mostrar_rectas_verticales : bool
        Si True, agrega tres rectas verticales en el segundo gráfico:
        a la izquierda de la columna más alta, entre 10-11 y entre 15-16.
    """
    df = df.copy()
    df["tiempo_cursando"] = df["anio_egreso"] - df["año_inscripcion_facultad"] + 1

    filtrado = df[(df["anio_egreso"] >= anio_desde) & (df["anio_egreso"] <= anio_hasta)]

    tabla = (
        filtrado.groupby("tiempo_cursando")["cantidad_egresados"]
        .sum()
        .reset_index(name="total_egreso")
        .sort_values("tiempo_cursando")
    )

    grand_total = tabla["total_egreso"].sum()

    sns.set_theme(style="dark")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(tabla["tiempo_cursando"], tabla["total_egreso"])
    ax.set_xlabel("Años cursando")
    ax.set_ylabel("Total de egresados")
    ax.set_title(f"Egresados por tiempo cursando ({anio_desde}–{anio_hasta}). Carrera: {carrera}")
    ax.set_xticks(tabla["tiempo_cursando"])
    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim()[0] / grand_total, ax.get_ylim()[1] / grand_total)
    ax_right.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_right.set_ylabel("Proporción")
    plt.tight_layout()
    plt.show()

    pivot = (
        filtrado.groupby(["tiempo_cursando", "anio_egreso"])["cantidad_egresados"]
        .sum()
        .unstack(fill_value=0)
    )
    pivot = pivot.loc[sorted(pivot.index)]

    n_cohortes = filtrado["anio_egreso"].nunique()
    prom_ponderado = filtrado.groupby("tiempo_cursando")["proporcion"].sum() / n_cohortes

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_xlabel("Cantidad de años cursando", fontsize=15)
    ax.set_ylabel("Cantidad de egresados", fontsize=15)
    ax.set_title(f"Tiempo de cursada al momento de egreso ({anio_desde}–{anio_hasta}). Carrera: {carrera}", fontsize=15)
    ax.legend(title="Año de egreso", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)
    ax.tick_params(axis="x", rotation=0, labelsize=13)
    ax.tick_params(axis="y", labelsize=13)

    if mostrar_rectas_verticales:
        # Posiciones de cortes verticales sobre el eje categórico de barras.
        idx_pivot = pivot.index
        totales_por_tiempo = pivot.sum(axis=1)
        pos_columna_mas_alta = idx_pivot.get_loc(totales_por_tiempo.idxmax())

        def _posicion_entre(val_izq: int, val_der: int) -> Optional[float]:
            en_izq = val_izq in idx_pivot
            en_der = val_der in idx_pivot

            if en_izq and en_der:
                return (idx_pivot.get_loc(val_izq) + idx_pivot.get_loc(val_der)) / 2
            if en_izq:
                return idx_pivot.get_loc(val_izq) + 0.5
            if en_der:
                return idx_pivot.get_loc(val_der) - 0.5
            return None

        # 1) División inmediatamente a la izquierda de la columna más alta.
        corte_pico = pos_columna_mas_alta - 0.5
        if corte_pico >= -0.5:
            ax.axvline(x=corte_pico, color="black", linestyle="--", linewidth=2)

        # 2) Corte entre 10 y 11 años cursando.
        corte_10_11 = _posicion_entre(10, 11)
        if corte_10_11 is not None:
            ax.axvline(x=corte_10_11, color="green", linestyle="--", linewidth=2)

        # 3) Corte entre 15 y 16 años cursando.
        corte_15_16 = _posicion_entre(15, 16)
        if corte_15_16 is not None:
            ax.axvline(x=corte_15_16, color="blue", linestyle="--", linewidth=2)

        # Etiquetas de tramos: a | b | c | d.
        if corte_10_11 is not None and corte_15_16 is not None:
            x_min, x_max = ax.get_xlim()
            y_top = ax.get_ylim()[1] * 0.96
            segmentos = [
                ("a", (x_min + corte_pico) / 2),
                ("b", (corte_pico + corte_10_11) / 2),
                ("c", (corte_10_11 + corte_15_16) / 2),
                ("d", (corte_15_16 + x_max) / 2),
            ]
            for etiqueta, x_pos in segmentos:
                ax.text(x_pos, y_top, etiqueta, ha="center", va="top", fontsize=14, fontweight="bold")

    for i, tc in enumerate(pivot.index):
        total_height = pivot.loc[tc].sum()
        proporcion_simple = (total_height / grand_total) if grand_total else 0
        ax.text(i, total_height, f"{proporcion_simple:.1%}", ha="center", va="bottom", fontsize=11)
    ax_right2 = ax.twinx()
    ax_right2.set_ylim(ax.get_ylim()[0] / grand_total, ax.get_ylim()[1] / grand_total)
    ax_right2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_right2.tick_params(axis="y", labelsize=13)
    ax_right2.set_ylabel("% de egresados", fontsize=13)
    plt.tight_layout()
    plt.show()

    print(tabla.to_string(index=False))
    print(pivot.to_string())
    print(f"Promedio ponderado de proporción por tiempo cursando: {prom_ponderado.to_string()}")
    print(f"Total de egresados en el período: {grand_total}")
    return tabla, pivot, prom_ponderado

def estimacion_abandono(carrera, titulo):

    # calculamos los egresados por año de egreso y año de inscripción
    egresados_2012_2025 = pd.DataFrame()
    for anio in range(2012, 2026):
        egresados = pd.read_csv(f'../../../assets/datos_agrupados/{carrera}/resumen_egresados_{carrera}_{anio}.csv')
        egresados_2012_2025 = pd.concat([egresados_2012_2025, egresados])
    
    # agrupamos por año de inscripción y sumamos la cantidad de egresados
    total_egresados_por_anio_inscripcion = egresados_2012_2025.groupby('año_inscripcion_facultad')['cantidad_egresados'].sum().reset_index()

    # calculamos los inscriptos por año de inscripción
    personas = pd.read_csv(
        "../../../assets/bronze/FCEN/FCEN_oficial_2005_2025/reporte_personas_desde_2005.csv",
        usecols=["dni", "carrera_principal", "año_inscripcion_facultad"],
        dtype={"dni": str},
    )
    personas_carrera = personas[personas['carrera_principal'] == titulo]
    inscripciones_por_anio = personas_carrera.groupby('año_inscripcion_facultad')['dni'].nunique().reset_index()
    inscripciones_por_anio.rename(columns={'dni': 'cantidad_inscripciones'}, inplace=True)
    
    calculo = inscripciones_por_anio.merge(total_egresados_por_anio_inscripcion, on='año_inscripcion_facultad', how='left')

    # calculamos el abandono y la tasa de abandono
    calculo['tasa_egreso'] = calculo['cantidad_egresados'] / calculo['cantidad_inscripciones']
    calculo['abandono'] = calculo['cantidad_inscripciones'] - calculo['cantidad_egresados']
    calculo['tasa_abandono'] = calculo['abandono'] / calculo['cantidad_inscripciones']

    # calculamos abadono considerando solo los que tardaron 15 años o menos
    egresados_15_anios_2005_2011 = (
            egresados_2012_2025[
                (egresados_2012_2025['año_inscripcion_facultad'].between(2005, 2011))
                & (
                    egresados_2012_2025['anio_egreso']
                    <= egresados_2012_2025['año_inscripcion_facultad'] + 14
                )
            ]
            .groupby('año_inscripcion_facultad', as_index=False)['cantidad_egresados']
            .sum()
            .rename(columns={'cantidad_egresados': 'egresados_en_15_anios_o_menos'})
    )
    calculo_15_años = inscripciones_por_anio.merge(egresados_15_anios_2005_2011, on='año_inscripcion_facultad', how='left')
    calculo_15_años['tasa_egreso_15_años'] = calculo_15_años['egresados_en_15_anios_o_menos'] / calculo_15_años['cantidad_inscripciones']
    calculo_15_años['abandono_15_años'] = calculo_15_años['cantidad_inscripciones'] - calculo_15_años['egresados_en_15_anios_o_menos']
    calculo_15_años['tasa_abandono_15_años'] = calculo_15_años['abandono_15_años'] / calculo_15_años['cantidad_inscripciones']
    calculo_15_años = calculo_15_años[calculo_15_años['año_inscripcion_facultad'].between(2005, 2011)]

    # calculamos el d
    calculo_para_d = calculo[calculo['año_inscripcion_facultad'].between(2005, 2011)][['año_inscripcion_facultad', 'cantidad_egresados']]
    calculo_15_años['d'] = (calculo_para_d['cantidad_egresados'] - calculo_15_años['egresados_en_15_anios_o_menos'])/ calculo_15_años['cantidad_inscripciones']

    return total_egresados_por_anio_inscripcion, inscripciones_por_anio, calculo, calculo_15_años