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
    sns.set_theme(style="darkgrid")
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


def plot_egresados_biologia_por_anio(df: pd.DataFrame) -> None:
    """Genera un par de barplots por cada año de egreso con la distribución de cohortes.

    Delega en plot_egresados_por_anio para cada valor distinto de 'anio_egreso'
    presente en el DataFrame, garantizando el mismo aspecto visual que el resto
    de carreras.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame devuelto por get_egresados_biologia, con columnas
        'dni', 'año_inscripcion_facultad' y 'anio_egreso'.
    """
    for anio, grupo in df.groupby("anio_egreso"):
        plot_egresados_por_anio(grupo, anio)


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


def plot_egresados_fisica_por_anio(df: pd.DataFrame) -> None:
    """Genera un par de barplots por cada año de egreso con la distribución de cohortes.

    Delega en plot_egresados_por_anio para cada valor distinto de 'anio_egreso'
    presente en el DataFrame.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame devuelto por get_egresados_fisica, con columnas
        'dni', 'año_inscripcion_facultad' y 'anio_egreso'.
    """
    for anio, grupo in df.groupby("anio_egreso"):
        plot_egresados_por_anio(grupo, anio)

