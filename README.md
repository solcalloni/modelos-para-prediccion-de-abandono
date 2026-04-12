# Tesis de Licenciatura de Sol Calloni. 
Director: Martín Pustilnik

Co-director: Guillermo Durán

<div align="center">
  
![Banner](https://github.com/solcalloni/tesis-lic-ciencias-de-datos/blob/main/images/logofcen.jpg)

</div>

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

Repositorio principal de la tesis de Licenciatura en Ciencias de Datos de Sol Calloni.

Se busca entrenar un modelo de Aprendizaje Automático para identificar a los estudiantes de la Facultad de Ciencias Exactas y Naturales (FCEN) de la Universidad Nacional de Buenos Aires (UBA) que se encuentran en riesgo de abandono. Los datos provienen del SIU-Guaraní (Sistema de Información Universitario Guaraní) del CBC (Ciclo Básico Común), del SIU-Guaraní del FCEN y de una encuesta a los ingresantes a la FCEN realizada por el programa +Acompañamiento.

A su vez, en la carpeta `packages/exploratory-data-analysis/FCEN_2005_2025` se encuentra el trabajo realizado para el trabajo "Un estimador estadístico del abandono para carreras de la facultad de Ciencias Exactas y Naturales de la Universidad de Buenos Aires" de Sol Calloni y Martin Pustilnik. También se encuentran los datos agrupados en la carpeta `assets/datos_agrupados`

## Setup

### Opción 1: Script automático (Recomendado)

1. Clona el repositorio:
```bash
git clone <url-del-repositorio>
cd modelos-para-prediccion-de-abandono
```

2. Ejecuta el script de configuración:
```bash
chmod +x set_up.sh
./set_up.sh
```

El script creará automáticamente un entorno virtual, lo activará e instalará todas las dependencias.

### Opción 2: Configuración manual

1. Clona el repositorio:
```bash
git clone <url-del-repositorio>
cd modelos-para-prediccion-de-abandono
```

2. Crea un entorno virtual:
```bash
python3 -m venv .venv
```

3. Activa el entorno virtual:
```bash
# En macOS/Linux:
source .venv/bin/activate

# En Windows:
.venv\Scripts\activate
```

4. Instala las dependencias:
```bash
pip install -r requirements.txt
```

### Activación del entorno virtual

Cada vez que trabajes en el proyecto, recuerda activar el entorno virtual:

```bash
# En macOS/Linux:
source .venv/bin/activate

# En Windows:
.venv\Scripts\activate
```

Para desactivar el entorno virtual:
```bash
deactivate
```

## Archivos del repositorio

```
├── README.md
├── assets
│   ├── bronze
│   ├── datos_agrupados
│   │   ├── biologia
│   │   ├── computacion
│   │   └── fisica
│   ├── gold
│   ├── resultados_modelos
│   └── silver
├── images
│   └── logofcen.jpg
├── packages
│   ├── constants
│   │   ├── materias_biologia.yaml
│   │   ├── materias_computacion.yaml
│   │   └── materias_fisica.yaml
│   ├── experiments
│   │   ├── 01-division_datos.ipynb
│   │   ├── 02-experimento_0.ipynb
│   │   ├── 03-experimento_1.ipynb
│   │   ├── 04-experimento_2.ipynb
│   │   ├── 05-analisis_de_arboles.ipynb
│   │   └── 06-analisis_de_arboles.ipynb
│   └── exploratory-data-analysis
│       ├── CBC
│       ├── FCEN
│       ├── FCEN_2005_2025
│       ├── datos_unificados
│       ├── distancia_de_viaje.ipynb
│       ├── encuestas
├── requirements.txt
└── set_up.sh
```

