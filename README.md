# Tesis de Licenciatura de Sol Calloni. 
Director: Martín Pustilnik

Co-director: Guillermo Durán

<div align="center">
  
![Banner](https://github.com/solcalloni/tesis-lic-ciencias-de-datos/blob/main/images/logofcen.jpg)

</div>

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

Repositorio principal para el desarrollo de la tesis de Licenciatura en Ciencias de Datos de Sol Calloni

Se busca entrenar un modelo de Aprendizaje Automático para identificar a los estudiantes de la Facultad de Ciencias Exactas y Naturales (FCEN) de la Universidad Nacional de Buenos Aires (UBA) que se encuentran en riesgo de abandono. Los datos provienen del SIU-Guaraní (Sistema de Información Universitario Guaraní) del CBC (Ciclo Básico Común), del SIU-Guaraní del FCEN y de una encuesta a los ingresantes al FCEN.

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
