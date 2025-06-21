import pandas as pd

def drop_columns(encuesta_2c_2023, primer_c_2024_4, segundo_c_2024_1):
    #Eliminamos las columnas vacías y las que sabemos que no usaremos
    encuesta_2c_2023.drop(columns=['Libreta Universitaria Nº', 'Apellido/s y Nombres', 'Lenguaje inicial', 'Semilla', 'Contraseña', 'Fecha de envío', 'Última página', 'Comentarios Finales'], inplace=True)
    primer_c_2024_4.drop(columns=['Apellido/s y Nombres', 'Lenguaje inicial', 'Semilla', 'Fecha de envío', 'Última página', 'Unnamed: 5', 'Unnamed: 8', 'Unnamed: 65', 'Unnamed: 66', 'Comentarios Finales'], inplace = True)
    segundo_c_2024_1.drop(columns=['Apellido/s y Nombres', 'Lenguaje inicial', 'Semilla', 'Fecha de envío', 'Última página','Unnamed: 5', 'Unnamed: 8', 'Unnamed: 65', 'Unnamed: 66', 'Comentarios Finales'], inplace=True)

def delete_null_values(encuesta_2c_2023, primer_c_2024_4, segundo_c_2024_1):
    # saco los valores nulos
    encuesta_2c_2023.dropna(subset=['Nº de documento'], inplace=True)
    primer_c_2024_4.dropna(subset=['Nº de documento'], inplace=True)
    segundo_c_2024_1.dropna(subset=['Nº de documento'], inplace=True)

def drop_dni_duplicates(encuesta_2c_2023, primer_c_2024_4, segundo_c_2024_1):
    # Saco los duplicados
    encuesta_2c_2023.drop_duplicates(inplace=True)
    primer_c_2024_4.drop_duplicates(inplace=True)
    segundo_c_2024_1.drop_duplicates(inplace=True)
    segundo_c_2024_1 = segundo_c_2024_1.drop_duplicates(subset='Nº de documento', keep='last')
    encuesta_2c_2023 = encuesta_2c_2023.drop_duplicates(subset='Nº de documento', keep='last')
    primer_c_2024_4 = primer_c_2024_4.drop_duplicates(subset='Nº de documento', keep='last')

def drop_columns_final_version(encuesta_2c_2023, primer_c_2024_4, segundo_c_2024_1):
    #Eliminamos las columnas vacías y las que sabemos que no usaremos
    encuesta_2c_2023.drop(columns=['ID de respuesta','Libreta Universitaria Nº', 'Apellido/s y Nombres',
                                   'Lenguaje inicial', 'Semilla', 'Contraseña', 'Fecha de envío',
                                   'Última página', 'Comentarios Finales',
                                   '¿Cuándo ingresaste a Exactas?\xa0',
                                   '¿Cuándo ingresaste a Exactas?\xa0 [Otro]',
                                   'La facultad tiene un programa -Sin Barreras- que busca eliminar las barreras del contexto de modo que si estás dentro del colectivo o querés hacer alguna consulta vinculada podés comunicarte a sinbarreras@de.fcen.uba.ar',
                                   '¿Contas con un espacio para el estudio en tu hogar, qué características tiene? [Es adecuado]',
                                   '¿Contas con un espacio para el estudio en tu hogar, qué características tiene? [Es medianamente adecuado]',
                                   '¿Contas con un espacio para el estudio en tu hogar, qué características tiene? [Es inadecuado]',
                                   '¿Cuál o cuáles materias del CBC te generaron mayor dificultad? [Ninguna]',
                                   '¿Cuál o cuáles materias del CBC te generaron mayor dificultad? [No corresponde]',
                                   '¿Aprobaste alguna materia por UBA XXI?',
                                   '¿Cuál/es? [Análisis Matemático]',
                                   '¿Cuál/es? [Algebra]',
                                   '¿Cuál/es? [Matemática]',
                                   '¿Cuál/es? [Física]',
                                   '¿Cuál/es? [Química]',
                                   '¿Cuál/es? [Biología]',
                                   '¿Cuál/es? [Introducción al Conocimiento Científico (IPC)]',
                                   '¿Cuál/es? [Introducción al Conocimiento de la Sociedad y el Estado (ICSE)]'],
                                    inplace=True)
    primer_c_2024_4.drop(columns=['ID de respuesta','Apellido/s y Nombres',
                                  'Lenguaje inicial', 'Semilla', 'Fecha de envío',
                                  'Última página', 'Comentarios Finales',
                                  '¿En qué año ingresaste a Exactas?\xa0',
                                  '¿En qué cuatrimestre ingresaste?\xa0',
                                  'La facultad tiene un programa -Sin Barreras- que busca eliminar las barreras del contexto de modo que si estás dentro del colectivo o querés hacer alguna consulta vinculada podés comunicarte a sinbarreras@de.fcen.uba.ar',
                                  'Uno de los propósitos de esta encuesta es adaptar las herramientas de acompañamiento de la Facultad para que se adecuen a las necesidades de cada estudiante, por lo que nos ayudaría que nos dejes tus datos personales:',
                                  '¿Participaste de alguna de las siguientes actividades de la Facultad? [Exactas Programa]',
                                  '¿Participaste de alguna de las siguientes actividades de la Facultad? [Actividad de Laboratorio de la materia Quimica del CBC (CU)]',
                                  '¿Contas con un espacio para el estudio en tu hogar, qué características tiene?',
                                  '¿Cómo evaluás este espacio destinado al estudio?',
                                  'Indicanos la modalidad de cursada y aprobación de cada materia del CBC [Int. al Conocimiento Científico (IPC)]',
                                  'Indicanos la modalidad de cursada y aprobación de cada materia del CBC [Int. al Conocimiento de la Sociedad y el Estado (ICSE)]',
                                  'Indicanos la modalidad de cursada y aprobación de cada materia del CBC [Análisis Matemático]',
                                  'Indicanos la modalidad de cursada y aprobación de cada materia del CBC [Algebra]',
                                  'Indicanos la modalidad de cursada y aprobación de cada materia del CBC [Matemática]',
                                  'Indicanos la modalidad de cursada y aprobación de cada materia del CBC [Física]',
                                  'Indicanos la modalidad de cursada y aprobación de cada materia del CBC [Química]',
                                  'Indicanos la modalidad de cursada y aprobación de cada materia del CBC [Biología]',
                                  '¿En caso de haber tenido dificultades, cómo las resolviste?\xa0¿recurriste a alguien? [Considero no haber tenido dificultades]'],
                                  inplace = True)
    segundo_c_2024_1.drop(columns=['ID de respuesta','Apellido/s y Nombres', 
                                   'Lenguaje inicial', 'Semilla', 'Fecha de envío',
                                   'Última página', 'Comentarios Finales',
                                   '¿En qué año ingresaste a Exactas?\xa0',
                                   '¿En qué cuatrimestre ingresaste?\xa0',                                   
                                   'La facultad tiene un programa -Sin Barreras- que busca eliminar las barreras del contexto de modo que si estás dentro del colectivo o querés hacer alguna consulta vinculada podés comunicarte a sinbarreras@de.fcen.uba.ar',
                                   'Uno de los propósitos de esta encuesta es adaptar las herramientas de acompañamiento de la Facultad para que se adecuen a las necesidades de cada estudiante, por lo que nos ayudaría que nos dejes tus datos personales:',
                                   '¿Participaste de alguna de las siguientes actividades de la Facultad? [Exactas Programa]',
                                   '¿Participaste de alguna de las siguientes actividades de la Facultad? [Actividad de Laboratorio de la materia Quimica del CBC (CU)]',
                                   '¿Contas con un espacio para el estudio en tu hogar, qué características tiene?',
                                   '¿Cómo evaluás este espacio destinado al estudio?',
                                   'Indicanos la modalidad de cursada y aprobación de cada materia del CBC [Int. al Conocimiento Científico (IPC)]',
                                   'Indicanos la modalidad de cursada y aprobación de cada materia del CBC [Int. al Conocimiento de la Sociedad y el Estado (ICSE)]',
                                   'Indicanos la modalidad de cursada y aprobación de cada materia del CBC [Análisis Matemático]',
                                   'Indicanos la modalidad de cursada y aprobación de cada materia del CBC [Algebra]',
                                   'Indicanos la modalidad de cursada y aprobación de cada materia del CBC [Matemática]',
                                   'Indicanos la modalidad de cursada y aprobación de cada materia del CBC [Física]',
                                   'Indicanos la modalidad de cursada y aprobación de cada materia del CBC [Química]',
                                   'Indicanos la modalidad de cursada y aprobación de cada materia del CBC [Biología]',
                                   '¿En caso de haber tenido dificultades, cómo las resolviste?\xa0¿recurriste a alguien? [Considero no haber tenido dificultades]'],
                                    inplace=True)