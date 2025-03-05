# data_loader.py

import pandas as pd
import numpy as np

def load_data(archivo_entrada):
    # Leer hojas
    df_areas_conocimiento = pd.read_excel(archivo_entrada, sheet_name="AreasConocimiento")
    df_asignaturas = pd.read_excel(archivo_entrada, sheet_name="Asignaturas")
    df_docentes = pd.read_excel(archivo_entrada, sheet_name="Docentes")
    
    # Ordenar por ID (opcional, según como estén definidos)
    df_asignaturas.sort_values("Id", inplace=True)
    df_docentes.sort_values("Id", inplace=True)
    
    # Diccionario para asignaturas: clave Id, valor: (Nombre, Area de conocimiento)
    asignaturas = {}
    for _, row in df_asignaturas.iterrows():
        asignaturas[row["Id"]] = {
            "Nombre": row["Nombre"],
            "Area": row["Area de conocimiento"]
        }
    
    # Diccionario para docentes: clave Id, valor: (Nombre, Capacidad, Conjunto de Areas)
    docentes = {}
    for _, row in df_docentes.iterrows():
        areas = set()
        for i in range(1, 5):
            area = row[f"Area de conocimiento {i}"]
            if pd.notna(area):
                areas.add(area)
        docentes[row["Id"]] = {
            "Nombre": row["Nombre"],
            "Capacidad": int(row["Capacidad grupo"]),
            "Areas": areas
        }
    
    # Lista de docentes internos (suponemos que todos los docentes de la hoja son internos)
    docente_ids = list(docentes.keys())
    D = len(docente_ids)
    
    # Vector de capacidades (ordenados según docente_ids)
    capacity = np.array([docentes[doc_id]["Capacidad"] for doc_id in docente_ids])
    
    # Listas para grupos
    group_subject = []  # ID de la asignatura de cada grupo
    group_names = []    # Nombre de la asignatura (opcional)
    group_number = []   # Número de grupo (1, 2, 3, ...)
    
    for _, row in df_asignaturas.iterrows():
        for g in range(1, row["Grupos"] + 1):
            group_subject.append(row["Id"])
            group_names.append(row["Nombre"])
            group_number.append(g)
    
    M = len(group_subject)  # cantidad total de grupos
    
    # Construir matriz/diccionario de elegibilidad
    asignatura_area = {aid: data["Area"] for aid, data in asignaturas.items()}
    eligibility = {}
    for aid, data in asignaturas.items():
        area = asignatura_area[aid]
        eleg_list = []
        for doc_id in docente_ids:
            eleg_list.append(1 if area in docentes[doc_id]["Areas"] else 0)
        eligibility[aid] = eleg_list
    
    return (
        asignaturas,
        docentes,
        docente_ids,
        D,
        capacity,
        group_subject,
        group_names,
        group_number,
        M,
        eligibility
    )
