import numpy as np
import pandas as pd
import random

# -----------------------------
# 1. Lectura de datos desde Excel
# -----------------------------
# Se asume que el archivo Excel tiene 3 hojas:
# "AreasConocimiento": columnas: Id, Nombre
# "Asignaturas": columnas: Id, Nombre, Area de conocimiento, Grupos
# "Docentes": columnas: Id, Nombre, Area de conocimiento 1, Area de conocimiento 2, Area de conocimiento 3, Area de conocimiento 4, Capacidad grupo

archivo_excel = "PlantillaAG_Entrada.xlsx"

# Leer hojas
df_areas_conocimiento = pd.read_excel(archivo_excel, sheet_name="AreasConocimiento")
df_asignaturas = pd.read_excel(archivo_excel, sheet_name="Asignaturas")
df_docentes = pd.read_excel(archivo_excel, sheet_name="Docentes")

# Ordenar por ID (opcional, según como estén definidos)
df_asignaturas.sort_values("Id", inplace=True)
df_docentes.sort_values("Id", inplace=True)

# -----------------------------
# 2. Procesamiento de Datos y Parámetros
# -----------------------------

# Diccionario para asignaturas: clave Id, valor: (Nombre, Area de conocimiento)
asignaturas = {}
for _, row in df_asignaturas.iterrows():
    asignaturas[row["Id"]] = {"Nombre": row["Nombre"], "Area": row["Area de conocimiento"]}

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
# Se usará un orden basado en la lista obtenida
docente_ids = list(docentes.keys())
D = len(docente_ids)

# Vector de capacidades (ordenados según docente_ids)
capacity = np.array([docentes[doc_id]["Capacidad"] for doc_id in docente_ids])

# Para cada grupo, determinar a qué asignatura pertenece y extraer su área.
# Además, guardar el nombre del grupo.


# M = len(df_asignaturas)  # número total de grupos (asumiendo que cada asignatura tiene un grupo)
# group_subject = []  # almacena el Id de cada grupo (en el mismo orden de df_asignaturas)
# group_names = []    # almacena el Nombre para imprimir el resultado

# for _, row in df_asignaturas.iterrows():
#     group_subject.append(row["Id"])
#     group_names.append(row["Nombre"])

group_subject = []  # almacena el ID de la asignatura de cada grupo
group_names = []    # almacena el NombreGrupo para imprimir el resultado

for _, row in df_asignaturas.iterrows():
    for _ in range(row["Grupos"]):  # Asumiendo que la columna "Grupos" indica la cantidad de grupos por asignatura
        group_subject.append(row["Id"])
        group_names.append(row["Nombre"])

M = len(group_subject)  # número total de grupos

# Construir la matriz de elegibilidad:
# eligibility[i][j] = 1 si el docente j es elegible para impartir la asignatura i.
# Usaremos la posición de la asignatura según su Id (suponiendo que los IDs están en df_asignaturas).
# Creamos un diccionario para mapear Id a su Area.
asignatura_area = {aid: data["Area"] for aid, data in asignaturas.items()}

# Creamos una matriz (o diccionario) de elegibilidad para cada asignatura y cada docente.
# Lo haremos como un diccionario: eligibility[asignatura_id][j] = 1 o 0.
eligibility = {}  # clave: asignatura_id, valor: lista de 0/1 para cada docente (según el orden en docente_ids)
for aid in asignaturas.keys():
    area = asignatura_area[aid]
    eleg_list = []
    for doc_id in docente_ids:
        if area in docentes[doc_id]["Areas"]:
            eleg_list.append(1)
        else:
            eleg_list.append(0)
    eligibility[aid] = eleg_list

# Para facilitar el uso en el algoritmo, definimos una función que dado el índice de grupo, retorna la elegibilidad vectorial para los docentes internos para esa asignatura.
def get_eligibility_for_group(group_index):
    aid = group_subject[group_index]
    return eligibility[aid]  # lista de tamaño D

# -----------------------------
# 3. Parámetros del Algoritmo Genético
# -----------------------------
pop_size = 50
generations = 200
tournament_size = 3
mutation_rate = 0.1

# Penalizaciones
lambda_cap = 10   # penalización por violar capacidad
mu_elig = 20      # penalización por asignar docente no elegible

# Nota sobre la codificación:
# Cada cromosoma se representa como una lista de longitud M.
# Cada gen toma un valor en {0, 1, ..., D-1, D} donde:
#   - 0 a D-1 -> índice en docente_ids (docentes internos)
#   - D -> opción de "Docente Ocasional" (externo)

# -----------------------------
# 4. Funciones del Algoritmo Genético
# -----------------------------

def initialize_population():
    population = []
    for _ in range(pop_size):
        chromosome = [random.randint(0, D) for _ in range(M)]
        population.append(chromosome)
    return population

def fitness(chromosome):
    fit = 0
    # Penalización por uso de docentes externos
    external_count = sum(1 for gene in chromosome if gene == D)
    fit += external_count
    
    # Contar asignaciones para cada docente interno
    assign_count = np.zeros(D, dtype=int)
    
    # Penalización por elegibilidad
    for idx, gene in enumerate(chromosome):
        if gene < D:
            assign_count[gene] += 1
            eleg_vector = get_eligibility_for_group(idx)
            if eleg_vector[gene] == 0:  # docente no elegible para esta asignatura
                fit += mu_elig
    
    # Penalización por capacidad
    for j in range(D):
        if assign_count[j] > capacity[j]:
            fit += lambda_cap * (assign_count[j] - capacity[j])
    
    return fit

def tournament_selection(population):
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=lambda ind: fitness(ind))
    return tournament[0]

def crossover(parent1, parent2):
    point = random.randint(1, M - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(chromosome):
    new_chromosome = chromosome.copy()
    for i in range(M):
        if random.random() < mutation_rate:
            new_chromosome[i] = random.randint(0, D)
    return new_chromosome

def repair(chromosome):
    new_chrom = chromosome.copy()
    # Reparar asignaciones no elegibles: si se asigna docente interno no elegible, asignar docente externo.
    for i, gene in enumerate(new_chrom):
        if gene < D:
            eleg_vector = get_eligibility_for_group(i)
            if eleg_vector[gene] == 0:
                new_chrom[i] = D
    # Contar asignaciones por docente interno
    assign_count = np.zeros(D, dtype=int)
    for gene in new_chrom:
        if gene < D:
            assign_count[gene] += 1
    # Si se excede la capacidad, reasignar algunos grupos a docente externo.
    for j in range(D):
        if assign_count[j] > capacity[j]:
            exceso = assign_count[j] - capacity[j]
            indices = [i for i, gene in enumerate(new_chrom) if gene == j]
            indices_to_change = random.sample(indices, exceso)
            for idx in indices_to_change:
                new_chrom[idx] = D
    return new_chrom

def genetic_algorithm():
    population = initialize_population()
    best = None
    best_fit = float('inf')
    
    for gen in range(generations):
        new_population = []
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            child1 = repair(child1)
            child2 = repair(child2)
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        population = new_population
        
        current_best = min(population, key=lambda ind: fitness(ind))
        current_fit = fitness(current_best)
        if current_fit < best_fit:
            best_fit = current_fit
            best = current_best
        if gen % 20 == 0:
            print(f"Generación {gen}: Mejor fitness = {best_fit}")
    
    return best, best_fit

# -----------------------------
# 5. Ejecución del Algoritmo y Salida de Resultados
# -----------------------------
best_solution, best_solution_fit = genetic_algorithm()
print("\nMejor solución encontrada:")
print(best_solution)
print("Fitness:", best_solution_fit)

# Interpretación e impresión de la solución final:
# Para cada grupo, se imprime el nombre del grupo, el nombre de la asignatura y el docente asignado.
# Si el gen es menor que D, se obtiene el docente interno correspondiente; si es D, se imprime "Docente Ocasional".

print("\nAsignación final por grupo:")
for i, gene in enumerate(best_solution):
    # Obtener la asignatura del grupo
    aid = group_subject[i]
    asignatura_nombre = asignaturas[aid]["Nombre"]
    grupo_nombre = group_names[i]
    if gene < D:
        docente_id = docente_ids[gene]
        docente_nombre = docentes[docente_id]["Nombre"]
    else:
        docente_nombre = "Docente Ocasional"
    print(f"Grupo: {grupo_nombre} (Asignatura: {asignatura_nombre}) --> {docente_nombre}")