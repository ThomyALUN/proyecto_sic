import os
import random

from modules.data_loader import load_data
from modules.genethic import genetic_algorithm
from modules.output_manager import export_solution_to_excel, print_final_solution

if __name__ == "__main__":
    # Crear carpeta output si no existe
    if not os.path.exists("output"):
        os.makedirs("output")
    
    # Cargar datos
    archivo_entrada = "data/AsignaturasAG_EntradaSimulacion.xlsx"
    (asignaturas,
     docentes,
     docente_ids,
     D,
     capacity,
     group_subject,
     group_names,
     group_number,
     M,
     eligibility) = load_data(archivo_entrada)
    
    # Definir el helper para elegibilidad
    def get_eligibility_for_group(group_index):
        # Devuelve la lista de elegibilidad [0/1] para cada docente en la asignatura
        aid = group_subject[group_index]
        return eligibility[aid]
    
    # Parámetros
    pop_size = 50
    generations = 200
    tournament_size = 3
    random.seed(42)
    
    crossover_rate = 0.8
    mutation_rate = 0.1
    
    lambda_cap = 10  # penalización por violar capacidad
    mu_elig = 20     # penalización por docente no elegible
    
    # Ejecutar AG
    best_solution, best_solution_fit = genetic_algorithm(
        pop_size=pop_size,
        generations=generations,
        tournament_size=tournament_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        D=D,
        M=M,
        capacity=capacity,
        mu_elig=mu_elig,
        lambda_cap=lambda_cap,
        get_eligibility_for_group=get_eligibility_for_group
    )
    
    # Mostrar resultados finales
    print("\nMejor solución encontrada:")
    print(best_solution)
    print("Fitness:", best_solution_fit)
    
    # Exportar a Excel
    output_file = "output/AsignacionDocentes.xlsx"
    export_solution_to_excel(
        best_solution,
        filename=output_file,
        group_subject=group_subject,
        group_names=group_names,
        group_number=group_number,
        asignaturas=asignaturas,
        docentes=docentes,
        docente_ids=docente_ids,
        D=D
    )
    
    # Imprimir asignaciones en consola
    print_final_solution(
        best_solution,
        group_subject,
        group_names,
        asignaturas,
        docente_ids,
        docentes,
        D
    )
