# ga.py

import random
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Para evitar problemas con entornos sin Tkinter
import matplotlib.pyplot as plt

def initialize_population(pop_size, M, D):
    population = []
    for _ in range(pop_size):
        chromosome = [random.randint(0, D) for _ in range(M)]
        population.append(chromosome)
    return population

def fitness(individual, D, capacity, mu_elig, lambda_cap, get_eligibility_for_group):
    fit = 0
    # Penalización por uso de docentes externos
    external_count = sum(1 for gene in individual if gene == D)
    fit += external_count
    
    # Contar asignaciones para cada docente interno
    assign_count = np.zeros(D, dtype=int)
    
    # Penalización por elegibilidad
    for idx, gene in enumerate(individual):
        if gene < D:
            assign_count[gene] += 1
            eleg_vector = get_eligibility_for_group(idx)
            if eleg_vector[gene] == 0:
                fit += mu_elig
    
    # Penalización por capacidad
    for j in range(D):
        if assign_count[j] > capacity[j]:
            fit += lambda_cap * (assign_count[j] - capacity[j])
    
    return fit

def tournament_selection(population, tournament_size, fitness_fn):
    tournament = random.sample(population, tournament_size)
    # Ordenar en base al fitness (menor es mejor)
    tournament.sort(key=lambda ind: fitness_fn(ind))
    return tournament[0]

def crossover(parent1, parent2, M, crossover_rate):
    if random.random() < crossover_rate:
        point = random.randint(1, M - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    else:
        # Sin cruce
        return parent1, parent2

def mutate(chromosome, M, D, mutation_rate):
    new_chromosome = chromosome.copy()
    for i in range(M):
        if random.random() < mutation_rate:
            new_chromosome[i] = random.randint(0, D)
    return new_chromosome

def repair(chromosome, D, capacity, get_eligibility_for_group):
    new_chrom = chromosome.copy()
    # Reparar asignaciones no elegibles
    for i, gene in enumerate(new_chrom):
        if gene < D:
            eleg_vector = get_eligibility_for_group(i)
            if eleg_vector[gene] == 0:
                new_chrom[i] = D
    
    # Contar asignaciones
    assign_count = np.zeros(D, dtype=int)
    for gene in new_chrom:
        if gene < D:
            assign_count[gene] += 1
    
    # Corregir excedentes
    for j in range(D):
        if assign_count[j] > capacity[j]:
            exceso = assign_count[j] - capacity[j]
            indices = [idx for idx, g in enumerate(new_chrom) if g == j]
            indices_to_change = random.sample(indices, exceso)
            for idx in indices_to_change:
                new_chrom[idx] = D
    return new_chrom

def genetic_algorithm(pop_size,
                      generations,
                      tournament_size,
                      crossover_rate,
                      mutation_rate,
                      D,
                      M,
                      capacity,
                      mu_elig,
                      lambda_cap,
                      get_eligibility_for_group):
    # Inicializar población
    population = initialize_population(pop_size, M, D)
    
    best_solution = None
    best_solution_fit = float('inf')
    best_fit = float('inf')
    
    # Para gráficos
    offline_fitness_list = []
    online_fitness_list = []
    best_so_far_list = []
    
    # Definir función de fitness con los parámetros fijos
    def fitness_fn(ind):
        return fitness(ind, D, capacity, mu_elig, lambda_cap, get_eligibility_for_group)
    
    for gen in range(generations):
        new_population = []
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, tournament_size, fitness_fn)
            parent2 = tournament_selection(population, tournament_size, fitness_fn)
            
            child1, child2 = crossover(parent1, parent2, M, crossover_rate)
            child1 = mutate(child1, M, D, mutation_rate)
            child2 = mutate(child2, M, D, mutation_rate)
            child1 = repair(child1, D, capacity, get_eligibility_for_group)
            child2 = repair(child2, D, capacity, get_eligibility_for_group)
            
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        
        population = new_population
        # Calcular fitness de cada individuo
        fitness_values = [fitness_fn(ind) for ind in population]
        
        # Offline performance
        current_offline = min(fitness_values)
        # Online performance
        current_online = sum(fitness_values) / len(fitness_values)
        
        # Actualizar best-so-far
        if current_offline < best_fit:
            best_fit = current_offline
        
        offline_fitness_list.append(current_offline)
        online_fitness_list.append(current_online)
        best_so_far_list.append(best_fit)
        
        # Actualizar mejor solución
        current_best = population[fitness_values.index(current_offline)]
        if current_offline < best_solution_fit:
            best_solution_fit = current_offline
            best_solution = current_best
        
        if gen % 20 == 0:
            print(f"Generación {gen}: Mejor fitness = {best_solution_fit}")
    
    # Graficar las curvas finales
    plt.figure()
    plt.title("Rendimiento del Algoritmo Genético")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    
    plt.plot(range(generations), offline_fitness_list)
    plt.plot(range(generations), online_fitness_list)
    plt.plot(range(generations), best_so_far_list)
    
    plt.legend(["Offline", "Online", "Best-so-far"])
    plt.savefig("output/performance.png")
    print("Gráfico guardado en 'output/performance.png'")

    return best_solution, best_solution_fit
