import random
from functools import reduce
from random import randint
from secrets import randbits

from matplotlib import pyplot as plt


class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = 0
    def __str__(self):
        return 'chromosome: {}, fitness: {}'.format(self.chromosome, self.fitness)



# def filter_restrictions(individual, *restriction):
#     pass


def evaluate_individual(individual):
    chromosome = individual.chromosome
    weights = list(backpack.keys())
    values = list(backpack.values())
    fitness = sum(chromosome[i] * values[i] for i in range(len(chromosome)))
    weight = sum(chromosome[i] * weights[i] for i in range(len(chromosome)))

    return fitness if weight < max_weight else 0


def select_parents(population):
    x = population[randint(0, len(population) - 1)]
    y = population[randint(0, len(population) - 1)]
    return x, y

def mutate(chromosome):
    random_index = randint(0, len(chromosome) - 1)
    gene = chromosome[random_index]
    chromosome[random_index] = gene ^ 1
    return chromosome

def crossover(x, y, mutation_rate=0.01):
    strongest, weaker = (x,y) if x.fitness < y.fitness else (y,x)
    chromosome = list(range(len(x.chromosome)))
    for i in range(len(x.chromosome)):
        if random.random()  > 0.6:
            chromosome[i] = strongest.chromosome[i]
        else:
            chromosome[i] = weaker.chromosome[i]

    if random.random() < mutation_rate:
        chromosome = mutate(chromosome)
    return Individual(chromosome)


def generate_first_generation(number_chromosomes=5, number_individuals=100):
    population = []
    for  i in range(number_individuals):
        chromosome = [randint(0, 1) for _ in range(number_chromosomes)]
        population.append(Individual(chromosome))

    return population


def create_generation(population, num_individuals, elitism_rate=0.6, mutation_rate=0.01):
    mutation_rate =  0.1 if len(population) > 12 else 0.4
    population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
    size = int(len(population) * elitism_rate)
    strongest_individuals = population[:size]
    new_generation = []
    for i in range(num_individuals):
        x, y = select_parents(strongest_individuals)
        new_individual = crossover(x, y, mutation_rate)
        new_generation.append(new_individual)
    return new_generation


def create_info_message(generation, strongest_individual, satisfactory_solution):
    info = f"G({generation}): best chromossome: {strongest_individual.chromosome} fitness: {strongest_individual.fitness}"
    if satisfactory_solution is not None:
        gap = (strongest_individual.fitness / satisfactory_solution) * 100
        info += f" gap: {gap:.2f}"
    return info


def algorithm(num_individuals, num_generations=None, solution=None, satisfactory_solution=None):
    if solution is not None:
        satisfactory_solution = evaluate_individual(Individual(solution))
        print(satisfactory_solution)

        if len(solution) != len(backpack):
            raise ValueError('The number of solution must be equal to the number of chromosomes')

    if num_generations is None and satisfactory_solution is None:
        raise RuntimeError(
            'Unable to determine the stopping point for the algorithm. Please define either a satisfactory solution or a number of generations.'
        )

    population = generate_first_generation(number_chromosomes=len(backpack), number_individuals=num_individuals)

    # last_fitness = 0
    # generation_without_changes = 0
    # max_without_changes = 5
    mutation_rate = 0.01
    strongest_individual = population[0]
    fitness = population[0].fitness
    generation = 0
    while True:
        if num_generations is not None and generation >= num_generations:
            break
        # Evaluate
        for individual in population:
            individual.fitness = evaluate_individual(individual)

        current_strongest_individual = max(population, key=lambda individual: individual.fitness)
        # todo: reajustar essa logica que esta errada:
        # if strongest_individual.fitness == last_fitness:
        #     generation_without_changes+=1
        # else:
        #     print("Zerando")
        #     generation_without_changes = 0
        # if generation_without_changes > max_without_changes:
        #     mutation_rate += 0.01
        # if generation_without_changes > 50:
        #     print(generation_without_changes)
        #     return strongest_individual


        if current_strongest_individual.fitness > strongest_individual.fitness:
            strongest_individual = current_strongest_individual
            fitness = strongest_individual.fitness

        info = create_info_message(generation, strongest_individual, satisfactory_solution)


        if satisfactory_solution is not None and fitness >= satisfactory_solution: return strongest_individual

        print(info)
        # last_fitness = strongest_individual.fitness
        best_fitness_history.append(current_strongest_individual.fitness)
        population = create_generation(population, num_individuals, mutation_rate=mutation_rate)
        generation+=1

    return strongest_individual

backpack = {
    7: 15,   # Corda de alpinismo
    10: 30,  # Lanterna de alta duração
    5: 12,   # Canivete multifunção
    20: 50,  # Kit médico completo
    8: 20,   # Mapa detalhado da ilha
    12: 25,  # Rádio comunicador de longo alcance
    9: 22,   # Barraca resistente a tempestades
    18: 45,  # Saco de dormir térmico
    14: 40,  # Fogareiro e combustível extra
    6: 10,   # Kit de pesca
    2: 5,    # Isqueiro resistente à água
    17: 35,  # Estoque de alimentos desidratados
    11: 28,  # Mochila adicional
    3: 8,    # Apito de emergência
    4: 9,    # Espelho para sinalização
    16: 38,  # Kit de ferramentas básicas
    13: 32,  # Jaqueta térmica
    19: 48,  # Purificador de água portátil
    15: 36,  # Roupas extras para frio extremo
    1: 2,    # Pequena bússola
}

max_weight = 80


# [0, 1, 1, 0, 0, 0, 1, 1]


# best_chromosome = [1, 1, 1, 0, 0, 1,1,0,1]

# value =  5 +  9 +  10 +  + 12 = 36
# weight =  3 +  4 +  2 +  5  = 14

best_fitness_history = []

#10000
print(algorithm(num_individuals=10, num_generations=100, satisfactory_solution=210))


plt.plot(best_fitness_history)
plt.title('Fitness Progression Over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.show()









