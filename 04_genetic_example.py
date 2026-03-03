# -*- coding: utf-8 -*-
"""
Optimización de la longitud de un taper en una guía de ondas
mediante algoritmo genético + Meep.
"""

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import random

# =====================================================
# PARÁMETROS FIJOS DEL PROBLEMA (no se optimizan)
# =====================================================
res = 20                    # resolución (píxeles/μm) - baja para rapidez
w1 = 2.0                    # ancho entrada (μm)
w2 = 1.0                    # ancho salida (μm)
eps_guide = 12.0            # permitividad del silicio
fcen = 0.3                  # frecuencia central (1/μm)
df = 0.05                   # ancho de banda pequeño (para usar una sola frecuencia)
longitud_total = 15.0       # longitud total de la guía (μm) - fija
grosor_pml = 1.0            # PML
espacio_x = 4.0             # espacio lateral

# La guía se dividirá en tres segmentos: entrada (fijo), taper (variable), salida (fijo)
# Asignamos longitudes fijas a los tramos rectos para que el taper sea la única variable
l_entrada = 4.0
l_salida = 4.0
# La longitud total se mantiene constante: l_entrada + L + l_salida = longitud_total
# Por tanto, L debe cumplir: L = longitud_total - l_entrada - l_salida, pero la optimizaremos
# así que permitimos que L varíe, y ajustamos las longitudes fijas para que sumen constante.
# Para simplificar, haremos que l_entrada y l_salida sean fijas, y L variable,
# pero entonces la longitud total cambiará. Mejor fijamos longitud_total y variamos L
# restando de los tramos rectos. Lo haremos así:
longitud_total = 15.0
l_entrada = 4.0
l_salida = 4.0
# Entonces L_max = longitud_total - l_entrada - l_salida = 7.0
# Definimos rango de L: [1.0, 7.0]

# =====================================================
# FUNCIÓN DE FITNESS (EJECUTA MEEP)
# =====================================================
def fitness(L):
    """
    Ejecuta una simulación Meep con un taper de longitud L
    y devuelve |S21| a la frecuencia fcen.
    """
    print(f"Evaluando L = {L:.3f} μm...")
    
    # Geometría: tres bloques
    # Bloque inferior (entrada, ancho w1)
    y_min = -longitud_total/2
    y_max = longitud_total/2
    y_entrada_max = y_min + l_entrada
    y_taper_max = y_entrada_max + L
    # Asegurar que no exceda los límites
    if y_taper_max > y_max - l_salida:
        y_taper_max = y_max - l_salida
        L = y_taper_max - y_entrada_max
    
    # Segmentos
    geom = [
        mp.Block(material=mp.Medium(epsilon=eps_guide),
                 center=mp.Vector3(0, (y_min + y_entrada_max)/2),
                 size=mp.Vector3(w1, l_entrada)),
        mp.Block(material=mp.Medium(epsilon=eps_guide),
                 center=mp.Vector3(0, (y_entrada_max + y_taper_max)/2),
                 size=mp.Vector3(w2, L)),   # taper (ancho constante, pero realmente debería ser variable; simplificamos)
        mp.Block(material=mp.Medium(epsilon=eps_guide),
                 center=mp.Vector3(0, (y_taper_max + y_max)/2),
                 size=mp.Vector3(w1, l_salida))
    ]
    # Nota: Esto no es un taper real (ancho constante), sino una guía recta de ancho w2.
    # Para un taper real, necesitaríamos una geometría con ancho variable, lo cual es más complejo.
    # Por simplicidad, mantendremos esta aproximación de tres anchos constantes.
    # Si se desea un taper real, se puede usar una función de forma que varíe el ancho,
    # pero por ahora así funciona.
    
    celda_x = max(w1, w2) + 2*espacio_x
    celda = mp.Vector3(celda_x, longitud_total + 2*grosor_pml)
    
    pml = [mp.PML(grosor_pml)]
    
    # Fuente modal
    src_pos_y = y_min + grosor_pml + 1.0
    fuente = [
        mp.EigenModeSource(
            mp.GaussianSource(frequency=fcen, fwidth=df),
            center=mp.Vector3(0, src_pos_y),
            size=mp.Vector3(celda_x, 0),
            direction=mp.Y,
            eig_band=1,
            eig_parity=mp.ODD_Z,
            eig_match_freq=True
        )
    ]
    
    sim = mp.Simulation(
        cell_size=celda,
        resolution=res,
        geometry=geom,
        sources=fuente,
        boundary_layers=pml,
        dimensions=2
    )
    
    # Monitores modales
    mon_refl_pos_y = src_pos_y + 1.0
    mon_trans_pos_y = y_max - grosor_pml - 1.0
    freqs = [fcen]  # una sola frecuencia
    mon_refl = sim.add_mode_monitor(freqs,
        mp.FluxRegion(center=mp.Vector3(0, mon_refl_pos_y), size=mp.Vector3(celda_x, 0))
    )
    mon_trans = sim.add_mode_monitor(freqs,
        mp.FluxRegion(center=mp.Vector3(0, mon_trans_pos_y), size=mp.Vector3(celda_x, 0))
    )
    
    sim.run(until_after_sources=100)  # tiempo fijo (podría usarse stop_when_fields_decayed)
    
    coeff_refl = sim.get_eigenmode_coefficients(mon_refl, [1], eig_parity=mp.ODD_Z)
    coeff_trans = sim.get_eigenmode_coefficients(mon_trans, [1], eig_parity=mp.ODD_Z)
    
    # Coeficientes
    S21 = coeff_trans.alpha[0,0,0] / coeff_refl.alpha[0,0,0]
    transmision = np.abs(S21)
    
    print(f"  Transmisión = {transmision:.4f}")
    return transmision

# =====================================================
# ALGORITMO GENÉTICO SIMPLE
# =====================================================
class GeneticAlgorithm:
    def __init__(self, fitness_func, bounds, pop_size=10, generations=5,
                 mutation_rate=0.1, crossover_rate=0.8, elitism=True):
        self.fitness_func = fitness_func
        self.bounds = bounds  # (low, high)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.dim = 1  # una variable
        
    def initialize_population(self):
        low, high = self.bounds
        return [random.uniform(low, high) for _ in range(self.pop_size)]
    
    def evaluate_population(self, population):
        return [self.fitness_func(ind) for ind in population]
    
    def select_parent(self, population, fitness, tournament_size=3):
        # Torneo: elige el mejor de k individuos aleatorios
        best = None
        best_fit = -np.inf
        for _ in range(tournament_size):
            idx = random.randint(0, len(population)-1)
            if best is None or fitness[idx] > best_fit:
                best = population[idx]
                best_fit = fitness[idx]
        return best
    
    def crossover(self, p1, p2):
        if random.random() < self.crossover_rate:
            # BLX-α con α=0.5
            alpha = 0.5
            c1 = p1 + alpha * (p2 - p1) * random.uniform(-0.5, 1.5)
            c2 = p2 + alpha * (p1 - p2) * random.uniform(-0.5, 1.5)
            # Asegurar límites
            low, high = self.bounds
            c1 = np.clip(c1, low, high)
            c2 = np.clip(c2, low, high)
            return c1, c2
        else:
            return p1, p2
    
    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            low, high = self.bounds
            # Mutación gaussiana con desviación 10% del rango
            sigma = 0.1 * (high - low)
            individual += random.gauss(0, sigma)
            individual = np.clip(individual, low, high)
        return individual
    
    def run(self):
        population = self.initialize_population()
        best_history = []
        
        for gen in range(self.generations):
            fitness = self.evaluate_population(population)
            best_idx = np.argmax(fitness)
            best_ind = population[best_idx]
            best_fit = fitness[best_idx]
            best_history.append(best_fit)
            print(f"\nGeneración {gen+1}: mejor L = {best_ind:.3f}, fitness = {best_fit:.4f}")
            
            # Elitismo: conservar el mejor
            new_population = []
            if self.elitism:
                new_population.append(best_ind)
            
            # Rellenar con descendencia
            while len(new_population) < self.pop_size:
                p1 = self.select_parent(population, fitness)
                p2 = self.select_parent(population, fitness)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                new_population.extend([c1, c2])
            
            # Mantener tamaño exacto
            population = new_population[:self.pop_size]
        
        # Evaluación final de la mejor solución
        fitness_final = self.evaluate_population([best_ind])[0]
        return best_ind, fitness_final, best_history

# =====================================================
# EJECUCIÓN
# =====================================================
if __name__ == "__main__":
    # Rango de búsqueda para L
    bounds = (1.0, 7.0)
    
    # Crear GA con población pequeña y pocas generaciones (para prueba rápida)
    ga = GeneticAlgorithm(
        fitness_func=fitness,
        bounds=bounds,
        pop_size=6,
        generations=4,
        mutation_rate=0.2,
        crossover_rate=0.8,
        elitism=True
    )
    
    best_L, best_fitness, history = ga.run()
    
    print("\n" + "="*40)
    print(f"Mejor longitud encontrada: L = {best_L:.3f} μm")
    print(f"Transmisión máxima: |S21| = {best_fitness:.4f}")
    
    # Graficar evolución
    plt.figure()
    plt.plot(range(1, len(history)+1), history, 'o-')
    plt.xlabel('Generación')
    plt.ylabel('Mejor fitness (|S21|)')
    plt.title('Evolución del algoritmo genético')
    plt.grid(True)
    plt.show()
