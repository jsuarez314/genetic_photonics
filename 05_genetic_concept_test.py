# -*- coding: utf-8 -*-
"""
Optimización del factor de Purcell para una fuente puntual en una cavidad
fotónica 1D (espejos de Bragg) mediante algoritmo genético.

Parámetros libres:
  - Lcav : longitud de la cavidad (distancia entre espejos) [μm]
  - dx   : desplazamiento de la fuente desde el centro de la cavidad [μm]

Restricción: la fuente debe estar al menos a 0.1 μm de los espejos (para evitar
que el monitor de flujo incluya los espejos). Si no se cumple, fitness = 0.

El factor de Purcell se calcula como:
    Fp = P_cavidad / P_bulk
donde P_cavidad es la potencia emitida por el dipolo en la cavidad,
y P_bulk es la potencia que emitiría el mismo dipolo en un medio homogéneo
(con la misma permitividad que el material de la cavidad).

Se incluye visualización final de la geometría de la mejor cavidad.
"""

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import random

# =====================================================
# PARÁMETROS FIJOS DEL PROBLEMA
# =====================================================
res = 30                    # resolución (píxeles/μm)
eps_alto = 12.0             # permitividad del material de alto índice (ej. silicio)
eps_bajo = 1.0              # permitividad del material de bajo índice (aire)
periodo = 0.3               # periodo de los agujeros/espejos (μm)
ancho_bloque = periodo / 2   # ancho de cada bloque de alto índice (factor de llenado 0.5)
n_pares = 5                 # número de pares de bloques en cada espejo
grosor_pml = 1.0            # PML
celda_x = 12.0              # tamaño fijo de la celda en x (μm)
celda_y = 8.0               # tamaño fijo de la celda en y (μm)

# Rango de búsqueda para los parámetros
bounds_L = (0.5, 2.0)       # longitud de cavidad entre 0.5 y 2.0 μm
bounds_dx = (-0.5, 0.5)     # desplazamiento máximo ±0.5 μm

# Frecuencia central del dipolo (debe coincidir con el modo de la cavidad)
fcen = 0.3                  # frecuencia en 1/μm (longitud de onda ~3.33 μm)
df = 0.05                   # ancho de banda (para pulso gaussiano)

# =====================================================
# FUNCIÓN PARA CREAR LA GEOMETRÍA DE LA CAVIDAD
# =====================================================
def crear_cavidad(Lcav, dx):
    """
    Crea una cavidad 1D con espejos de Bragg formados por bloques de alto índice.
    La cavidad es una región sin bloques de longitud Lcav centrada en x=0.
    La fuente se colocará en (dx, 0).
    Retorna: lista de geometría y la posición de la fuente.
    """
    geometria = []
    
    # Los bloques tienen un ancho en y (la guía es un slab de ancho fijo)
    ancho_gui = 0.5  # μm
    
    # Espejo izquierdo: desde x = -Lcav/2 - n_pares*periodo hasta x = -Lcav/2
    for i in range(n_pares):
        centro_x = -Lcav/2 - (i + 0.5) * periodo
        geometria.append(
            mp.Block(
                material=mp.Medium(epsilon=eps_alto),
                center=mp.Vector3(centro_x, 0),
                size=mp.Vector3(ancho_bloque, ancho_gui)
            )
        )
    
    # Espejo derecho: desde x = Lcav/2 hasta x = Lcav/2 + n_pares*periodo
    for i in range(n_pares):
        centro_x = Lcav/2 + (i + 0.5) * periodo
        geometria.append(
            mp.Block(
                material=mp.Medium(epsilon=eps_alto),
                center=mp.Vector3(centro_x, 0),
                size=mp.Vector3(ancho_bloque, ancho_gui)
            )
        )
    
    # La cavidad es la región entre -Lcav/2 y Lcav/2, que queda vacía (aire)
    # Posición de la fuente
    pos_fuente = mp.Vector3(dx, 0)
    
    return geometria, pos_fuente

# =====================================================
# FUNCIÓN PARA VISUALIZAR LA GEOMETRÍA
# =====================================================
def visualizar_geometria(geometria, pos_fuente, Lcav, dx, titulo="Geometría de la cavidad"):
    """
    Dibuja la permitividad de la geometría y marca la posición de la fuente.
    """
    # Crear una simulación temporal solo para obtener el arreglo de epsilon
    celda = mp.Vector3(celda_x, celda_y)
    sim = mp.Simulation(
        cell_size=celda,
        resolution=res,
        geometry=geometria,
        boundary_layers=[mp.PML(grosor_pml)],
        dimensions=2
    )
    sim.init_sim()
    eps_data = sim.get_array(center=mp.Vector3(0,0), size=celda, component=mp.Dielectric)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(eps_data.T, origin='lower', extent=[-celda_x/2, celda_x/2, -celda_y/2, celda_y/2],
               cmap='binary', aspect='equal')
    plt.colorbar(label='ε')
    plt.plot(pos_fuente.x, pos_fuente.y, 'ro', markersize=8, label='Fuente')
    plt.axvline(x=-Lcav/2, color='b', linestyle='--', linewidth=1, label='Límites cavidad')
    plt.axvline(x= Lcav/2, color='b', linestyle='--', linewidth=1)
    plt.xlabel('x (μm)')
    plt.ylabel('y (μm)')
    plt.title(titulo)
    plt.legend()
    plt.xlim(-celda_x/2, celda_x/2)
    plt.ylim(-celda_y/2, celda_y/2)
    plt.show()

# =====================================================
# FUNCIÓN DE FITNESS (CALCULA FACTOR DE PURCELL)
# =====================================================
def fitness(params):
    """
    params = [Lcav, dx]
    Ejecuta simulación Meep y retorna el factor de Purcell Fp.
    Incluye restricción: fuente no debe estar demasiado cerca de los espejos.
    """
    Lcav, dx = params
    
    # --- Restricción: distancia mínima a los espejos ---
    dist_izq = abs(dx + Lcav/2)
    dist_der = abs(dx - Lcav/2)
    if dist_izq < 0.1 or dist_der < 0.1:
        #print(f"  Violación: fuente demasiado cerca del espejo (izq={dist_izq:.3f}, der={dist_der:.3f}) -> fitness = 0")
        return 0.0
    
    #print(f"Evaluando: Lcav = {Lcav:.3f} μm, dx = {dx:.3f} μm")
    
    # Crear geometría
    geometria, pos_fuente = crear_cavidad(Lcav, dx)
    
    # Celda fija
    celda = mp.Vector3(celda_x, celda_y)
    pml = [mp.PML(grosor_pml)]
    
    # Fuente dipolar
    fuente = mp.Source(
        mp.GaussianSource(frequency=fcen, fwidth=df),
        component=mp.Ez,
        center=pos_fuente
    )
    
    # Simulación con cavidad
    sim = mp.Simulation(
        cell_size=celda,
        resolution=res,
        geometry=geometria,
        sources=[fuente],
        boundary_layers=pml,
        dimensions=2
    )
    
    # Monitor de flujo: cuadrado alrededor de la fuente que se adapta al tamaño de la cavidad
    dist_min = min(dist_izq, dist_der)
    tam_monitor = min(0.4, dist_min * 0.8)
    if tam_monitor < 0.1:
        tam_monitor = 0.1
    
    regiones = [
        mp.FluxRegion(center=pos_fuente + mp.Vector3(0, tam_monitor/2), size=mp.Vector3(tam_monitor, 0)),
        mp.FluxRegion(center=pos_fuente + mp.Vector3(0, -tam_monitor/2), size=mp.Vector3(tam_monitor, 0)),
        mp.FluxRegion(center=pos_fuente + mp.Vector3(tam_monitor/2, 0), size=mp.Vector3(0, tam_monitor)),
        mp.FluxRegion(center=pos_fuente + mp.Vector3(-tam_monitor/2, 0), size=mp.Vector3(0, tam_monitor))
    ]
    
    flujo_mon = sim.add_flux(fcen, df, 1, *regiones)
    
    # Ejecutar hasta que el campo decaiga
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pos_fuente, 1e-6))
    
    # Potencia emitida en la cavidad (suma sobre frecuencias, solo una)
    P_cav = np.sum(mp.get_fluxes(flujo_mon))
    
    # Simulación en bulk (medio homogéneo con permitividad eps_alto)
    sim_bulk = mp.Simulation(
        cell_size=celda,
        resolution=res,
        sources=[fuente],
        boundary_layers=pml,
        dimensions=2,
        default_material=mp.Medium(epsilon=eps_alto)
    )
    flujo_mon_bulk = sim_bulk.add_flux(fcen, df, 1, *regiones)
    sim_bulk.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pos_fuente, 1e-6))
    P_bulk = np.sum(mp.get_fluxes(flujo_mon_bulk))
    
    # Factor de Purcell
    Fp = P_cav / P_bulk
    #print(f"  P_cav = {P_cav:.3e}, P_bulk = {P_bulk:.3e}, Fp = {Fp:.3f}")
    return Fp

# =====================================================
# ALGORITMO GENÉTICO (para dos parámetros)
# =====================================================
class GeneticAlgorithm2D:
    def __init__(self, fitness_func, bounds1, bounds2, pop_size=20, generations=10,
                 mutation_rate=0.1, crossover_rate=0.8, elitism=True):
        self.fitness_func = fitness_func
        self.bounds = [bounds1, bounds2]
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.dim = 2
        
    def initialize_population(self):
        pop = []
        for _ in range(self.pop_size):
            ind = [random.uniform(b[0], b[1]) for b in self.bounds]
            pop.append(ind)
        return pop
    
    def evaluate_population(self, population):
        return [self.fitness_func(ind) for ind in population]
    
    def select_parent(self, population, fitness, tournament_size=3):
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
            c1 = []
            c2 = []
            for i in range(self.dim):
                low = min(p1[i], p2[i])
                high = max(p1[i], p2[i])
                range_i = high - low
                c1_i = p1[i] + alpha * (p2[i] - p1[i]) * random.uniform(-0.5, 1.5)
                c2_i = p2[i] + alpha * (p1[i] - p2[i]) * random.uniform(-0.5, 1.5)
                # Clip a bounds
                c1_i = np.clip(c1_i, self.bounds[i][0], self.bounds[i][1])
                c2_i = np.clip(c2_i, self.bounds[i][0], self.bounds[i][1])
                c1.append(c1_i)
                c2.append(c2_i)
            return c1, c2
        else:
            return p1.copy(), p2.copy()
    
    def mutate(self, individual):
        for i in range(self.dim):
            if random.random() < self.mutation_rate:
                low, high = self.bounds[i]
                sigma = 0.1 * (high - low)
                individual[i] += random.gauss(0, sigma)
                individual[i] = np.clip(individual[i], low, high)
        return individual
    
    def run(self):
        population = self.initialize_population()
        best_history = []
        best_ind_global = None
        best_fit_global = -np.inf
        
        for gen in range(self.generations):
            fitness = self.evaluate_population(population)
            best_idx = np.argmax(fitness)
            best_ind = population[best_idx]
            best_fit = fitness[best_idx]
            best_history.append(best_fit)
            
            if best_fit > best_fit_global:
                best_fit_global = best_fit
                best_ind_global = best_ind.copy()
            
            print(f"\nGeneración {gen+1}/{self.generations}: mejor Lcav={best_ind[0]:.3f}, dx={best_ind[1]:.3f}, Fp={best_fit:.3f}")
            
            new_population = []
            if self.elitism:
                new_population.append(best_ind)
            
            while len(new_population) < self.pop_size:
                p1 = self.select_parent(population, fitness)
                p2 = self.select_parent(population, fitness)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                new_population.extend([c1, c2])
            
            population = new_population[:self.pop_size]
        
        return best_ind_global, best_fit_global, best_history

# =====================================================
# EJECUCIÓN
# =====================================================
if __name__ == "__main__":
    ga = GeneticAlgorithm2D(
        fitness_func=fitness,
        bounds1=bounds_L,
        bounds2=bounds_dx,
        pop_size=20,
        generations=10,
        mutation_rate=0.2,
        crossover_rate=0.8,
        elitism=True
    )
    
    best_params, best_fp, history = ga.run()
    
    print("\n" + "="*40)
    print(f"Mejor parámetros encontrados:")
    print(f"  Lcav = {best_params[0]:.3f} μm")
    print(f"  dx   = {best_params[1]:.3f} μm")
    print(f"Factor de Purcell máximo: Fp = {best_fp:.3f}")
    
    # Gráfico de evolución
    plt.figure()
    plt.plot(range(1, len(history)+1), history, 'o-')
    plt.xlabel('Generación')
    plt.ylabel('Mejor Fp')
    plt.title('Evolución del factor de Purcell')
    plt.grid(True)
    plt.show()
    
    # Visualizar la geometría de la mejor cavidad
    mejor_geometria, mejor_pos_fuente = crear_cavidad(best_params[0], best_params[1])
    visualizar_geometria(mejor_geometria, mejor_pos_fuente, best_params[0], best_params[1],
                         titulo=f"Mejor cavidad: Lcav={best_params[0]:.3f} μm, dx={best_params[1]:.3f} μm") 
