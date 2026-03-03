# -*- coding: utf-8 -*-
"""
Paso 2: Guía de onda recta en 2D (losas dieléctricas)
Simula una guía de onda rectangular de silicio (ε=12) en aire.
La fuente es un haz gaussiano enfocado en la entrada.
Se visualiza el campo Ez para observar el confinamiento.
"""

import meep as mp
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# PARÁMETROS DE LA SIMULACIÓN
# ============================================
res = 30                 # resolución (píxeles/μm)
ancho_gui = 2.0          # ancho de la guía (μm)
eps_gui = 12.0           # permitividad del silicio (índice ~3.46)
longitud = 20.0          # longitud de la guía (μm)
grosor_pml = 1.0         # grosor de las PML

# Dimensiones de la celda (con espacio alrededor de la guía)
# La guía está centrada en x=0, se extiende desde -longitud/2 a +longitud/2 en y
# Añadimos espacio en x para que el campo evanescente decaiga y en y para las PML
espacio_x = 4.0          # espacio a cada lado en x (μm)
celda_x = ancho_gui + 2 * espacio_x
celda_y = longitud + 2 * grosor_pml + 2.0  # margen extra para fuente/monitores
celda = mp.Vector3(celda_x, celda_y)

# ============================================
# DEFINICIÓN DE LA GEOMETRÍA
# ============================================
# La guía es un bloque rectangular centrado en x=0, abarcando toda la longitud en y
geometria = [mp.Block(
    material=mp.Medium(epsilon=eps_gui),
    center=mp.Vector3(0, 0),
    size=mp.Vector3(ancho_gui, longitud)
)]

# Capas PML en todos los bordes
pml = [mp.PML(grosor_pml)]

# ============================================
# FUENTE: HAZ GAUSSIANO (MODO APROXIMADO)
# ============================================
# Colocamos la fuente cerca del extremo inferior de la guía
# Usamos una fuente con perfil espacial gaussiano para excitar preferentemente el modo fundamental
frecuencia = 0.3          # frecuencia en 1/μm (longitud de onda ~3.33 μm)
ancho_banda = 0.1         # ancho de banda del pulso

# La fuente es una línea horizontal que abarca el ancho de la guía,
# con un perfil gaussiano en x para aproximar el modo fundamental.
# En Meep, se puede usar una fuente con función de distribución espacial.
# Aquí usamos un conjunto de fuentes puntuales con amplitudes Gaussianas,
# o más simple: una fuente plana con perfil gaussiano usando `amp_func`.
def perfil_gaussiano(pos):
    """Perfil gaussiano en x centrado en 0, con desviación típica = ancho_gui/4"""
    return np.exp(-(pos.x**2) / (2 * (ancho_gui/4)**2))

fuente = mp.Source(
    mp.GaussianSource(frequency=frecuencia, fwidth=ancho_banda),
    component=mp.Ez,
    center=mp.Vector3(0, -longitud/2 + 1.0),   # un poco dentro de la guía
    size=mp.Vector3(ancho_gui, 0),              # línea horizontal
    amp_func=perfil_gaussiano                    # perfil espacial
)

# ============================================
# CREACIÓN Y EJECUCIÓN DE LA SIMULACIÓN
# ============================================
sim = mp.Simulation(
    cell_size=celda,
    resolution=res,
    geometry=geometria,
    sources=[fuente],
    boundary_layers=pml,
    dimensions=2
)

# Tiempo de simulación suficiente para que el pulso recorra la guía
tiempo_sim = 200  # unidades de tiempo (c=1 → 200 μm de propagación)
sim.run(until=tiempo_sim)

# ============================================
# VISUALIZACIÓN DEL CAMPO ESTACIONARIO
# ============================================
# Extraemos el campo Ez en toda la celda
campo = sim.get_array(center=mp.Vector3(0,0), size=celda, component=mp.Ez)

# Creamos la figura
plt.figure(figsize=(8, 8))

# Mostramos la permitividad como fondo (en escala de grises)
eps_data = sim.get_array(center=mp.Vector3(0,0), size=celda, component=mp.Dielectric)
plt.imshow(eps_data.T, origin='lower', extent=[-celda_x/2, celda_x/2, -celda_y/2, celda_y/2],
           cmap='gray', alpha=0.5, label='ε')

# Superponemos el campo Ez
im = plt.imshow(campo.T, origin='lower', extent=[-celda_x/2, celda_x/2, -celda_y/2, celda_y/2],
                cmap='RdBu', alpha=0.7, aspect='equal')
plt.colorbar(im, label='Ez')

# Líneas que indican los bordes de la guía
plt.axvline(x=-ancho_gui/2, color='k', linestyle='--', linewidth=0.8)
plt.axvline(x= ancho_gui/2, color='k', linestyle='--', linewidth=0.8)

plt.title(f'Guía de onda recta (ε={eps_gui}, ancho={ancho_gui} μm, f={frecuencia} 1/μm)')
plt.xlabel('x (μm)')
plt.ylabel('y (μm)')
plt.xlim(-celda_x/2, celda_x/2)
plt.ylim(-celda_y/2, celda_y/2)
plt.show()
