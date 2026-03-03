# -*- coding: utf-8 -*-
"""
Simulación de propagación de onda desde una fuente puntual en 2D con Meep.
Genera un GIF animado de la evolución del campo Ez.
"""

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ============================================
# PARÁMETROS DE LA SIMULACIÓN (ajustables)
# ============================================
resolucion = 30          # píxeles/μm (mayor resolución reduce dispersión numérica)
tam_celda = 16.0         # tamaño de la celda cuadrada en μm
grosor_pml = 2.0         # grosor de las capas PML (mejor absorción)
frec_centro = 0.3        # frecuencia central de la fuente
ancho_banda = 0.1        # ancho de banda de la fuente
tiempo_total = 100        # tiempo total de simulación (suficiente para ver propagación sin muchas reflexiones)
intervalo_tiempo = 1.0   # intervalo entre capturas de campo (para la animación)

# ============================================
# CONFIGURACIÓN DE LA CELDA Y FUENTE
# ============================================
celda = mp.Vector3(tam_celda, tam_celda)
pml = [mp.PML(grosor_pml)]

# Fuente: dipolo puntual gaussiano (componente Ez)
fuente = mp.Source(
    mp.GaussianSource(frequency=frec_centro, fwidth=ancho_banda),
    component=mp.Ez,
    center=mp.Vector3(0, 0)
)

# Crear simulación
sim = mp.Simulation(
    cell_size=celda,
    resolution=resolucion,
    sources=[fuente],
    boundary_layers=pml
)

# ============================================
# EJECUCIÓN PASO A PASO Y CAPTURA DE CAMPOS
# ============================================
# Inicializar lista para almacenar los campos en cada instante
campos_t = []
tiempos = []

# Tiempo actual
t_actual = 0

# Bucle de captura
while t_actual < tiempo_total:
    # Avanzar la simulación hasta el próximo instante de captura
    sim.run(until=intervalo_tiempo)
    t_actual += intervalo_tiempo

    # Obtener el campo Ez en toda la celda
    campo = sim.get_array(center=mp.Vector3(0,0), size=celda, component=mp.Ez)
    campos_t.append(campo)
    tiempos.append(t_actual)

    print(f"Tiempo {t_actual:.1f} / {tiempo_total} capturado")

# ============================================
# CREAR ANIMACIÓN
# ============================================
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-tam_celda/2, tam_celda/2)
ax.set_ylim(-tam_celda/2, tam_celda/2)
ax.set_xlabel('x (μm)')
ax.set_ylabel('y (μm)')
ax.set_title('Propagación de onda - Campo Ez')

# Determinar rango de valores para escala de colores fija (opcional)
vmin = min(np.min(campo) for campo in campos_t)
vmax = max(np.max(campo) for campo in campos_t)

im = ax.imshow(campos_t[0].T, origin='lower',
               extent=[-tam_celda/2, tam_celda/2, -tam_celda/2, tam_celda/2],
               cmap='RdBu', aspect='equal', vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax, label='Ez')

def actualizar(frame):
    im.set_data(campos_t[frame].T)
    ax.set_title(f'Tiempo = {tiempos[frame]:.1f} μm/c')  # c=1 en unidades Meep
    return [im]

ani = animation.FuncAnimation(fig, actualizar, frames=len(campos_t),
                              interval=200, blit=True)

# Mostrar la animación (si se ejecuta en Jupyter, usar %matplotlib notebook)
plt.show()

# Guardar como GIF (requiere pillow)
ani.save('propagacion_onda.gif', writer='pillow', fps=5)
print("GIF guardado como 'propagacion_onda.gif'")
