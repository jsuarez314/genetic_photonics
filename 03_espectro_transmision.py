# -*- coding: utf-8 -*-
"""
Paso 3: Espectro de transmisión (parámetros S) de una guía de onda recta.
Se utiliza una fuente modal para excitar el modo TE fundamental y monitores
modales para calcular S11 y S21 en un rango de frecuencias.
"""

import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# PARÁMETROS FIJOS Y GEOMÉTRICOS
# ============================================
res = 30                  # resolución (píxeles/μm)
ancho_gui = 2.0           # ancho de la guía (μm)
eps_gui = 12.0            # permitividad del silicio
longitud = 20.0           # longitud de la guía (μm)
grosor_pml = 1.0          # grosor de las PML
espacio_x = 4.0           # espacio a cada lado en x (μm)

# Dimensiones de la celda
celda_x = ancho_gui + 2 * espacio_x
celda_y = longitud + 2 * grosor_pml + 2.0  # margen extra
celda = mp.Vector3(celda_x, celda_y)

# Geometría: guía recta
geometria = [mp.Block(
    material=mp.Medium(epsilon=eps_gui),
    center=mp.Vector3(0, 0),
    size=mp.Vector3(ancho_gui, longitud)
)]

# Capas PML
pml = [mp.PML(grosor_pml)]

# ============================================
# PARÁMETROS DE LA FUENTE Y MONITORES
# ============================================
# Rango de frecuencias de interés (en 1/μm)
fcen = 0.3                # frecuencia central
df = 0.2                  # ancho de banda total
nfreq = 100               # número de frecuencias a muestrear
freqs = np.linspace(fcen - df/2, fcen + df/2, nfreq)

# Fuente modal: excita el modo TE fundamental (banda 1) en la dirección +Y
# Se coloca cerca del extremo inferior de la guía.
src_pos_y = -longitud/2 + 2.0
fuente = [
    mp.EigenModeSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=df),
        center=mp.Vector3(0, src_pos_y),
        size=mp.Vector3(ancho_gui + 1.0, 0),  # un poco más ancho que la guía
        direction=mp.Y,                        # propagación en +Y
        eig_band=1,                             # modo fundamental
        eig_parity=mp.ODD_Z,                    # simetría TE en 2D (Ez impar en x)
        eig_match_freq=True                      # ajusta el k-vector a la frecuencia
    )
]

# ============================================
# CONFIGURACIÓN DE LA SIMULACIÓN
# ============================================
sim = mp.Simulation(
    cell_size=celda,
    resolution=res,
    geometry=geometria,
    sources=fuente,
    boundary_layers=pml,
    dimensions=2
)

# Monitores modales:
# - Reflexión: antes de la discontinuidad (justo después de la fuente)
mon_refl_pos_y = src_pos_y + 1.0
mon_refl = sim.add_mode_monitor(
    freqs,
    mp.FluxRegion(center=mp.Vector3(0, mon_refl_pos_y), size=mp.Vector3(ancho_gui + 1.0, 0))
)

# - Transmisión: al final de la guía (antes de la PML)
mon_trans_pos_y = longitud/2 - 1.5
mon_trans = sim.add_mode_monitor(
    freqs,
    mp.FluxRegion(center=mp.Vector3(0, mon_trans_pos_y), size=mp.Vector3(ancho_gui + 1.0, 0))
)

# ============================================
# EJECUCIÓN DE LA SIMULACIÓN
# ============================================
print("Iniciando simulación...")
sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(0, 0), 1e-6))
print("Simulación finalizada.")

# ============================================
# PROCESAMIENTO DE RESULTADOS
# ============================================
# Coeficientes modales en el monitor de reflexión
coeff_refl = sim.get_eigenmode_coefficients(
    mon_refl,
    [1],                # banda 1 (modo fundamental)
    eig_parity=mp.ODD_Z
)

# Coeficientes modales en el monitor de transmisión
coeff_trans = sim.get_eigenmode_coefficients(
    mon_trans,
    [1],
    eig_parity=mp.ODD_Z
)

# Frecuencias reales de los monitores (pueden diferir ligeramente)
freqs_obtenidas = mp.get_flux_freqs(mon_refl)

# Los coeficientes tienen forma (nfreq, 1, 2): (frecuencia, banda, dirección)
# dirección 0 = +Y, dirección 1 = -Y
S11 = coeff_refl.alpha[:, 0, 1] / coeff_refl.alpha[:, 0, 0]   # reflexión
S21 = coeff_trans.alpha[:, 0, 0] / coeff_refl.alpha[:, 0, 0]  # transmisión

# ============================================
# VISUALIZACIÓN
# ============================================
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(freqs_obtenidas, np.abs(S11), 'r-', linewidth=2, label='|S11| (reflexión)')
plt.plot(freqs_obtenidas, np.abs(S21), 'b-', linewidth=2, label='|S21| (transmisión)')
plt.xlabel('Frecuencia (1/μm)')
plt.ylabel('Magnitud')
plt.legend()
plt.grid(True)
plt.title('Coeficientes de scattering - Guía recta')
plt.ylim(0, 1.1)

plt.subplot(2, 1, 2)
plt.plot(freqs_obtenidas, np.angle(S11), 'r-', linewidth=2, label='arg(S11)')
plt.plot(freqs_obtenidas, np.angle(S21), 'b-', linewidth=2, label='arg(S21)')
plt.xlabel('Frecuencia (1/μm)')
plt.ylabel('Fase (radianes)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Cálculo de pérdidas (absorción + radiación)
perdidas = 1 - np.abs(S11)**2 - np.abs(S21)**2
plt.figure()
plt.plot(freqs_obtenidas, perdidas, 'k-', linewidth=2, label='Pérdidas')
plt.xlabel('Frecuencia (1/μm)')
plt.ylabel('1 - |S11|² - |S21|²')
plt.grid(True)
plt.title('Pérdidas totales')
plt.ylim(-0.1, 0.5)  # ajustar según resultados
plt.show()
