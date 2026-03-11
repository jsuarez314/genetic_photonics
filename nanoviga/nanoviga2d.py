import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Parámetros (ajustables)
w = 0.461          # ancho de guía (μm)
a = 0.205          # periodo de agujeros (μm)
r = 0.060          # radio de agujeros (μm)
res = 30           # resolución (píxeles/μm)
dpml = 1.0         # grosor PML (μm)
celda_x = 4.0
celda_y = 8.0
fcen = 1 / 0.637   # frecuencia central (1/μm)
df = 0.2 * fcen    # ancho de banda

# Geometría
geometry = [mp.Block(material=mp.Medium(epsilon=4.0),
                     center=mp.Vector3(0, 0),
                     size=mp.Vector3(w, celda_y))]

y_min = -celda_y/2 + dpml
y_max = celda_y/2 - dpml
y_pos = np.arange(y_min, y_max, a)
centro = 0
dist_cav = 2 * a   # longitud de la cavidad (defecto)
for y in y_pos:
    if abs(y - centro) < dist_cav / 2:
        continue
    geometry.append(mp.Cylinder(radius=r,
                                center=mp.Vector3(0, y),
                                height=mp.inf,
                                material=mp.Medium(epsilon=1.0)))

cell = mp.Vector3(celda_x, celda_y)
pml = [mp.PML(dpml)]

source = mp.Source(mp.GaussianSource(frequency=fcen, fwidth=df),
                   component=mp.Ez,
                   center=mp.Vector3(0, centro))

sim = mp.Simulation(cell_size=cell,
                    resolution=res,
                    geometry=geometry,
                    sources=[source],
                    boundary_layers=pml,
                    dimensions=2)

# Monitor de flujo lineal (pequeña línea horizontal)
mon_pt = mp.Vector3(0, centro)
nfreq = 400
freq_min = fcen - 0.1
freq_max = fcen + 0.1
flux_region = mp.FluxRegion(center=mon_pt, size=mp.Vector3(0.1, 0), direction=mp.X)
flux_mon = sim.add_flux(freq_min, freq_max, nfreq, flux_region)

sim.run(until_after_sources=200)

# Obtener espectro y convertirlo a array NumPy
flux_data = np.array(mp.get_fluxes(flux_mon))
freqs = np.linspace(freq_min, freq_max, nfreq)

# Encontrar picos
peaks, _ = find_peaks(flux_data, height=0.01 * np.max(flux_data))
if len(peaks) > 0:
    # Índice del pico más alto
    idx = peaks[np.argmax(flux_data[peaks])]
    f_res = freqs[idx]
    half_max = flux_data[idx] / 2
    left = np.where(flux_data[:idx] <= half_max)[0]
    right = np.where(flux_data[idx:] <= half_max)[0]
    if len(left) > 0 and len(right) > 0:
        f_left = freqs[left[-1]]
        f_right = freqs[idx + right[0]]
        fwhm = f_right - f_left
        Q = f_res / fwhm
    else:
        Q = 0
else:
    Q = 0
    f_res = 0

print(f"Frecuencia de resonancia: {f_res:.5f} 1/μm")
print(f"Factor de calidad Q: {Q:.1f}")

# ============================================
# VISUALIZACIÓN DE LA GEOMETRÍA Y EL CAMPO
# ============================================

# Extraer permitividad y campo Ez en toda la celda
eps_data = sim.get_array(center=mp.Vector3(0,0), size=cell, component=mp.Dielectric)
ez_data = sim.get_array(center=mp.Vector3(0,0), size=cell, component=mp.Ez)

# Crear figura con dos subplots
plt.figure(figsize=(12, 5))

# Subplot 1: Permitividad (geometría)
plt.subplot(1, 2, 1)
plt.imshow(eps_data.T, origin='lower', extent=[-celda_x/2, celda_x/2, -celda_y/2, celda_y/2],
           cmap='gray', aspect='equal')
plt.colorbar(label='ε')
plt.title('Geometría (permitividad)')
plt.xlabel('x (μm)')
plt.ylabel('y (μm)')
plt.plot(0, centro, 'ro', markersize=5, label='Fuente')  # marcar posición de la fuente
plt.legend()

# Subplot 2: Campo Ez (intensidad)
plt.subplot(1, 2, 2)
plt.imshow(ez_data.T, origin='lower', extent=[-celda_x/2, celda_x/2, -celda_y/2, celda_y/2],
           cmap='RdBu', aspect='equal')
plt.colorbar(label='Ez')
plt.title('Campo eléctrico Ez')
plt.xlabel('x (μm)')
plt.ylabel('y (μm)')
plt.plot(0, centro, 'ro', markersize=5, label='Fuente')
plt.legend()

plt.tight_layout()
plt.show()

# También podemos mostrar el espectro (opcional)
plt.figure()
plt.plot(freqs, flux_data)
if len(peaks) > 0:
    plt.plot(f_res, flux_data[idx], 'ro')
plt.xlabel('Frecuencia (1/μm)')
plt.ylabel('Potencia (u.a.)')
plt.title('Espectro de potencia')
plt.grid(True)
plt.show()