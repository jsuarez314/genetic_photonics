import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Parámetros (en μm)
t_sin = 0.2           # grosor SiN
t_sub = 1.0           # grosor del sustrato (suficiente para simular)
w = 0.461             # ancho de la viga
a = 0.205             # periodo de agujeros
r = 0.060             # radio de agujeros
res = 20              # resolución (píxeles/μm) - reducida para 3D
dpml = 1.0            # PML
fcen = 1 / 0.637      # frecuencia central (1/μm)
df = 0.2 * fcen       # ancho de banda

# Dimensiones de la celda
celda_x = 4.0
celda_y = 10.0
celda_z = t_sin + t_sub + 1.0   # espacio arriba
cell = mp.Vector3(celda_x, celda_y, celda_z)

# Materiales
SiN = mp.Medium(epsilon=4.0)
SiO2 = mp.Medium(epsilon=2.1)
air = mp.Medium(epsilon=1.0)

# PML en todas direcciones
pml_layers = [mp.PML(dpml)]

# Geometría
geometry = []

# Sustrato (bloque de SiO2 desde z = -t_sub hasta 0)
geometry.append(mp.Block(material=SiO2,
                         center=mp.Vector3(0, 0, -t_sub/2),
                         size=mp.Vector3(mp.inf, mp.inf, t_sub)))

# Nanoviga (bloque de SiN desde z=0 hasta t_sin)
geometry.append(mp.Block(material=SiN,
                         center=mp.Vector3(0, 0, t_sin/2),
                         size=mp.Vector3(w, mp.inf, t_sin)))

# Agujeros: cilindros de aire que atraviesan la viga (solo en la región de SiN)
y_min = -celda_y/2 + dpml
y_max = celda_y/2 - dpml
y_pos = np.arange(y_min, y_max, a)
centro = 0
dist_cav = 2 * a   # longitud de la cavidad (defecto)
for y in y_pos:
    if abs(y - centro) < dist_cav / 2:
        continue
    geometry.append(mp.Cylinder(material=air,
                                radius=r,
                                center=mp.Vector3(0, y, t_sin/2),
                                height=t_sin))

# Fuente: dipolo puntual con polarización en y (Ey) para excitar modos TE-like
source = mp.Source(mp.GaussianSource(frequency=fcen, fwidth=df),
                   component=mp.Ey,
                   center=mp.Vector3(0, centro, t_sin/2))

sim = mp.Simulation(cell_size=cell,
                    resolution=res,
                    geometry=geometry,
                    sources=[source],
                    boundary_layers=pml_layers,
                    dimensions=3)

# Monitor de flujo: línea horizontal en dirección x (longitud pequeña)
mon_pt = mp.Vector3(0, centro, t_sin/2)
nfreq = 400
freq_min = fcen - 0.1
freq_max = fcen + 0.1
# En 3D, una región de flujo lineal se define con size en una dirección y cero en las otras,
# y se especifica la dirección de la normal (perpendicular a la superficie).
# Para una línea horizontal en x, la normal puede ser en y o z; elegimos y.
flux_region = mp.FluxRegion(center=mon_pt, size=mp.Vector3(0.1, 0, 0), direction=mp.Y)
flux_mon = sim.add_flux(freq_min, freq_max, nfreq, flux_region)

sim.run(until_after_sources=300)

# Obtener espectro
flux_data = np.array(mp.get_fluxes(flux_mon))
freqs = np.linspace(freq_min, freq_max, nfreq)

# Encontrar picos
peaks, _ = find_peaks(flux_data, height=0.01 * np.max(flux_data))
if len(peaks) > 0:
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
# Extraer permitividad y campo Ey en un plano xy a la altura de la fuente (z = t_sin/2)
center_xy = mp.Vector3(0, 0, t_sin/2)
size_xy = mp.Vector3(celda_x, celda_y, 0)   # corte 2D en xy

eps_xy = sim.get_array(center=center_xy, size=size_xy, component=mp.Dielectric)
ey_xy = sim.get_array(center=center_xy, size=size_xy, component=mp.Ey)

# Crear figura
plt.figure(figsize=(12, 5))

# Subplot 1: Permitividad
plt.subplot(1, 2, 1)
plt.imshow(eps_xy.T, origin='lower', extent=[-celda_x/2, celda_x/2, -celda_y/2, celda_y/2],
           cmap='gray', aspect='equal')
plt.colorbar(label='ε')
plt.title(f'Geometría (plano xy, z = {t_sin/2:.2f} μm)')
plt.xlabel('x (μm)')
plt.ylabel('y (μm)')
plt.plot(0, centro, 'ro', markersize=5, label='Fuente')
plt.legend()

# Subplot 2: Campo Ey
plt.subplot(1, 2, 2)
plt.imshow(ey_xy.T, origin='lower', extent=[-celda_x/2, celda_x/2, -celda_y/2, celda_y/2],
           cmap='RdBu', aspect='equal')
plt.colorbar(label='Ey')
plt.title('Campo eléctrico Ey')
plt.xlabel('x (μm)')
plt.ylabel('y (μm)')
plt.plot(0, centro, 'ro', markersize=5, label='Fuente')
plt.legend()

plt.tight_layout()
plt.show()

# También mostrar el espectro
plt.figure()
plt.plot(freqs, flux_data)
if len(peaks) > 0:
    plt.plot(f_res, flux_data[idx], 'ro')
plt.xlabel('Frecuencia (1/μm)')
plt.ylabel('Potencia (u.a.)')
plt.title('Espectro de potencia (3D)')
plt.grid(True)
plt.show()