import meep as mp
import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks

def fitness(params):
    a, r = params  # μm
    # Parámetros fijos (basados en Olthaus et al.)
    w = 0.461
    res = 20
    dpml = 1.0
    celda_x = 4.0
    celda_y = 8.0
    fcen = 1 / 0.637
    df = 0.2 * fcen

    # Geometría 2D
    geometry = [mp.Block(material=mp.Medium(epsilon=4.0),
                         center=mp.Vector3(0, 0),
                         size=mp.Vector3(w, celda_y))]

    y_min = -celda_y/2 + dpml
    y_max = celda_y/2 - dpml
    y_pos = np.arange(y_min, y_max, a)
    centro = 0
    dist_cav = 2 * a
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

    # Monitor de flujo lineal
    mon_pt = mp.Vector3(0, centro)
    nfreq = 400
    freq_min = fcen - 0.1
    freq_max = fcen + 0.1
    flux_region = mp.FluxRegion(center=mon_pt, size=mp.Vector3(0.1, 0), direction=mp.X)
    flux_mon = sim.add_flux(freq_min, freq_max, nfreq, flux_region)

    sim.run(until_after_sources=200)

    flux_data = np.array(mp.get_fluxes(flux_mon))
    freqs = np.linspace(freq_min, freq_max, nfreq)

    # Encontrar pico principal
    peaks, _ = find_peaks(flux_data, height=0.01 * np.max(flux_data))
    if len(peaks) == 0:
        return 0

    idx = peaks[np.argmax(flux_data[peaks])]
    f_res = freqs[idx]
    half_max = flux_data[idx] / 2
    left = np.where(flux_data[:idx] <= half_max)[0]
    right = np.where(flux_data[idx:] <= half_max)[0]
    if len(left) == 0 or len(right) == 0:
        return 0
    f_left = freqs[left[-1]]
    f_right = freqs[idx + right[0]]
    fwhm = f_right - f_left
    if fwhm <= 0:
        return 0
    Q = f_res / fwhm
    return -Q   # minimizar para maximizar Q

# Límites de búsqueda (en μm)
bounds = [(0.18, 0.23), (0.04, 0.08)]

print("Iniciando optimización...")
result = differential_evolution(fitness, bounds, maxiter=10, popsize=5, disp=True)

print("\n" + "="*40)
print("Mejor a = {:.5f} μm".format(result.x[0]))
print("Mejor r = {:.5f} μm".format(result.x[1]))
print("Mejor Q = {:.1f}".format(-result.fun))