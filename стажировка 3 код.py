import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

real_data = pd.read_csv(r"C:/Users/adilk/Downloads/AP56469959.csv")
real_data = real_data.rename(columns={'hjd': 'Time', 'mag': 'Magnitude', 'mag_err': 'Error'})

real_data = real_data.dropna()
real_data = real_data[real_data['Magnitude'] < 30]

mag0 = real_data['Magnitude'].min()
real_data['Flux'] = 10 ** (-0.4 * (real_data['Magnitude'] - mag0))

P_days = 217 * 2
real_data['Phase'] = (real_data['Time'] % P_days) / P_days

real_flux_min = real_data['Flux'].min()
real_flux_max = real_data['Flux'].max()

G = 6.67430e-11
M_sun = 1.98847e30
m1 = 17.0 * M_sun
m2 = 8.3 * M_sun
R1 = 7.5 * 6.957e8
R2 = 5.1 * 6.957e8
P = P_days * 24 * 3600  

a = (G * (m1 + m2) * (P / (2 * np.pi)) ** 2) ** (1 / 3)

r1_0 = np.array([-m2 / (m1 + m2) * a, 0])
v1_0 = np.array([0, -np.sqrt(G * m2**2 / (a * (m1 + m2)))])
r2_0 = np.array([m1 / (m1 + m2) * a, 0])
v2_0 = np.array([0, np.sqrt(G * m1**2 / (a * (m1 + m2)))])
y0 = np.concatenate([r1_0, v1_0, r2_0, v2_0])

def derivatives(t, y):
    r1 = y[:2]
    v1 = y[2:4]
    r2 = y[4:6]
    v2 = y[6:8]
    r = r2 - r1
    distance = np.linalg.norm(r)
    a1 = G * m2 * r / distance**3
    a2 = -G * m1 * r / distance**3
    return [*v1, *a1, *v2, *a2]

T = P * 2
num_points = 2000
t_span = (0, T)
t_eval = np.linspace(*t_span, num_points)

solution = solve_ivp(derivatives, t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
r1 = solution.y[:2]
r2 = solution.y[4:6]

brightness_raw = []
for i in range(len(t_eval)):
    x1, y1_star = r1[:, i]
    x2, y2_star = r2[:, i]
    dx = np.abs(x1 - x2)
    overlap = dx < (R1 + R2)
    if overlap:
        if y1_star > y2_star:
            total_brightness = 1.0
        else:
            total_brightness = 0.9
    else:
        total_brightness = 1.9
    brightness_raw.append(total_brightness)

brightness_raw = np.array(brightness_raw)
model_min = brightness_raw.min()
model_max = brightness_raw.max()
brightness_scaled = real_flux_min + (brightness_raw - model_min) * (real_flux_max - real_flux_min) / (model_max - model_min)

synthetic_phase = (t_eval / (3600 * 24) % P_days) / P_days

plt.figure(figsize=(10, 5))
plt.scatter(real_data['Phase'], real_data['Flux'], color='royalblue', s=10, label='Реальные данные (фаза)', alpha=0.6)
plt.plot(synthetic_phase, brightness_scaled, color='orange', label='Синтетическая модель', linewidth=2)
plt.title('Фазовая кривая блеска V Scorpii: реальные данные + модель (масштабировано)')
plt.xlabel('Фаза (0–1)')
plt.ylabel('Нормированная яркость')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
