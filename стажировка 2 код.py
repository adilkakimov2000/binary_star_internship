import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


G = 6.67430e-11


m1 = 17.0 * 1.989e30  
m2 = 8.3 * 1.989e30   

# Радиусы звёзд
R1 = 7.5 * 6.957e8
R2 = 5.1 * 6.957e8


r_orbit = 3.5e11

v_orbit = np.sqrt(G * (m1 + m2) / r_orbit)

r1 = np.array([-m2 / (m1 + m2) * r_orbit, 0])
r2 = np.array([m1 / (m1 + m2) * r_orbit, 0])

v1 = np.array([0, -m2 / (m1 + m2) * v_orbit])
v2 = np.array([0, m1 / (m1 + m2) * v_orbit])

y0 = np.concatenate((r1, v1, r2, v2))

def two_body(t, y):
    r1 = y[0:2]
    v1 = y[2:4]
    r2 = y[4:6]
    v2 = y[6:8]
    
    r = r2 - r1
    norm_r = np.linalg.norm(r)
    
    a1 = G * m2 * r / norm_r**3
    a2 = -G * m1 * r / norm_r**3
    
    return np.concatenate((v1, a1, v2, a2))

t_span = (0, 1.0e8)
t_eval = np.linspace(t_span[0], t_span[1], 5000)

sol = solve_ivp(two_body, t_span, y0, t_eval=t_eval, rtol=1e-8)

x1, y1 = sol.y[0], sol.y[1]
x2, y2 = sol.y[4], sol.y[5]


L1 = m1 ** 3.5
L2 = m2 ** 3.5

brightness = []

for i in range(len(t_eval)):

    x_star1 = x1[i]
    x_star2 = x2[i]

    distance = np.abs(x_star1 - x_star2)

    if distance < (R1 + R2):
        if y1[i] > y2[i]:
            visible_luminosity = L1 + max(0, L2 * (1 - (R1 / R2) ** 2))
        else:
            visible_luminosity = L2 + max(0, L1 * (1 - (R2 / R1) ** 2))
    else:
        visible_luminosity = L1 + L2

    brightness.append(visible_luminosity)

brightness = brightness / np.max(brightness)

plt.figure(figsize=(8, 4))
plt.plot(t_eval / (60 * 60 * 24), brightness, color='orange')
plt.title('Синтетическая кривая блеска системы V Scorpii')
plt.xlabel('Время (дни)')
plt.ylabel('Относительная яркость')
plt.grid()
plt.show()

