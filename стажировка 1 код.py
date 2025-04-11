import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

G = 6.67430e-11  

m1 = 17.0 * 1.989e30  
m2 = 8.3 * 1.989e30   

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

plt.figure(figsize=(8, 6))
plt.plot(x1, y1, label='Звезда A (V Scorpii A)')
plt.plot(x2, y2, label='Звезда B (V Scorpii B)')
plt.scatter([x1[0], x2[0]], [y1[0], y2[0]], color='red', marker='o', label='Начальное положение')
plt.xlabel('x [м]')
plt.ylabel('y [м]')
plt.legend()
plt.title('Орбиты системы V Scorpii')
plt.grid()
plt.axis('equal')
plt.show()
