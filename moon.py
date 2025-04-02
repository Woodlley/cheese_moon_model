import numpy as np
import matplotlib.pyplot as plt

# Define Moon parameters
R_moon = 1737e3  # Radius of the Moon (m)
rho = 1800       # Density (kg/m^3)
Cp = 840         # Specific heat (J/kg.K)
k = 9.3e-3       # Thermal conductivity (W/m.K)
alpha = k / (rho * Cp)  # Thermal diffusivity
day_length = 27.3 * 24 * 3600  # Length of lunar day (s)
rot_freq = 2 * np.pi / day_length  # Angular frequency of lunar rotation
thickness = 1  # Thickness of the model (m)

# Define grid resolution
N_lay = 10  # Number of layers
N_lat = 25  # Latitude divisions
N_lon = 50  # Longitude divisions
N_days = 10  # Number of moon days
N_time = N_days*240  # number of times steps
dt = 2.73 * 3600  # Time step (s) 1/240 moon days
# dt = day_length / 200  # Time step (sec)
# print(dt)
dx = R_moon * np.pi / N_lat  # Grid spacing
dr = thickness / N_lay  # Layer thickness

# dt_max = dx**2 / (4 * alpha)
# print(f"Recommended max dt: {dt_max}")
# if dt > dt_max:
#     print("Warning: Time step exceeds stability limit!")

# sphere
phi = np.linspace(0, np.pi, N_lat)
phi += 0.5 * np.pi / N_lat
theta = np.linspace(0, 2*np.pi, N_lon)
theta += 0.5 * np.pi / N_lon
Th, Ph = np.meshgrid(theta, phi) # returns shape (len(phi), len(theta))
# print(Ph)
# Initialize temperature array
T = 250 * np.ones_like(Th)  # Initial temperature (K)
T = np.stack([T] * N_lay, axis=0)  # Stack layers

# Define solar flux (simplified)
S_max = 1360.9  # Solar radiation at 1 AU (W/m^2)
emissivity = 0.95
sigma = 5.67e-8  # Stefan-Boltzmann constant

T_day = T.copy()

# Time evolution loop
for t in range(N_time):
    print(t)
    # Approximate Laplacian using finite difference
    T_new = T.copy()
    for l in range (0, N_lay-1):
        for i in range(0, N_lat):
            for j in range(0, N_lon):
                #1D laplacian
                if l==0:
                    laplacian_T = (0 + T[1, i, j] - 2*T[l, i, j]) / dr**2
                else:
                    laplacian_T = (T[l+1, i, j] + T[l-1, i-1, j] - 2*T[l, i, j]) / dr**2
                # grad_T = np.sqrt((T[l, i+1, j] - T[l, i-1, j])**2 + (T[l, i, j+1] - T[l, i, j-1])**2) / (2 * dx)
                T_new[l, i, j] = T[l, i, j] + alpha * laplacian_T * dt
                # print(laplacian_T)
                # Solar heating (day side approximation)
                if l == 0:
                    sol_wave = np.cos(rot_freq * t * dt)
                    if sol_wave > 0:  # Rough east/west division for sunlit side
                        S = S_max * np.sin(Ph[i, j]) * sol_wave  # solar flux
                    # print(S)
                    # print(Ph[i, j])
                        T_new[l, i, j] += (S * dt) / (rho * Cp * thickness)  # Heat absorbed

                    # Radiative cooling (Stefan-Boltzmann law)
                    T_new[l, i, j] -= emissivity * sigma * T[l, i, j]**4 * dt / (rho * Cp * thickness)
                T_new[l, i, j] = max(min(T_new[l, i, j], 500), -100)  # Keep within realistic lunar temperature limits
    
    # print(T_new[1, 1])
    if t != 0:
        if t % 240 == 0:
            print(T_new[0, 12, 25])
            print(np.abs(T_new[:, 12, :] - T_day[:, 12, :]).max())
            if np.all(np.abs(T_new[:, 12, :] - T_day[:, 12, :]) < 1):
                T = T_new.copy()
                break
            T_day = T_new.copy()
    T = T_new.copy()
# print(T)
# Plot results
plt.imshow(T[0], cmap='hot', aspect='auto')
plt.colorbar(label='Temperature (K)')
plt.title("Approximate Lunar Surface Temperature Distribution")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig("moon_temperature.png", dpi=300)
plt.show()
plt.plot(T[0, 12, :])
plt.savefig("moon_temperature_profile.png", dpi=300)
plt.show()
