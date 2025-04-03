import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# # Define Moon parameters
# R_moon = 1737e3  # Radius of the Moon (m)
# rho = 1800       # Density (kg/m^3)
# Cp = 840        # Specific heat (J/kg.K)
# k = 9.3e-3       # Thermal conductivity (W/m.K)
# alpha = k / (rho * Cp)  # Thermal diffusivity

# Cheese parameters
mass = 7.346e22  # Mass of moon (kg)
rho = 1120        # Density (kg/m^3)
R_moon = np.cbrt((3 * mass)/(4*rho*np.pi))  # Radius of the Moon (m)
Cp = 2444         # Specific heat (J/kg.K)
k = 0.354      # Thermal conductivity (W/m.K)
alpha = k / (rho * Cp)  # Thermal diffusivity

day_length = 29.53 * 24 * 3600  # Length of lunar day (s)
rot_freq = 2 * np.pi / day_length  # Angular frequency of lunar rotation
dr = 0.1  # Layer thickness

# Define grid resolution
N_lay = 10  # Number of layers
N_lat = 51  # Latitude divisions
N_lon = 100  # Longitude divisions
N_days = 160  # Number of moon days
N_time = (N_days * 240) + 1   # number of times steps
dt = 2.953 * 3600  # Time step (s) 1/240 moon days
# dt = day_length / 200  # Time step (sec)
# print(dt)
dx = R_moon * np.pi / N_lat  # Grid spacing
thickness = N_lay * dr # Thickness of the model (m)

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
T = np.repeat(T[np.newaxis, :, :], N_lay - 1, axis=0)
surf_T = 100 * np.sin(Th) * np.sin(Ph) + 250  # Surface temperature (K)
T = np.concatenate((surf_T[np.newaxis, :, :], T), axis=0)  # Add surface layer
# print(T[-1])
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
    for l in range (0, N_lay):
        for i in range(0, N_lat):
            for j in range(0, N_lon):
                #1D laplacian
                if l==0:
                    laplacian_T = (T[1, i, j] - T[l, i, j]) / dr**2 # no conductivity with vacuum
                      #T[1, i, j]
                    # print()
                    # if j == 49 and t % 240 == 0:
                    #     print(laplacian_T*alpha*dt)
                elif l == N_lay-1:
                    laplacian_T = (T[l-1, i, j] + T[l, i, j] - 2*T[l, i, j]) / dr**2
                else:
                    laplacian_T = (T[l+1, i, j] + T[l-1, i-1, j] - 2*T[l, i, j]) / dr**2
                # grad_T = np.sqrt((T[l, i+1, j] - T[l, i-1, j])**2 + (T[l, i, j+1] - T[l, i, j-1])**2) / (2 * dx)
                T_new[l, i, j] = T[l, i, j] + alpha * laplacian_T * dt
                # print(laplacian_T)
                # Solar heating (day side approximation)
                if l == 0:
                    sol_wave = np.sin(rot_freq * t * dt + Th[i,j])
                    if sol_wave > 0: # Only consider day side
                        S = S_max * np.sin(Ph[i, j]) * sol_wave  # solar flux
                    # print(S)
                    # print(Ph[i, j])
                        T_new[l, i, j] += (S * dt) / (rho * Cp * dr)  # Heat absorbed
                        # print(S * dt / (rho * Cp * thickness))
                    # Radiative cooling (Stefan-Boltzmann law)
                    emission = emissivity * sigma * T[l, i, j]**4 * dt / (rho * Cp * dr)
                    T_new[l, i, j] -= emission
                    if j == 49 and t % 240 == 0:
                        print(emission)
                    # print(emissivity * sigma * T[l, i, j]**4 * dt / (rho * Cp * thickness))
                T_new[l, i, j] = max(min(T_new[l, i, j], 500), 0)  # Keep within realistic lunar temperature limits
    
    # print(T_new[1, 1])
    if t != 0:
        if t % 240 == 0:
            print(T_new[0, :, :].max())
            print(T_new[0, :, :].min())
            print(np.abs(T_new[:, 12, :] - T_day[:, 12, :]).max())
            if np.all(np.abs(T_new[:, 12, :] - T_day[:, 12, :]) < 0.1):
                T = T_new.copy()
                break
            if np.any(T_new[0, :-1, :] == 0) or np.any(T_new[0, :, :] == 500):
                print("Temperature out of bounds")
                # T = T_day.copy()
                T = T_new.copy()
                break
            T_day = T_new.copy()
    T = T_new.copy()
# print(T)
# Plot results
temps = T[0, :, :] - 273.15  # Surface temperature (^oC)
shape = temps.shape
min_temp = temps[:-1, :].min()
max_temp = temps.max()

np.savetxt('temps.txt', temps, delimiter=',')

# Use below to make sphere same shape as temps, 
# Assuming axis 0 is phi (0 -> pi)
# axis 1 is theta (0 -> 2pi)
# If not, swap 0 & 1 below and the creation of x & y

""""
# # sphere
phi = np.linspace(0, np.pi, shape[0])
theta = np.linspace(0, 2*np.pi, shape[1])
T, P = np.meshgrid(theta, phi) # returns shape (len(phi), len(theta))
r = 1

# #cartesian coordinates
x = r * np.sin(P) * np.cos(T)
y = r * np.sin(P) * np.sin(T)
z = r * np.cos(P)

colour_map = mpl.colormaps['RdYlBu_r'] # Red to yellow to blue colour map

perf_temp = 25 # chosen temperature for the cheese

# Sets colour map to centre on perf_temp
if np.abs(perf_temp-max_temp) >= np.abs(perf_temp-min_temp):
    norm = mpl.colors.Normalize(vmin=(2*perf_temp)-max_temp, vmax=max_temp)
elif min_temp < 0:
    norm = mpl.colors.Normalize(vmin=min_temp, vmax=(2*perf_temp)-min_temp)

colours = colour_map(norm(temps))

plt.imshow(temps, cmap=colour_map, interpolation='nearest', origin='lower', extent=[0, 2*np.pi, 0, np.pi])
plt.colorbar(label='Temperature (°C)')
plt.savefig('cheeseMoon_temp.png', dpi=600, transparent=True)
# Plotting moon
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(x, y, z, norm=norm, edgecolor=None, linewidth=0.5, cmap=colour_map, facecolors=colours)
ax.set_axis_off()
# Adding colourbar
cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colour_map), ax=ax)
cbar.set_label('Temperature (°C)')
if np.abs(perf_temp-max_temp) >= np.abs(perf_temp-min_temp):  # Changes colourbar axis to match above
    cbar.set_ticks([(2*perf_temp)-max_temp, min_temp, -100, -50, 0, perf_temp, 50, 100, max_temp], labels=['{:.0f}'.format((2*perf_temp)-max_temp), '{:.0f}'.format(min_temp), '-100', '-50', '0', 'perfect', '50', '100', '{:.0f}'.format(max_temp)])
elif min_temp < 0:
    cbar.set_ticks([min_temp, -100, -50, 0, perf_temp, 50, 100, max_temp, (2*perf_temp)-min_temp], labels=['{:.0f}'.format(min_temp), '-100', '-50', '0', 'perfect', '50', '100', '{:.0f}'.format(max_temp), '{:.0f}'.format((2*perf_temp)-min_temp)])
# In set_ticks 
# First list is value positions, second is labels
# Indices have to match


ax.set_title('{:.2f} $\pi$ resolution'.format(1 / shape[0]))
ax.set_box_aspect((1, 1, 1)) # to be regular sphere
ax.view_init(30, 55, 0) # Change viewing angle https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.view_init.html#mpl_toolkits.mplot3d.axes3d.Axes3D.view_init
plt.savefig('cheeseMoon.png', dpi=600, transparent=True)
plt.show()

eq_index = shape[0] // 2 # Find equator index of temps

# Plot temps across equator
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(theta/np.pi, temps[eq_index], c='k')
ax1.spines[["left", "bottom"]].set_position(("data", 0)) # Puts x axis at y=0, can remove
ax1.spines[["top", "right"]].set_visible(False)
plt.savefig('cheeseMoon_eq.png', dpi=600, transparent=True)
plt.show()

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(theta/np.pi, temps[12], c='k')
ax1.spines[["left", "bottom"]].set_position(("data", 0)) # Puts x axis at y=0, can remove
ax1.spines[["top", "right"]].set_visible(False)
plt.savefig('cheeseMoon_N.png', dpi=600, transparent=True)
plt.show()

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(theta/np.pi, temps[37], c='k')
ax1.spines[["left", "bottom"]].set_position(("data", 0)) # Puts x axis at y=0, can remove
ax1.spines[["top", "right"]].set_visible(False)
plt.savefig('cheeseMoon_S.png', dpi=600, transparent=True)
plt.show()

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(phi/np.pi, temps[:, 24], c='k')
ax1.spines[["left", "bottom"]].set_position(("data", 0)) # Puts x axis at y=0, can remove
ax1.spines[["top", "right"]].set_visible(False)
plt.savefig('cheeseMoon_NS.png', dpi=600, transparent=True)
plt.show()

"""