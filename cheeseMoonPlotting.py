import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# sphere
# phi = np.linspace(0, np.pi, 25)
# theta = np.linspace(0, 2*np.pi, 50)
# T, P = np.meshgrid(theta, phi) # returns shape (len(phi), len(theta))
# r = 1

# #cartesian coordinates
# x = r * np.sin(P) * np.cos(T)
# y = r * np.sin(P) * np.sin(T)
# z = r * np.cos(P)

# Replace with temperatures from analysis
temps = np.genfromtxt('temps.txt', delimiter=',')
min_temp = temps.min()
max_temp = temps.max()

shape = temps.shape

# Use below to make sphere same shape as temps, 
# Assuming axis 0 is phi (0 -> pi)
# axis 1 is theta (0 -> 2pi)
# If not, swap 0 & 1 below and the creation of x & y

# sphere
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
ax1.set_ylabel('Temperature (°C)')
ax1.set_xlabel('Longitude ($\pi$ radians)')
ax1.set_title('Temperature around the equator')
plt.savefig('cheeseMoon_eq.png', dpi=600, transparent=True)
plt.show()

counts = np.sum((temps >= perf_temp - 5) & (temps <= perf_temp + 5), axis=1) # Count number of temps around perf_temp
row_index = np.argmax(counts) # Find row with most temps around perf_temp
print(row_index)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(theta/np.pi, temps[row_index], c='k')
ax2.fill_between(theta/np.pi, temps[row_index], 20, where=(temps[row_index] >= 20), color='k', alpha=0.2)
ax2.fill_between(theta/np.pi, temps[row_index], 30, where=((temps[row_index] <= 30) & (temps[row_index] >= 20)), color='k', alpha=0.2)
ax2.set_title('Latitude with longest time around {:.0f}°C. Latitude = {:.2f}$\pi$'.format(perf_temp, (0.5 + row_index - 25)/shape[0]))
# plt.fill_between([0, 2], 20, 30, color='k', alpha=0.2, label='Error Band')
ax2.spines[["left", "bottom"]].set_position(("data", 0)) # Puts x axis at y=0, can remove
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_ylabel('Temperature (°C)')
ax2.set_xlabel('Longitude ($\pi$ radians)')
plt.savefig('cheeseMoon_row.png', dpi=600, transparent=True)
plt.show()
