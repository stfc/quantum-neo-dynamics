import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt


plt.style.use("proton.mplstyle")
bohr_to_angstrom = 0.529177210903

# Your 3 control points (in Bohr units)
orb_centers = np.array([
[-2.22176148,-0.18711600],
[-2.18129112, 0.00000000],
[-2.22176148, 0.18711600]
])
orb_centers /= bohr_to_angstrom
o_x = orb_centers[:,1]
o_y = orb_centers[:,0]

points = np.array([
    [-2.32501550,-0.19558330],
    [-2.28629872, 0.00000000],
    [-2.32501550, 0.19558330]
])
points /= bohr_to_angstrom

points = points[np.argsort(points[:, 1])]

# Create a smooth spline
xx = points[:, 1]
yy = points[:, 0]
spline = make_interp_spline(xx, yy, k=2)  # Quadratic spline

x_smooth = np.linspace(xx.min(), xx.max(), 300)
y_smooth = spline(x_smooth)

data=np.load('CASdenALL.npy')

nx, ny = 2000, 2000
x = np.linspace(-4.75, -3.95, nx)
y = np.linspace(-0.75, 0.75, ny)

# Plot image
fig, ax = plt.subplots(constrained_layout=True)
im = ax.imshow(data.T, extent=(y.min(), y.max(), x.min(), x.max()), origin='lower', cmap='inferno')

# Plot spline
ax.plot(x_smooth, y_smooth, color='white', linewidth=2)

ax.scatter(xx, yy, color='w', s=30, edgecolor='tab:blue', zorder=3)
ax.scatter(o_x, o_y, color='tab:blue', s=40, edgecolor='w', zorder=3)

ax.set_ylabel(r"$R_x$ [$\mathrm{\AA}$]")
ax.set_xlabel(r"$R_y$ [$\mathrm{\AA}$]")
xticks = ax.get_xticks()
#ax.set_xticks(xticks) 
ax.set_xticklabels([f"{tick * bohr_to_angstrom:.2f}" for tick in xticks])

ax.xaxis.set_ticks_position('top')       # Move ticks to top
ax.xaxis.set_label_position('top')       # Move label to top

ax.tick_params(axis='both', color="white")

# Convert and relabel y-axis ticks
yticks = ax.get_yticks()
#ax.set_yticks(yticks)  # <-- also this line
ax.set_yticklabels([f"{tick * bohr_to_angstrom:.2f}" for tick in yticks])

# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# import matplotlib.image as mpimg
#
# # Load and add image (e.g., arrow)
# img = mpimg.imread("ActiveSpace.png")  # replace with your image path
# imagebox = OffsetImage(img, zoom=0.08, interpolation='none')  # zoom controls image size
#
# ab = AnnotationBbox(
#     imagebox,
#     (0.5, -0.55),  # 50% across, slightly below axis
#     xycoords='axes fraction',
#     frameon=False
# )
# ax.add_artist(ab)
#
plt.colorbar(im, ax=ax, orientation='horizontal', label='Proton Density (Bohr$^{-3}$)', shrink=1)


plt.savefig("fig1.pdf")
plt.show()
