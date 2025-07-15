import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Matplotlib style settings
plt.style.use("proton.mplstyle")

titles = ['300', '210', '120', '030', '021', '012', '003']
methods = ['CAS', 'ADA', 'AQC']
bohr_to_angstrom = 0.529177210903

nx, ny = 2000, 2000
x = np.linspace(-4.75, -3.95, nx) * bohr_to_angstrom
y = np.linspace(-0.75, 0.75, ny) * bohr_to_angstrom
X, Y = np.meshgrid(x, y)


fig, axes = plt.subplots(len(titles), len(methods),
                         figsize=(3.321 * 2, 3.0 * 2.5),
                         sharex=True, sharey=True,
                         constrained_layout=True)

axes = np.atleast_2d(axes)

for i, title in enumerate(titles):
    for j, method in enumerate(methods):
        ax = axes[i, j]

        data = np.load("../"+method+'den'+title+'.npy')

        im = ax.imshow(
            data.T,
            extent=(y.min(), y.max(), x.min(), x.max()),
            origin='lower',
            cmap='inferno',
            # norm=LogNorm(vmin=1e-1)
        )

        # Add title label for left column
        if j == 0:
            data_casci = data
            ax.contour(Y, X, data_casci, [10], colors="white", linestyles="--")
            ax.set_ylabel(r"$R_x$ ($\mathrm{\AA}$)")

            # Add internal title
            if i < 4:
                ax.text(0.95, 0.95, title, transform=ax.transAxes, fontsize="medium", ha='right', va='top', color='white')
            else:
                ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize="medium", ha='left', va='top', color='white')


        elif j > 0:
            ax.contour(Y, X, data_casci, [10], colors="white", linestyles="--")
            contour10 = ax.contour(Y, X, data, [10], colors="white")

        if i == len(titles) - 1:
            ax.set_xlabel(r"$R_y$ ($\mathrm{\AA}$)")

        ax.tick_params(axis='both', color="white")
        for t in ax.get_xticklabels():
            t.set_color('black')
        for t in ax.get_yticklabels():
            t.set_color('black')

axes[0, 0].set_title("CASCI", fontsize='x-large')
axes[0, 1].set_title("VQE-shallow", fontsize='x-large')
axes[0, 2].set_title("AQC-low", fontsize='x-large')

axes[0, 0].text(0.95, 0.05, r"$\longleftarrow$ Time", transform=axes[0, 0].transAxes, fontsize="medium", rotation=90, ha='right', va='bottom', color='white', alpha=0.75)
axes[3, 0].text(0.95, 0.05, r"$\longleftarrow$ Time", transform=axes[3, 0].transAxes, fontsize="medium", rotation=90, ha='right', va='bottom', color='white', alpha=0.75)
axes[5, 0].text(0.05, 0.05, r"$\longleftarrow$ Time", transform=axes[5, 0].transAxes, fontsize="medium", rotation=90, ha='left', va='bottom', color='white', alpha=0.75)

cbar = fig.colorbar(im, ax=axes, shrink=0.45, aspect=20*0.45*2, orientation='vertical', label='Proton density', pad=0.02)
cbar.add_lines(contour10)

plt.savefig("fig4.pdf")
plt.show()