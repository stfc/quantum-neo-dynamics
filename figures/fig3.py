import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines
import matplotlib.patheffects as mpe


palette = ["#003f5c", "#bc5090", "#ffa600"]
offset = -258.9475070402643 + 259.2063702444815
labels = ['300', '210', '120', '030', '021', '012', '003']
label_positions = np.arange(7) + 0.5

def rate_constant(energy_diff_hartree: float, temperature: np.array) -> np.array:

    k_B = 1.380649e-23  # Boltzmann constant (J/K)
    h = 6.62607015e-34  # Planck's constant (J·s)
    R = 8.314462618  # Gas constant (J/(mol·K))
    hartree_to_joule = 4.3597447222071e-18  # Conversion factor: 1 Hartree = J
    avogadro = 6.02214076e23  # Avogadro's number

    energy_diff_jmol = energy_diff_hartree * hartree_to_joule * avogadro
    return (k_B * temperature / h) * np.exp(-energy_diff_jmol / (R * temperature))


if __name__ == "__main__":

    with open("../results/statevector_data.json", "r") as f:
        file = json.load(f)
        aqc_high = file["aqc-high"]["030"] - file["aqc-high"]["300"] - offset
        aqc_low = file["aqc-low"]["030"] - file["aqc-low"]["300"] - offset
        hf = file["product"]["030"] - file["product"]["300"] - offset
        casci = file["casci"]["030"] - file["casci"]["300"] - offset
        vqe_shallow = file["adapt-vqe-low"]["030"] - file["adapt-vqe-low"]["300"] - offset
        vqe_deep = file["adapt-vqe-high"]["030"] - file["adapt-vqe-high"]["300"] - offset

    zne_fit_first_mean = (23.7198, 12.1986)
    zne_diff_first_mean = (17.6721, 3.3033)
    # Fit first intercept: 23.7198 ± 12.1986 mHa
    # Diff first intercept: 17.6721 ± 3.3033 mHa

    # ============ Create the barrier plot ============
    plt.style.use("proton.mplstyle")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.0 * 0.9, 3.2 * 0.9), constrained_layout=True)

    ax0.bar([0], [vqe_shallow * 1e3], color=palette[0])
    ax0.bar([1], [aqc_high * 1e3], color=palette[1], edgecolor='white')
    ax0.bar([2], [aqc_low * 1e3], color=palette[1], hatch="\\\\\\", edgecolor='white')
    ax0.bar([3], [zne_fit_first_mean[0]], color=palette[2], edgecolor='white', hatch="\\\\\\")
    ax0.bar([3], [zne_fit_first_mean[0]], yerr=zne_fit_first_mean[1], color="none", edgecolor='grey', ecolor="grey", capsize=3, zorder=100)
    ax0.bar([4], [zne_diff_first_mean[0]], color=palette[2], edgecolor='white', hatch="\\\\\\")
    ax0.bar([4], [zne_diff_first_mean[0]], yerr=zne_diff_first_mean[1], color="none", edgecolor='grey', ecolor="grey", capsize=3, zorder=100)


    pe1 = [mpe.Stroke(linewidth=3, foreground='white'),
           mpe.Stroke(foreground='white', alpha=1),
           mpe.Normal()]
    ax0.set_ylabel("Barrier heights [mHa]")
    ax0.set_xticks([0, 1, 2, 3, 4])
    ax0.set_xticklabels(['VQE-shallow', 'AQC-high', 'AQC-low', 'ZNE (fit first)', 'ZNE (diff first)'], rotation=30, ha="right")
    ax0.grid(which='both', linestyle=':', alpha=0.75, color="grey", lw=0.75, zorder=0)
    ax0.axhline(y=casci * 1e3, color='#008000', linestyle='-', linewidth=2, path_effects=pe1, label="CASCI")
    ax0.axhline(y=hf * 1e3, color='#008000', linestyle='--', linewidth=2, path_effects=pe1, label="PROD")
    ax0.set_xlim(-3, 5)


    ax0.text(-2.9, casci * 1e3 + 0.25, "CASCI", va="bottom", ha="left", fontsize="large")
    ax0.text(-2.9, hf * 1e3 + 0.25, "HF-product", color="black", va="bottom", ha="left", fontsize="large")

    # Compute rate constants over a temperature range
    temperatures = np.linspace(110, 150, 30)
    ax1.plot(temperatures, rate_constant(casci, temperatures), label="CASCI", c='#008000', lw=2, path_effects=pe1)
    ax1.plot(temperatures, rate_constant(vqe_shallow, temperatures), label="VQE-shallow", c=palette[0], lw=2, path_effects=pe1)
    ax1.plot(temperatures, rate_constant(aqc_high, temperatures), label="AQC-high", c=palette[1], lw=2, path_effects=pe1)

    ax1.set_xlabel("Temperature [K]")
    ax1.set_ylabel(r"Rate constant, $k(T)$ [s$^{-1}$]")
    ax1.grid(which='both', linestyle=':', alpha=0.75, color="grey", lw=0.75, zorder=0)

    handles = [
        lines.Line2D([], [], color="#008000", lw=2, path_effects=pe1, label="CASCI"),
        lines.Line2D([], [], color=palette[0], lw=2, path_effects=pe1, linestyle="-", label="VQE-shallow"),
        lines.Line2D([], [], color=palette[1], lw=2, path_effects=pe1, label=r"AQC-high"),
    ]
    ax1.legend(handles=handles, loc="upper left", frameon=True, edgecolor="none", facecolor="w")

    ax1.set_ylim(-0.05, 0.66)
    ax1.set_xlim(110, 145)
    plt.savefig("fig3.pdf")
    plt.show()
