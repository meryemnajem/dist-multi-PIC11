

import numpy as np
from distillation_multicomposants import Compound, ThermodynamicPackage, ShortcutDistillation
from visualization import DistillationVisualizer


def exemple_btx():

    print("\n=== EXEMPLE BTX — Distillation Multicomposants Simplifiée ===\n")

    # ---------------------------------------------------------------
    # 1️⃣ Définition du système BTX
    # ---------------------------------------------------------------
    compound_names = ["benzene", "toluene", "xylene"]
    compounds = [Compound(c) for c in compound_names]

    thermo = ThermodynamicPackage(compounds)

    F = 100.0               # kmol/h
    z_F = np.array([0.333, 0.333, 0.334])
    P = 101325.0            # Pa

    print("Système :", compound_names)
    print("Débit F =", F, "kmol/h")
    print("z_F =", z_F)
    print("Pression =", P, "Pa")

    # ---------------------------------------------------------------
    # 2️⃣ Calcul Shortcut complet
    # ---------------------------------------------------------------
    shortcut = ShortcutDistillation(thermo, F, z_F, P)

    results = shortcut.complete_shortcut_design(
        recovery_LK_D=0.95,
        recovery_HK_B=0.95,
        R_factor=1.3,
        q=1.0,
        efficiency=0.7
    )

    print("\n=== Résultats Shortcut Simplifiés ===")
    for k, v in results.items():
        print(f"{k:15s}: {v}")

    # ---------------------------------------------------------------
    # 3️⃣ Création des 4 graphiques officiels
    # ---------------------------------------------------------------
    visualizer = DistillationVisualizer(compound_names)

    print("\nGénération des graphiques…")

    # 1. Bilan matière
    visualizer.plot_material_balance(
        F, results["D"], results["B"],
        z_F, results["x_D"], results["x_B"],
        save_path="btx_bilan_matiere.png"
    )

    # 2. Résultats Shortcut
    visualizer.plot_shortcut_results(
        results,
        save_path="btx_shortcut_results.png"
    )

    # 3. Profils de composition (approximation très simple)
    N_real = int(round(results["N_real"]))
    if N_real < 3:
        N_real = 3

    stages = np.arange(1, N_real + 1)

    x_profiles = np.linspace(results["x_D"], results["x_B"], N_real)
    y_profiles = x_profiles.copy()  # même tendance simplifiée

    visualizer.plot_composition_profiles_matplotlib(
        stages, x_profiles, y_profiles,
        results["feed_stage"],
        save_path="btx_composition_profiles.png"
    )

    # 4. Profil de température
    Tb_values = np.array([c.Tb for c in compounds])
    T_top = np.sum(results["x_D"] * Tb_values)
    T_bottom = np.sum(results["x_B"] * Tb_values)
    temperatures = np.linspace(T_top, T_bottom, N_real)

    visualizer.plot_temperature_profile(
        stages, temperatures,
        results["feed_stage"],
        save_path="btx_temperature_profile.png"
    )

    print("\n🎉 Graphiques générés :")
    print("   ✓ btx_bilan_matiere.png")
    print("   ✓ btx_shortcut_results.png")
    print("   ✓ btx_composition_profiles.png")
    print("   ✓ btx_temperature_profile.png")

    print("\n=== FIN DE L'EXEMPLE BTX ===\n")

    return results


if __name__ == "__main__":
    exemple_btx()
