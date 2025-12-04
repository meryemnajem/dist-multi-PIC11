import matplotlib
matplotlib.use('Agg')  # Pour éviter les erreurs Tkinter

from flask import Flask, render_template, request
import numpy as np

from distillation_multicomposants import Compound, ThermodynamicPackage, ShortcutDistillation
from visualization import DistillationVisualizer

app = Flask(__name__)

AVAILABLE_COMPOUNDS = ['benzene', 'toluene', 'xylene']


# ------------------------------------------------------------
# 1️⃣ Page d'accueil
# ------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template(
        "index.html",
        compounds=AVAILABLE_COMPOUNDS,
        selected=[],
        compound_info=None
    )


# ------------------------------------------------------------
# 2️⃣ Propriétés : Tb, Mw, K, α
# ------------------------------------------------------------
@app.route("/properties", methods=["POST"])
def properties():

    selected = request.form.getlist('compounds')
    compounds = [Compound(name) for name in selected]
    thermo = ThermodynamicPackage(compounds)

    # Propriétés Tb, Mw
    compound_info = [
        {
            'name': c.name,
            'Tb': getattr(c, 'Tb', None),
            'Mw': getattr(c, 'MW', None)  # Correction : MW est l'attribut correct
        }
        for c in compounds
    ]

    try:
        F = float(request.form['F'])
        P = float(request.form['P']) * 1e5
        z_F = np.array([float(request.form[f'comp_{i}']) for i in range(len(compounds))])

        # Calcul automatique K et α
        T_avg = np.mean([c.Tb for c in compounds if getattr(c, 'Tb', None) is not None])
        K = thermo.K_values(T_avg, P)
        alpha = thermo.relative_volatilities(T_avg, P)

    except (KeyError, ValueError):
        F, P, z_F, K, alpha = None, None, None, None, None

    data = {
        'compounds': selected,
        'F': F,
        'P': P,
        'z_F': z_F.tolist() if z_F is not None else None,
        'compound_info': compound_info,
        'K': K.tolist() if K is not None else None,
        'alpha': alpha.tolist() if alpha is not None else None
    }

    return render_template("properties.html", data=data)


# ------------------------------------------------------------
# 3️⃣ Identification LK / HK
# ------------------------------------------------------------
@app.route("/identification", methods=["POST"])
def identification():

    compound_names = request.form.getlist('compounds')
    compounds = [Compound(name) for name in compound_names]

    F = float(request.form['F'])
    P = float(request.form['P'])
    z_F = np.array([float(x) for x in request.form.getlist('z_F')])

    Tb_values = [c.Tb for c in compounds]
    idx_LK = np.argmin(Tb_values)
    idx_HK = np.argmax(Tb_values)

    LK = compound_names[idx_LK]
    HK = compound_names[idx_HK]

    return render_template(
        "identification.html",
        compounds=compound_names,
        F=F,
        P=P,
        z_F=z_F.tolist(),
        LK=LK,
        HK=HK
    )


# ------------------------------------------------------------
# 4️⃣ Résumé final
# ------------------------------------------------------------
@app.route("/resume", methods=["POST"])
def resume():

    compound_names = request.form.getlist('compounds')
    compounds = [Compound(name) for name in compound_names]
    thermo = ThermodynamicPackage(compounds)

    # Données entrée utilisateur
    F = float(request.form['F'])
    P = float(request.form['P'])
    z_F = np.array([float(x) for x in request.form.getlist('z_F')])

    # Paramètres avancés
    recovery_LK_D = float(request.form.get('recovery_LK_D', 0.95))
    recovery_HK_B = float(request.form.get('recovery_HK_B', 0.95))
    R_factor = float(request.form.get('R_factor', 1.3))
    q = float(request.form.get('q', 1.0))
    efficiency = float(request.form.get('efficiency', 0.7))

    # Calcul Shortcut
    distillation = ShortcutDistillation(thermo, F, z_F, P)
    results = distillation.complete_shortcut_design(
        recovery_LK_D=recovery_LK_D,
        recovery_HK_B=recovery_HK_B,
        R_factor=R_factor,
        q=q,
        efficiency=efficiency
    )

    visualizer = DistillationVisualizer(compound_names)

    # -----------------------------
    # Graphique 1 : Bilan matière
    # -----------------------------
    visualizer.plot_material_balance(
        F, results['D'], results['B'], z_F,
        results['x_D'], results['x_B'],
        save_path='static/btx_bilan_matiere.png'
    )

    # -----------------------------
    # Graphique 2 : Résultats Shortcut
    # -----------------------------
    visualizer.plot_shortcut_results(
        results,
        save_path='static/btx_shortcut_results.png'
    )

    # -----------------------------
    # Profils colonne (⚠ corrigé)
    # -----------------------------
    N_real = results['N_real']
    N_real = int(round(N_real))

    if N_real < 2:
        N_real = 2

    stages = np.arange(1, N_real + 1)
    x_profiles = np.zeros((N_real, len(compounds)))
    y_profiles = np.zeros((N_real, len(compounds)))

    for j, stage in enumerate(stages):
        if stage <= results['feed_stage']:
            ratio = (stage - 1) / results['feed_stage']
            x_stage = results['x_D'] + ratio * (z_F - results['x_D'])
        else:
            ratio = (stage - results['feed_stage']) / max(1, (N_real - results['feed_stage']))
            x_stage = z_F + ratio * (results['x_B'] - z_F)

        x_profiles[j, :] = x_stage / np.sum(x_stage)
        y_profiles[j, :] = x_stage

    visualizer.plot_composition_profiles_matplotlib(
        stages, x_profiles, y_profiles,
        results['feed_stage'],
        save_path='static/btx_composition_profiles.png'
    )

    # -----------------------------
    # Profil température
    # -----------------------------
    temperatures = np.linspace(compounds[0].Tb, compounds[-1].Tb, N_real)
    visualizer.plot_temperature_profile(
        stages, temperatures,
        results['feed_stage'],
        save_path='static/btx_temperature_profile.png'
    )

    # -----------------------------
    # Graphique interactif Plotly
    # -----------------------------
    try:
        visualizer.plot_composition_profiles_plotly(
            stages, x_profiles, y_profiles,
            results['feed_stage'],
            save_path='static/composition_profiles_interactive.html'
        )
        interactive_path = 'static/composition_profiles_interactive.html'
    except:
        interactive_path = None

    # chemins images
    graphs = {
        'bilan_matiere': 'static/btx_bilan_matiere.png',
        'shortcut_results': 'static/btx_shortcut_results.png',
        'composition_profiles': 'static/btx_composition_profiles.png',
        'temperature_profile': 'static/btx_temperature_profile.png',
        'interactive': interactive_path
    }

    return render_template(
        "resume.html",
        compounds=compound_names,
        results=results,
        graphs=graphs
    )


# ------------------------------------------------------------
# Lancement
# ------------------------------------------------------------
if __name__ == "__main__":
    print("""
============================================
🚀 Application DistillationPro démarrée !
🌐 Accédez à : http://localhost:5000
============================================
""")
    app.run(debug=True, host='0.0.0.0', port=5000)
