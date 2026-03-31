import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
import io
import base64

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="STC Services Pharma -Analyse", layout="wide", page_icon="📊")

# ============================================================
# AUTHENTIFICATION
# ============================================================
USERS = {
    "stc": "stc2025",
    "admin": "admin2025",
    "ems": "ems2025",
}

def login():
    _logo_login = os.path.join(os.path.dirname(__file__), "logo_stc.png")

    # Masquer sidebar, header, footer et bloquer le scroll
    st.markdown("""<style>
        [data-testid="stSidebar"] {display: none;}
        header {display: none !important;}
        footer {display: none !important;}
        [data-testid="stAppViewContainer"] {
            display: flex; align-items: center; justify-content: center;
            min-height: 100vh; max-height: 100vh; overflow: hidden;
        }
        .block-container {
            max-width: 420px !important; padding-top: 0 !important;
            padding-bottom: 0 !important;
        }
    </style>""", unsafe_allow_html=True)

    st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
    if os.path.exists(_logo_login):
        st.image(_logo_login, width=130)
    st.markdown("## STC Services Pharma")
    st.markdown("*Tableau de bord -Analyse portefeuille clients*")
    st.markdown("---")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.form("login_form"):
        user = st.text_input("Identifiant")
        pwd = st.text_input("Mot de passe", type="password")
        submit = st.form_submit_button("Se connecter", use_container_width=True)

    if submit:
        if user in USERS and USERS[user] == pwd:
            st.session_state["authenticated"] = True
            st.session_state["user"] = user
            st.rerun()
        else:
            st.error("Identifiant ou mot de passe incorrect.")

    st.caption("Contactez votre administrateur si vous avez oublie vos identifiants.")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
    st.stop()

# Palette neutre
C = {
    "primary": "#2563EB",       # bleu -donnees principales, 2025, actif
    "secondary": "#94A3B8",     # gris -donnees secondaires, 2024, comparaison
    "positive": "#16A34A",      # vert -croissance, bon
    "warning": "#D97706",       # ambre -attention, baisse
    "danger": "#DC2626",        # rouge -critique, perdu
    "neutral": "#475569",       # gris fonce -texte, barres neutres
    "light": "#E2E8F0",        # gris clair -fonds
    "accent1": "#7C3AED",      # violet -nouveaux, special
    "accent2": "#0891B2",      # cyan -stable
}

# Segments
COLORS_SEG = {
    "En croissance": C["positive"],
    "Stable": C["accent2"],
    "Nouveau": C["accent1"],
    "En baisse": C["warning"],
    "A risque": C["danger"],
    "Perdu": C["neutral"],
    "Inactif": C["light"],
}

# Annees
COLORS_ANNEE = {"2024": C["secondary"], "2025": C["primary"]}
COLORS_ANNEE_INT = {2024: C["secondary"], 2025: C["primary"]}

BASE = os.path.dirname(__file__)
DATA_CLIENTS = os.path.join(BASE, "DATASETS", "clients")
DATA_VENTES = os.path.join(BASE, "DATASETS", "ventes")

# ============================================================
# CHARGEMENT DES DONNEES
# ============================================================

@st.cache_data
def load_data():
    # Clients
    stc = pd.read_csv(os.path.join(DATA_CLIENTS, "clients_STC.csv"))
    ems = pd.read_csv(os.path.join(DATA_CLIENTS, "clients_EMS.csv"))

    # Normaliser les colonnes STC
    stc = stc.rename(columns={
        stc.columns[0]: "Client",
        stc.columns[1]: "CP",
        stc.columns[2]: "Ville",
        stc.columns[3]: "Adresse",
        stc.columns[4]: "Type",
        stc.columns[5]: "CA_Loc_2025",
        stc.columns[6]: "CA_Vte_2025",
        stc.columns[7]: "CA_Loc_2024",
        stc.columns[8]: "CA_Vte_2024",
    })
    stc["Entite"] = "STC"
    stc = stc[["Client", "CP", "Ville", "Type", "Entite", "CA_Loc_2025", "CA_Vte_2025", "CA_Loc_2024", "CA_Vte_2024"]]

    # Normaliser les colonnes EMS
    ems = ems.rename(columns={
        ems.columns[0]: "Client",
        ems.columns[1]: "CP",
        ems.columns[2]: "Ville",
        ems.columns[6]: "Type",
        ems.columns[8]: "CA_Loc_2025",
        ems.columns[9]: "CA_Vte_2025",
        ems.columns[10]: "CA_Loc_2024",
        ems.columns[11]: "CA_Vte_2024",
    })
    ems["Entite"] = "EMS"
    ems = ems[["Client", "CP", "Ville", "Type", "Entite", "CA_Loc_2025", "CA_Vte_2025", "CA_Loc_2024", "CA_Vte_2024"]]

    clients = pd.concat([stc, ems], ignore_index=True)
    for col in ["CA_Loc_2025", "CA_Vte_2025", "CA_Loc_2024", "CA_Vte_2024"]:
        clients[col] = pd.to_numeric(clients[col], errors="coerce").fillna(0)

    clients["CA_2025"] = clients["CA_Loc_2025"] + clients["CA_Vte_2025"]
    clients["CA_2024"] = clients["CA_Loc_2024"] + clients["CA_Vte_2024"]
    clients["Evolution_EUR"] = clients["CA_2025"] - clients["CA_2024"]
    clients["Evolution_PCT"] = np.where(clients["CA_2024"] > 0, clients["Evolution_EUR"] / clients["CA_2024"] * 100, np.nan)

    # Ventes
    v24 = pd.read_csv(os.path.join(DATA_VENTES, "Extraction_Ventes_2024.csv"))
    v25 = pd.read_csv(os.path.join(DATA_VENTES, "Extraction_Ventes_2025.csv"))

    cols_ventes = ["Facture", "Date", "Client", "Code_Article", "Libelle", "Reference", "Qte", "PU_HT", "Total_HT", "Total_TTC", "Rem_HT", "Rem_PCT", "Code_TVA", "Px_Revient_HT", "Famille"]
    v24.columns = cols_ventes
    v25.columns = cols_ventes

    v24["Annee"] = 2024
    v25["Annee"] = 2025
    ventes = pd.concat([v24, v25], ignore_index=True)
    ventes = ventes.dropna(subset=["Facture"])
    ventes["Date"] = pd.to_datetime(ventes["Date"], errors="coerce")
    ventes["Mois"] = ventes["Date"].dt.to_period("M").astype(str)
    for col in ["Qte", "PU_HT", "Total_HT", "Total_TTC", "Px_Revient_HT"]:
        ventes[col] = pd.to_numeric(ventes[col], errors="coerce").fillna(0)
    ventes["Famille"] = ventes["Famille"].astype(str).str.replace(".0", "", regex=False)

    # Mapper entite dans ventes (permet de filtrer STC/EMS dans les analyses)
    client_entite = dict(zip(clients["Client"], clients["Entite"]))
    ventes["Entite"] = ventes["Client"].map(client_entite).fillna("STC")

    # Familles
    familles = pd.read_csv(os.path.join(DATA_VENTES, "familles.csv"))
    familles["Code Famille"] = familles["Code Famille"].astype(str)

    fam_map = {}
    for _, row in familles.iterrows():
        code = str(row["Code Famille"])
        fam_map[code] = {"Libelle_Famille": row["Libelle"], "Rayon": row["Rayon"]}
        if code.startswith("0"):
            fam_map[code[1:]] = {"Libelle_Famille": row["Libelle"], "Rayon": row["Rayon"]}

    ventes["Libelle_Famille"] = ventes["Famille"].map(lambda x: fam_map.get(str(x), {}).get("Libelle_Famille", "Autre"))
    ventes["Rayon"] = ventes["Famille"].map(lambda x: fam_map.get(str(x), {}).get("Rayon", "Autre"))

    return clients, ventes, familles


clients, ventes, familles = load_data()

# ============================================================
# HELPERS
# ============================================================
def insight(text, icon="💡"):
    st.markdown(f"""<div style="background:#F8FAFC;border-left:4px solid {C['primary']};border-radius:0 8px 8px 0;padding:14px 18px;margin:12px 0 24px 0;font-size:14px;color:#374151;line-height:1.6"><b>{icon} Interpretation :</b> {text}</div>""", unsafe_allow_html=True)

def csv_download(df, filename, label="📥 Telecharger (CSV pour Excel)"):
    csv_data = df.to_csv(index=False, sep=";").encode("utf-8-sig")
    st.download_button(label, csv_data, file_name=filename, mime="text/csv")

def label_risque(score):
    if score >= 70:
        return "🔴 Critique"
    if score >= 50:
        return "🟠 Eleve"
    if score >= 30:
        return "🟡 Modere"
    return "🟢 Faible"

# ============================================================
# SIDEBAR -STYLE (inspire de servicespharma-marseille.fr)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=Raleway:wght@400;500;600&display=swap');

/* --- Sidebar fond blanc, barre verte en haut --- */
section[data-testid="stSidebar"] > div:first-child {
    background: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}
section[data-testid="stSidebar"] > div:first-child::before {
    content: "";
    display: block;
    height: 5px;
    background: linear-gradient(90deg, #679509 0%, #76b860 100%);
    margin-bottom: 8px;
}

/* Titre sidebar */
section[data-testid="stSidebar"] h1 {
    color: #76b860 !important;
    font-family: 'Oswald', sans-serif !important;
    font-size: 1.3rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Label "Navigation" + "Entite" */
section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] .stRadio > label {
    color: #005fa9 !important;
    font-family: 'Oswald', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Radio boutons -texte normal */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span {
    color: #374151 !important;
    font-family: 'Raleway', sans-serif !important;
    font-size: 0.9rem !important;
    transition: color 0.2s;
}

/* Radio bouton -hover */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover span {
    color: #76b860 !important;
}

/* Radio bouton selectionne */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] span {
    color: #76b860 !important;
    font-weight: 700 !important;
}

/* Radio dot couleur verte */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] div[data-testid="stMarkdownContainer"] {
    border-color: #76b860 !important;
}

/* Separateurs */
section[data-testid="stSidebar"] hr {
    border-color: #E5E7EB !important;
    margin: 12px 0 !important;
}

/* Multiselect tag chips */
section[data-testid="stSidebar"] span[data-baseweb="tag"] {
    background-color: #6ec1df !important;
    color: #FFFFFF !important;
    font-family: 'Raleway', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 4px !important;
}

/* Multiselect dropdown focus */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    border-color: #76b860 !important;
}

/* Logo en bas centre */
.sidebar-logo-bottom {
    position: fixed;
    bottom: 18px;
    left: 0;
    width: 300px;
    text-align: center;
    opacity: 0.55;
    transition: opacity 0.3s;
    z-index: 999;
}
.sidebar-logo-bottom:hover {
    opacity: 0.9;
}
.sidebar-logo-bottom img {
    width: 36px !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR -CONTENU
# ============================================================
_stc_logo_path = os.path.join(BASE, "logo_stc.png")
if os.path.exists(_stc_logo_path):
    st.sidebar.image(_stc_logo_path, width=120)
st.sidebar.title("STC Services Pharma")
st.sidebar.caption(f"Connecte : **{st.session_state.get('user', '')}**")
if st.sidebar.button("Se deconnecter", use_container_width=True):
    st.session_state["authenticated"] = False
    st.session_state["user"] = ""
    st.rerun()
st.sidebar.markdown("---")

onglet = st.sidebar.radio("Navigation", [
    "Vue Globale",
    "Sante Portefeuille",
    "Impact Commerciaux",
    "Produits & Familles",
    "Opportunites"
])

st.sidebar.markdown("---")
filtre_entite = st.sidebar.multiselect("Entite", ["STC", "EMS"], default=["STC", "EMS"])

# Logo centre en bas de la sidebar
_logo_path = os.path.join(BASE, "evologoSVG.png")
if os.path.exists(_logo_path):
    with open(_logo_path, "rb") as _f:
        _logo_b64 = base64.b64encode(_f.read()).decode()
    st.sidebar.markdown(
        f'<div class="sidebar-logo-bottom"><img src="data:image/png;base64,{_logo_b64}"></div>',
        unsafe_allow_html=True
    )
clients_f = clients[clients["Entite"].isin(filtre_entite)]
ventes_f = ventes[ventes["Entite"].isin(filtre_entite)]

# ============================================================
# ONGLET 1 : VUE GLOBALE
# ============================================================
if onglet == "Vue Globale":
    st.title("Vue Globale")
    st.markdown("---")

    ca_25 = clients_f["CA_2025"].sum()
    ca_24 = clients_f["CA_2024"].sum()
    nb_actifs_25 = (clients_f["CA_2025"] > 0).sum()
    nb_actifs_24 = (clients_f["CA_2024"] > 0).sum()
    perdus_mask = (clients_f["CA_2024"] > 0) & (clients_f["CA_2025"] == 0)
    nouveaux_mask = (clients_f["CA_2024"] == 0) & (clients_f["CA_2025"] > 0)
    perdus = perdus_mask.sum()
    nouveaux = nouveaux_mask.sum()
    ca_perdus = clients_f[perdus_mask]["CA_2024"].sum()
    ca_nouveaux = clients_f[nouveaux_mask]["CA_2025"].sum()
    panier_moy_25 = ca_25 / nb_actifs_25 if nb_actifs_25 else 0
    panier_moy_24 = ca_24 / nb_actifs_24 if nb_actifs_24 else 0

    loc_25 = clients_f["CA_Loc_2025"].sum()
    vte_25 = clients_f["CA_Vte_2025"].sum()
    loc_24 = clients_f["CA_Loc_2024"].sum()
    vte_24 = clients_f["CA_Vte_2024"].sum()

    # --- KPIs principaux ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CA 2025", f"{ca_25:,.0f} EUR", f"{(ca_25-ca_24)/ca_24*100:+.1f}%" if ca_24 else None)
    c2.metric("CA 2024", f"{ca_24:,.0f} EUR")
    c3.metric("Clients actifs 2025", nb_actifs_25, f"{nb_actifs_25-nb_actifs_24:+d} vs 2024")
    c4.metric("Panier moyen 2025", f"{panier_moy_25:,.0f} EUR", f"{(panier_moy_25-panier_moy_24)/panier_moy_24*100:+.1f}%" if panier_moy_24 else None)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Nouveaux clients", nouveaux, f"{ca_nouveaux:,.0f} EUR de CA")
    c6.metric("Clients perdus", perdus, f"-{ca_perdus:,.0f} EUR de CA")
    c7.metric("CA Location 2025", f"{loc_25:,.0f} EUR", f"{(loc_25-loc_24)/loc_24*100:+.1f}%" if loc_24 else None)
    c8.metric("CA Vente 2025", f"{vte_25:,.0f} EUR", f"{(vte_25-vte_24)/vte_24*100:+.1f}%" if vte_24 else None)

    evol_ca = (ca_25 - ca_24) / ca_24 * 100 if ca_24 else 0
    solde_clients = nouveaux - perdus
    insight(f"Le CA global {'progresse' if evol_ca > 0 else 'recule'} de {abs(evol_ca):.1f}% entre 2024 et 2025. Le solde clients est de {solde_clients:+d} ({nouveaux} nouveaux pour {ca_nouveaux:,.0f} EUR contre {perdus} perdus pour {ca_perdus:,.0f} EUR). {'Le portefeuille se developpe.' if solde_clients > 0 else 'Le portefeuille se contracte : la reconquete des clients perdus est prioritaire.'}")

    st.markdown("---")

    # --- Repartition clients par tranche de CA ---
    st.subheader("Repartition des clients par tranche de CA 2025")
    actifs_tranches = clients_f[clients_f["CA_2025"] > 0].copy()
    actifs_tranches["Tranche"] = pd.cut(
        actifs_tranches["CA_2025"],
        bins=[0, 500, 2000, 5000, 10000, 20000, float("inf")],
        labels=["< 500 EUR", "500 - 2 000 EUR", "2 000 - 5 000 EUR", "5 000 - 10 000 EUR", "10 000 - 20 000 EUR", "> 20 000 EUR"]
    )
    tranches_count = actifs_tranches["Tranche"].value_counts().sort_index().reset_index()
    tranches_count.columns = ["Tranche", "Nb clients"]
    tranches_ca = actifs_tranches.groupby("Tranche", observed=True)["CA_2025"].sum().reset_index()
    tranches_ca.columns = ["Tranche", "CA total"]
    tranches_merge = tranches_count.merge(tranches_ca, on="Tranche")

    col_tr1, col_tr2 = st.columns(2)
    with col_tr1:
        fig_tr = px.bar(tranches_merge, x="Tranche", y="Nb clients", color_discrete_sequence=[C["primary"]], labels={"Tranche": "", "Nb clients": "Nombre de clients"})
        fig_tr.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_tr, use_container_width=True)
    with col_tr2:
        fig_tr2 = px.bar(tranches_merge, x="Tranche", y="CA total", color_discrete_sequence=[C["positive"]], labels={"Tranche": "", "CA total": "CA total (EUR)"})
        fig_tr2.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_tr2, use_container_width=True)

    gros = len(actifs_tranches[actifs_tranches["CA_2025"] >= 10000])
    moyens = len(actifs_tranches[(actifs_tranches["CA_2025"] >= 2000) & (actifs_tranches["CA_2025"] < 10000)])
    petits = len(actifs_tranches[actifs_tranches["CA_2025"] < 2000])
    insight(f"{gros} gros clients (> 10 000 EUR), {moyens} clients moyens (2 000 - 10 000 EUR), {petits} petits clients (< 2 000 EUR). Les gros clients pesent lourd mais les petits representent un vivier de developpement.")

    st.markdown("---")

    # --- Evolution CA Loc / Vte 2024 vs 2025 ---
    st.subheader("Evolution Location / Vente (2024 vs 2025)")
    evol_df = pd.DataFrame({
        "Type": ["Location", "Location", "Vente", "Vente"],
        "Annee": ["2024", "2025", "2024", "2025"],
        "CA": [loc_24, loc_25, vte_24, vte_25]
    })
    fig_evol_lv = px.bar(
        evol_df, x="Type", y="CA", color="Annee", barmode="group",
        color_discrete_map=COLORS_ANNEE,
        labels={"CA": "CA (EUR)", "Type": ""}
    )
    fig_evol_lv.update_layout(height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig_evol_lv, use_container_width=True)

    evol_loc = (loc_25 - loc_24) / loc_24 * 100 if loc_24 else 0
    evol_vte = (vte_25 - vte_24) / vte_24 * 100 if vte_24 else 0
    insight(f"La location {'progresse' if evol_loc > 0 else 'recule'} de {abs(evol_loc):.1f}% et la vente {'progresse' if evol_vte > 0 else 'recule'} de {abs(evol_vte):.1f}%. {'Les deux activites evoluent dans le meme sens.' if (evol_loc > 0) == (evol_vte > 0) else 'Les deux activites evoluent en sens contraire, ce qui merite attention.'}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Repartition Loc / Vte 2025")
        fig_pie = px.pie(
            values=[loc_25, vte_25],
            names=["Location", "Vente"],
            color_discrete_sequence=[C["primary"], C["neutral"]],
            hole=0.4
        )
        fig_pie.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("STC vs EMS (2024 et 2025)")
        stc_25 = clients[clients["Entite"] == "STC"]["CA_2025"].sum()
        ems_25 = clients[clients["Entite"] == "EMS"]["CA_2025"].sum()
        stc_24 = clients[clients["Entite"] == "STC"]["CA_2024"].sum()
        ems_24 = clients[clients["Entite"] == "EMS"]["CA_2024"].sum()
        entite_df = pd.DataFrame({
            "Entite": ["STC", "STC", "EMS", "EMS"],
            "Annee": ["2024", "2025", "2024", "2025"],
            "CA": [stc_24, stc_25, ems_24, ems_25]
        })
        fig_entite = px.bar(
            entite_df, x="Entite", y="CA", color="Annee", barmode="group",
            color_discrete_map=COLORS_ANNEE,
            labels={"CA": "CA (EUR)", "Entite": ""}
        )
        fig_entite.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_entite, use_container_width=True)

    evol_stc = (stc_25 - stc_24) / stc_24 * 100 if stc_24 else 0
    evol_ems = (ems_25 - ems_24) / ems_24 * 100 if ems_24 else 0
    insight(f"STC : {stc_25:,.0f} EUR en 2025 ({evol_stc:+.1f}% vs 2024). EMS : {ems_25:,.0f} EUR en 2025 ({evol_ems:+.1f}% vs 2024).")

    st.markdown("---")

    # --- CA par secteur geographique ---
    st.subheader("CA par secteur geographique (Code Postal)")
    ca_cp = clients_f[clients_f["CA_2025"] > 0].groupby("CP").agg(
        CA_2025=("CA_2025", "sum"),
        CA_2024=("CA_2024", "sum"),
        Nb_Clients=("Client", "count")
    ).reset_index().sort_values("CA_2025", ascending=False).head(15)
    ca_cp["CP"] = ca_cp["CP"].astype(str)

    fig_cp = px.bar(
        ca_cp, x="CP", y=["CA_2025", "CA_2024"], barmode="group",
        color_discrete_map={"CA_2025": C["primary"], "CA_2024": C["secondary"]},
        labels={"value": "CA (EUR)", "CP": "Code Postal", "variable": "Annee"}
    )
    fig_cp.update_layout(height=400, margin=dict(t=20, b=20), xaxis_type="category")
    st.plotly_chart(fig_cp, use_container_width=True)

    top_cp = ca_cp.iloc[0] if len(ca_cp) else None
    if top_cp is not None:
        insight(f"Le code postal {top_cp['CP']} concentre le plus de CA avec {top_cp['CA_2025']:,.0f} EUR en 2025 ({top_cp['Nb_Clients']} clients). Ciblez les secteurs en baisse pour concentrer vos efforts telephoniques.")

    st.markdown("---")

    # --- Nouveaux vs Perdus detail ---
    st.subheader("Nouveaux clients vs Clients perdus")

    col_n, col_p = st.columns(2)
    with col_n:
        st.markdown(f"**{nouveaux} nouveaux clients en 2025** -{ca_nouveaux:,.0f} EUR")
        df_nouveaux = clients_f[nouveaux_mask].sort_values("CA_2025", ascending=False)[["Client", "Entite", "Ville", "CA_2025", "CA_Loc_2025", "CA_Vte_2025"]].reset_index(drop=True)
        st.dataframe(df_nouveaux.head(20), use_container_width=True, hide_index=True, height=350)
        csv_download(df_nouveaux, "nouveaux_clients_2025.csv")

    with col_p:
        st.markdown(f"**{perdus} clients perdus en 2025** -{ca_perdus:,.0f} EUR perdus")
        df_perdus = clients_f[perdus_mask].sort_values("CA_2024", ascending=False)[["Client", "Entite", "Ville", "CA_2024", "CA_Loc_2024", "CA_Vte_2024"]].reset_index(drop=True)
        st.dataframe(df_perdus.head(20), use_container_width=True, hide_index=True, height=350)
        csv_download(df_perdus, "clients_perdus_2025.csv")

    if ca_nouveaux > ca_perdus:
        insight(f"Le CA des nouveaux clients ({ca_nouveaux:,.0f} EUR) compense le CA des clients perdus ({ca_perdus:,.0f} EUR). Solde net : {ca_nouveaux - ca_perdus:+,.0f} EUR.")
    else:
        insight(f"Le CA des clients perdus ({ca_perdus:,.0f} EUR) depasse celui des nouveaux ({ca_nouveaux:,.0f} EUR). Deficit net : {ca_nouveaux - ca_perdus:,.0f} EUR. La reconquete est prioritaire.", "⚠️")

    st.markdown("---")

    # --- Top clients 2025 ---
    st.subheader("Top 20 clients 2025")
    top20 = clients_f[clients_f["CA_2025"] > 0].sort_values("CA_2025", ascending=False).head(20)
    fig_top = px.bar(
        top20, x="CA_2025", y="Client", orientation="h",
        color="Entite", color_discrete_map={"STC": C["neutral"], "EMS": C["primary"]},
        labels={"CA_2025": "CA 2025 (EUR)", "Client": ""}
    )
    fig_top.update_layout(height=600, margin=dict(t=20, b=20, l=300), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_top, use_container_width=True)

    ca_top20 = top20["CA_2025"].sum()
    pct_top20 = ca_top20 / ca_25 * 100 if ca_25 else 0
    insight(f"Les 20 premiers clients representent {ca_top20:,.0f} EUR, soit {pct_top20:.0f}% du CA total 2025. {'Forte concentration : une perte parmi ces clients aurait un impact majeur.' if pct_top20 > 50 else 'Le portefeuille est relativement diversifie.'}")
    csv_download(top20[["Client", "Entite", "Ville", "CA_2025", "CA_2024"]].reset_index(drop=True), "top20_clients_2025.csv")

    st.markdown("---")

    # Evolution mensuelle depuis les ventes
    st.subheader("Evolution mensuelle du CA (donnees ventes)")

    ventes_mois = ventes_f.groupby(["Mois", "Annee"])["Total_HT"].sum().reset_index()
    ventes_mois = ventes_mois.sort_values("Mois")
    fig_line = px.line(
        ventes_mois, x="Mois", y="Total_HT",
        color="Annee",
        color_discrete_map=COLORS_ANNEE_INT,
        labels={"Total_HT": "CA HT (EUR)", "Mois": ""},
        markers=True
    )
    fig_line.update_layout(height=400, margin=dict(t=20, b=20))
    st.plotly_chart(fig_line, use_container_width=True)

    ca_moy_24 = ventes_mois[ventes_mois["Annee"] == 2024]["Total_HT"].mean()
    ca_moy_25 = ventes_mois[ventes_mois["Annee"] == 2025]["Total_HT"].mean()
    evol_mens = (ca_moy_25 - ca_moy_24) / ca_moy_24 * 100 if ca_moy_24 else 0
    entites_txt = " + ".join(filtre_entite)
    if "EMS" in filtre_entite and "STC" in filtre_entite:
        insight(f"Le CA mensuel moyen passe de {ca_moy_24:,.0f} EUR en 2024 a {ca_moy_25:,.0f} EUR en 2025 ({evol_mens:+.1f}%). Note : 2024 ne contient que STC dans les ventes. Pour une comparaison a perimetre constant, selectionnez uniquement STC dans la barre laterale.", "⚠️")
    else:
        insight(f"Le CA mensuel moyen ({entites_txt}) passe de {ca_moy_24:,.0f} EUR en 2024 a {ca_moy_25:,.0f} EUR en 2025 ({evol_mens:+.1f}%).")


# ============================================================
# ONGLET 2 : SANTE PORTEFEUILLE
# ============================================================
elif onglet == "Sante Portefeuille":
    st.title("Sante du Portefeuille Client")
    st.markdown("---")

    # Segmentation enrichie
    clients_f = clients_f.copy()

    # Recence : derniere commande depuis les ventes
    derniere_commande = ventes.groupby("Client")["Date"].max().reset_index()
    derniere_commande.columns = ["Client", "Derniere_Commande"]
    clients_f = clients_f.merge(derniere_commande, on="Client", how="left")
    date_ref = ventes["Date"].max()
    clients_f["Jours_Sans_Commande"] = (date_ref - clients_f["Derniere_Commande"]).dt.days
    clients_f["Jours_Sans_Commande"] = clients_f["Jours_Sans_Commande"].fillna(999)

    # Frequence depuis les ventes 2025
    freq_25 = ventes[ventes["Annee"] == 2025].groupby("Client")["Facture"].nunique().reset_index()
    freq_25.columns = ["Client", "Nb_Commandes_25"]
    clients_f = clients_f.merge(freq_25, on="Client", how="left")
    clients_f["Nb_Commandes_25"] = clients_f["Nb_Commandes_25"].fillna(0).astype(int)

    # Nb familles 2025
    fam_25 = ventes[ventes["Annee"] == 2025].groupby("Client")["Famille"].nunique().reset_index()
    fam_25.columns = ["Client", "Nb_Familles_25"]
    clients_f = clients_f.merge(fam_25, on="Client", how="left")
    clients_f["Nb_Familles_25"] = clients_f["Nb_Familles_25"].fillna(0).astype(int)

    # Delta EUR
    clients_f["Delta_EUR"] = clients_f["CA_2025"] - clients_f["CA_2024"]

    # Segmentation multicritere
    def segmenter(row):
        ca24, ca25 = row["CA_2024"], row["CA_2025"]
        delta = row["Delta_EUR"]

        if ca24 == 0 and ca25 > 0:
            return "Nouveau"
        if ca24 > 0 and ca25 == 0:
            return "Perdu"
        if ca24 == 0 and ca25 == 0:
            return "Inactif"

        evol_pct = (ca25 - ca24) / ca24

        if evol_pct <= -0.30 and abs(delta) > 500:
            return "A risque"
        if evol_pct <= -0.05 and abs(delta) > 200:
            return "En baisse"
        if evol_pct >= 0.15 and delta > 200:
            return "En croissance"

        return "Stable"

    clients_f["Segment"] = clients_f.apply(segmenter, axis=1)

    # Score de risque 0-100
    def score_risque(row):
        if row["Segment"] == "Perdu":
            return 100
        if row["Segment"] == "Nouveau":
            return 10
        if row["Segment"] == "Inactif":
            return 50

        score = 50

        if row["CA_2024"] > 0:
            evol = (row["CA_2025"] - row["CA_2024"]) / row["CA_2024"]
            if evol <= -0.50: score += 30
            elif evol <= -0.30: score += 20
            elif evol <= -0.10: score += 10
            elif evol >= 0.15: score -= 20
            elif evol >= 0: score -= 10

        jours = row["Jours_Sans_Commande"]
        if jours > 330: score += 15
        elif jours > 300: score += 5
        elif jours < 30: score -= 10

        if row["Nb_Commandes_25"] <= 1: score += 10
        elif row["Nb_Commandes_25"] >= 5: score -= 10

        return max(0, min(100, score))

    clients_f["Score_Risque"] = clients_f.apply(score_risque, axis=1)
    clients_f["Niveau_Risque"] = clients_f["Score_Risque"].apply(label_risque)

    # Criteres d'explication
    st.subheader("Comment lire cette page")
    with st.expander("Voir les criteres de classement des clients"):
        st.markdown("""
| Categorie | Comment on classe le client |
|-----------|---------------------------|
| **Perdu** | Il achetait en 2024 mais plus rien en 2025 |
| **A risque** | Son CA a chute de plus de 30% (et plus de 500 EUR de perdu) |
| **En baisse** | Son CA baisse de plus de 5% (et plus de 200 EUR de perdu) |
| **Stable** | Son CA varie peu (entre -5% et +15%) |
| **En croissance** | Son CA augmente de plus de 15% (et plus de 200 EUR de gagne) |
| **Nouveau** | Il n'achetait rien en 2024 et commande en 2025 |

**Niveau de risque** : combine l'evolution du CA, la date de la derniere commande, et le nombre de commandes en 2025.
- 🟢 **Faible** (0-30) : client en bonne sante
- 🟡 **Modere** (30-50) : a surveiller
- 🟠 **Eleve** (50-70) : action recommandee
- 🔴 **Critique** (70-100) : intervention urgente
        """)

    # KPIs segments
    seg_counts = clients_f["Segment"].value_counts()

    # --- KPIs 2025 ---
    st.subheader("Situation 2025")
    ca_total_25 = clients_f["CA_2025"].sum()
    nb_actifs = (clients_f["CA_2025"] > 0).sum()
    panier_moy = ca_total_25 / nb_actifs if nb_actifs else 0
    ca_median = clients_f[clients_f["CA_2025"] > 0]["CA_2025"].median()
    ca_top10 = clients_f.nlargest(10, "CA_2025")["CA_2025"].sum()
    pct_top10 = ca_top10 / ca_total_25 * 100 if ca_total_25 else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("CA total 2025", f"{ca_total_25:,.0f} EUR")
    k2.metric("Clients actifs", nb_actifs)
    k3.metric("CA moyen / client", f"{panier_moy:,.0f} EUR")
    k4.metric("CA median / client", f"{ca_median:,.0f} EUR")
    k5.metric("Top 10 clients", f"{pct_top10:.0f}% du CA")

    insight(f"Le CA moyen par client est de {panier_moy:,.0f} EUR mais la moitie des clients font moins de {ca_median:,.0f} EUR. Les 10 plus gros clients concentrent {pct_top10:.0f}% du CA : attention a la dependance.")

    st.markdown("---")

    # --- Segments ---
    st.subheader("Repartition des clients par categorie")

    seg_df = clients_f.groupby("Segment").agg(
        Nb=("Client", "count"),
        CA_2025=("CA_2025", "sum"),
        CA_2024=("CA_2024", "sum"),
        CA_Moy_25=("CA_2025", "mean"),
    ).reset_index()
    seg_df["Pct_CA_25"] = seg_df["CA_2025"] / seg_df["CA_2025"].sum() * 100

    segments_order = ["En croissance", "Stable", "Nouveau", "En baisse", "A risque", "Perdu", "Inactif"]
    segments_present = [s for s in segments_order if s in seg_counts.index]
    cols = st.columns(len(segments_present))
    for i, seg in enumerate(segments_present):
        count = seg_counts.get(seg, 0)
        ca25_seg = clients_f[clients_f["Segment"] == seg]["CA_2025"].sum()
        ca24_seg = clients_f[clients_f["Segment"] == seg]["CA_2024"].sum()
        cols[i].markdown(f"**{seg}**")
        cols[i].markdown(f"### {count}")
        cols[i].caption(f"CA 2025 : {ca25_seg:,.0f} EUR")
        cols[i].caption(f"CA 2024 : {ca24_seg:,.0f} EUR")

    nb_risque = seg_counts.get("A risque", 0) + seg_counts.get("En baisse", 0)
    nb_perdus = seg_counts.get("Perdu", 0)
    ca_risque_25 = clients_f[clients_f["Segment"].isin(["A risque", "En baisse"])]["CA_2025"].sum()
    ca_perdus_24 = clients_f[clients_f["Segment"] == "Perdu"]["CA_2024"].sum()
    pct_danger = (nb_risque + nb_perdus) / len(clients_f) * 100 if len(clients_f) else 0
    insight(f"{nb_risque} clients en baisse/a risque generent encore {ca_risque_25:,.0f} EUR en 2025 (a securiser). {nb_perdus} clients perdus representaient {ca_perdus_24:,.0f} EUR en 2024 (a reconquerir). Soit {pct_danger:.0f}% du portefeuille en zone de vigilance.", "⚠️")

    st.markdown("---")

    # --- CA par segment ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("CA 2025 par categorie")
        fig_seg_25 = px.bar(
            seg_df.sort_values("CA_2025", ascending=True), x="CA_2025", y="Segment", orientation="h",
            color="Segment", color_discrete_map=COLORS_SEG,
            labels={"CA_2025": "CA 2025 (EUR)", "Segment": ""}
        )
        fig_seg_25.update_layout(height=350, margin=dict(t=20, b=20, l=120), showlegend=False)
        st.plotly_chart(fig_seg_25, use_container_width=True)

    with col2:
        st.subheader("CA 2024 vs 2025 par categorie")
        seg_melt = seg_df.melt(id_vars=["Segment"], value_vars=["CA_2024", "CA_2025"], var_name="Annee", value_name="CA")
        seg_melt["Annee"] = seg_melt["Annee"].str.replace("CA_", "")
        fig_seg_comp = px.bar(
            seg_melt, x="Segment", y="CA", color="Annee", barmode="group",
            color_discrete_map=COLORS_ANNEE,
            labels={"CA": "CA (EUR)", "Segment": ""}
        )
        fig_seg_comp.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_seg_comp, use_container_width=True)

    # --- Loc/Vte par segment 2025 ---
    st.markdown("---")
    st.subheader("Repartition Loc / Vte par categorie (2025)")

    seg_lv = clients_f.groupby("Segment").agg(
        Loc=("CA_Loc_2025", "sum"),
        Vte=("CA_Vte_2025", "sum")
    ).reset_index()
    seg_lv_melt = seg_lv.melt(id_vars="Segment", var_name="Type", value_name="CA")
    fig_lv_seg = px.bar(
        seg_lv_melt, x="Segment", y="CA", color="Type", barmode="stack",
        color_discrete_map={"Loc": C["primary"], "Vte": C["neutral"]},
        labels={"CA": "CA 2025 (EUR)", "Segment": ""}
    )
    fig_lv_seg.update_layout(height=400, margin=dict(t=20, b=20))
    st.plotly_chart(fig_lv_seg, use_container_width=True)

    insight("Un client qui ne fait que de la vente ponctuelle est plus volatile qu'un client en location (engagement recurrent). Les clients en baisse qui passent de la location a la vente pourraient signaler un desengagement progressif.")

    # --- Liste clients par segment ---
    st.markdown("---")
    st.subheader("Explorer les clients par categorie")

    segments_dispo = sorted(clients_f["Segment"].unique().tolist())
    seg_choisi = st.selectbox("Choisir une categorie", segments_dispo, index=segments_dispo.index("A risque") if "A risque" in segments_dispo else 0)

    df_seg_filtre = clients_f[clients_f["Segment"] == seg_choisi].sort_values("Score_Risque", ascending=False).copy()
    nb_seg = len(df_seg_filtre)
    ca25_seg = df_seg_filtre["CA_2025"].sum()
    ca24_seg = df_seg_filtre["CA_2024"].sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clients", nb_seg)
    m2.metric("CA 2025", f"{ca25_seg:,.0f} EUR")
    m3.metric("CA 2024", f"{ca24_seg:,.0f} EUR")
    evol_seg = (ca25_seg - ca24_seg) / ca24_seg * 100 if ca24_seg else 0
    m4.metric("Evolution", f"{evol_seg:+.1f}%")

    colonnes_affichage = ["Client", "Entite", "Ville", "Niveau_Risque", "CA_2025", "CA_2024", "Delta_EUR", "Evolution_PCT", "CA_Loc_2025", "CA_Vte_2025", "Nb_Commandes_25", "Nb_Familles_25", "Jours_Sans_Commande"]
    colonnes_presentes = [c for c in colonnes_affichage if c in df_seg_filtre.columns]
    df_affiche = df_seg_filtre[colonnes_presentes].reset_index(drop=True)
    rename_map = {
        "Niveau_Risque": "Niveau risque",
        "CA_2025": "CA 2025",
        "CA_2024": "CA 2024",
        "Delta_EUR": "Delta EUR",
        "Evolution_PCT": "Evol %",
        "CA_Loc_2025": "Loc 2025",
        "CA_Vte_2025": "Vte 2025",
        "Nb_Commandes_25": "Cmd 2025",
        "Nb_Familles_25": "Familles",
        "Jours_Sans_Commande": "Jours sans cmd"
    }
    df_affiche = df_affiche.rename(columns=rename_map)
    st.dataframe(df_affiche, use_container_width=True, height=450)
    csv_download(df_affiche, f"clients_{seg_choisi.lower().replace(' ', '_')}.csv")

    if seg_choisi == "A risque":
        insight(f"Ces {nb_seg} clients ont perdu plus de 30% de CA avec un impact superieur a 500 EUR. Ils representent {ca25_seg:,.0f} EUR de CA 2025 a securiser et {ca24_seg - ca25_seg:,.0f} EUR de CA perdu a rattraper.", "⚠️")
    elif seg_choisi == "Perdu":
        insight(f"Ces {nb_seg} clients etaient actifs en 2024 ({ca24_seg:,.0f} EUR) et n'ont passe aucune commande en 2025. C'est l'opportunite de reconquete la plus directe.", "🎯")
    elif seg_choisi == "En baisse":
        insight(f"Ces {nb_seg} clients sont encore actifs mais montrent un desengagement ({ca24_seg - ca25_seg:,.0f} EUR de CA perdu). Intervenir avant qu'ils ne basculent en 'Perdu' est cle.")
    elif seg_choisi == "En croissance":
        insight(f"Ces {nb_seg} clients sont en progression ({ca25_seg - ca24_seg:,.0f} EUR de CA gagne). Ce sont des reussites a fideliser et a developper.")
    elif seg_choisi == "Stable":
        insight(f"Ces {nb_seg} clients ont un CA stable entre 2024 et 2025. Le portefeuille socle est solide.")
    elif seg_choisi == "Nouveau":
        insight(f"Ces {nb_seg} nouveaux clients generent {ca25_seg:,.0f} EUR en 2025. L'enjeu est de les fideliser pour eviter qu'ils ne deviennent des 'one-shot'.")

    st.markdown("---")

    # Scatter
    st.subheader("Carte des clients (CA 2024 vs CA 2025)")
    actifs = clients_f[(clients_f["CA_2024"] > 0) | (clients_f["CA_2025"] > 0)].copy()
    fig_scatter = px.scatter(
        actifs, x="CA_2024", y="CA_2025",
        color="Segment", color_discrete_map=COLORS_SEG,
        hover_name="Client",
        size=actifs[["CA_2024", "CA_2025"]].max(axis=1).clip(lower=100),
        labels={"CA_2024": "CA 2024 (EUR)", "CA_2025": "CA 2025 (EUR)"},
        size_max=30
    )
    max_val = max(actifs["CA_2024"].max(), actifs["CA_2025"].max())
    fig_scatter.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", line=dict(dash="dash", color=C["secondary"]),
        name="Stable", showlegend=False
    ))
    fig_scatter.update_layout(height=500, margin=dict(t=20, b=20))
    st.plotly_chart(fig_scatter, use_container_width=True)
    nb_dessous = ((actifs["CA_2025"] < actifs["CA_2024"]) & (actifs["CA_2024"] > 0)).sum()
    nb_dessus = ((actifs["CA_2025"] > actifs["CA_2024"]) & (actifs["CA_2024"] > 0)).sum()
    insight(f"Les clients en dessous de la ligne pointillee sont en baisse ({nb_dessous}), ceux au-dessus sont en hausse ({nb_dessus}). Les points sur l'axe horizontal (tout a droite, CA 2025 = 0) sont les clients perdus.")

    # Opportunite reconquete
    st.markdown("---")
    st.subheader("Opportunite de reconquete")

    perdus_df = clients_f[clients_f["Segment"] == "Perdu"].sort_values("CA_2024", ascending=False).copy()
    en_baisse = clients_f[clients_f["Segment"].isin(["A risque", "En baisse"])].copy()
    en_baisse["Delta"] = en_baisse["CA_2024"] - en_baisse["CA_2025"]
    en_baisse = en_baisse.sort_values("Score_Risque", ascending=False)

    ca_perdus = perdus_df["CA_2024"].sum()
    ca_delta = en_baisse["Delta"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("CA clients perdus (reconquete)", f"{ca_perdus:,.0f} EUR", f"{len(perdus_df)} clients")
    c2.metric("CA clients en baisse (rattrapage)", f"{ca_delta:,.0f} EUR", f"{len(en_baisse)} clients")
    score_moy = clients_f[clients_f["CA_2025"] > 0]["Score_Risque"].mean()
    c3.metric("Score de risque moyen (actifs)", f"{score_moy:.0f} / 100")

    insight(f"L'opportunite totale de reconquete et rattrapage represente {ca_perdus + ca_delta:,.0f} EUR. Les clients perdus ({ca_perdus:,.0f} EUR) sont les plus urgents a recontacter. Les clients en baisse ({ca_delta:,.0f} EUR) montrent un desengagement qu'il est encore possible d'inverser.", "🎯")

    tab1, tab2, tab3 = st.tabs(["Clients perdus", "Clients en baisse / a risque", "Tous les clients (niveau de risque)"])

    with tab1:
        st.markdown(f"**{len(perdus_df)} clients** -CA 2024 perdu : **{ca_perdus:,.0f} EUR**")
        # Ajouter info lien avec suppression commerciaux
        perdus_export = perdus_df[["Client", "Entite", "Ville", "CA_2024", "CA_Loc_2024", "CA_Vte_2024", "Derniere_Commande", "Jours_Sans_Commande"]].copy()
        perdus_export["Derniere_Commande"] = perdus_export["Derniere_Commande"].dt.strftime("%d/%m/%Y").fillna("Inconnue")
        perdus_export["Lien_Commerciaux"] = perdus_df["Derniere_Commande"].apply(
            lambda d: "⚠️ OUI (derniere cmd avant juin 2024)" if pd.notna(d) and d < pd.Timestamp("2024-06-01") else ("Non (derniere cmd apres juin 2024)" if pd.notna(d) else "?")
        )
        perdus_export.columns = ["Client", "Entite", "Ville", "CA 2024", "Loc 2024", "Vte 2024", "Derniere commande", "Jours sans cmd", "Lien suppression commerciaux"]
        st.dataframe(perdus_export.reset_index(drop=True), use_container_width=True, height=400)

        nb_avant_juin = (perdus_df["Derniere_Commande"] < pd.Timestamp("2024-06-01")).sum()
        nb_apres_juin = (perdus_df["Derniere_Commande"] >= pd.Timestamp("2024-06-01")).sum()
        ca_avant = perdus_df[perdus_df["Derniere_Commande"] < pd.Timestamp("2024-06-01")]["CA_2024"].sum()
        insight(f"Sur les {len(perdus_df)} clients perdus, **{nb_avant_juin}** ont passe leur derniere commande AVANT juin 2024 (suppression des commerciaux), representant {ca_avant:,.0f} EUR. **{nb_apres_juin}** ont commande apres. Ceux dont la derniere commande precede la suppression des commerciaux sont probablement directement lies a ce changement.", "⚠️")
        csv_download(perdus_export, "clients_perdus_reconquete.csv")

    with tab2:
        st.markdown(f"**{len(en_baisse)} clients** -CA a rattraper : **{ca_delta:,.0f} EUR** -Tries par niveau de risque")
        df_baisse = en_baisse[["Client", "Entite", "Segment", "Niveau_Risque", "CA_2025", "CA_2024", "Delta", "Evolution_PCT", "Nb_Commandes_25", "Jours_Sans_Commande"]].reset_index(drop=True)
        df_baisse.columns = ["Client", "Entite", "Categorie", "Niveau risque", "CA 2025", "CA 2024", "CA perdu", "Evol %", "Commandes 25", "Jours sans cmd"]
        st.dataframe(df_baisse, use_container_width=True, height=400)
        csv_download(df_baisse, "clients_en_baisse_a_risque.csv")

    with tab3:
        st.markdown("**Tous les clients actifs en 2025, tries par niveau de risque decroissant**")
        tous = clients_f[clients_f["CA_2025"] > 0].sort_values("Score_Risque", ascending=False)
        df_tous = tous[["Client", "Entite", "Segment", "Niveau_Risque", "CA_2025", "CA_2024", "Delta_EUR", "Nb_Commandes_25", "Nb_Familles_25", "Jours_Sans_Commande"]].reset_index(drop=True)
        df_tous.columns = ["Client", "Entite", "Categorie", "Niveau risque", "CA 2025", "CA 2024", "Delta EUR", "Commandes 25", "Familles 25", "Jours sans cmd"]
        st.dataframe(df_tous, use_container_width=True, height=500)
        csv_download(df_tous, "tous_clients_par_risque.csv")

    # Resume des alertes (remplace l'histogramme technique)
    st.markdown("---")
    st.subheader("Resume des alertes")
    actifs_score = clients_f[clients_f["CA_2025"] > 0]

    nb_critique = (actifs_score["Score_Risque"] >= 70).sum()
    ca_critique = actifs_score[actifs_score["Score_Risque"] >= 70]["CA_2025"].sum()
    nb_eleve = ((actifs_score["Score_Risque"] >= 50) & (actifs_score["Score_Risque"] < 70)).sum()
    ca_eleve = actifs_score[(actifs_score["Score_Risque"] >= 50) & (actifs_score["Score_Risque"] < 70)]["CA_2025"].sum()
    nb_modere = ((actifs_score["Score_Risque"] >= 30) & (actifs_score["Score_Risque"] < 50)).sum()
    ca_modere = actifs_score[(actifs_score["Score_Risque"] >= 30) & (actifs_score["Score_Risque"] < 50)]["CA_2025"].sum()
    nb_faible = (actifs_score["Score_Risque"] < 30).sum()
    ca_faible = actifs_score[actifs_score["Score_Risque"] < 30]["CA_2025"].sum()

    al1, al2, al3, al4 = st.columns(4)
    al1.markdown(f"### 🔴 {nb_critique}")
    al1.caption(f"Critiques -{ca_critique:,.0f} EUR")
    al2.markdown(f"### 🟠 {nb_eleve}")
    al2.caption(f"Eleves -{ca_eleve:,.0f} EUR")
    al3.markdown(f"### 🟡 {nb_modere}")
    al3.caption(f"Moderes -{ca_modere:,.0f} EUR")
    al4.markdown(f"### 🟢 {nb_faible}")
    al4.caption(f"Faibles -{ca_faible:,.0f} EUR")

    insight(f"**{nb_critique} clients critiques** ({ca_critique:,.0f} EUR) necessitent une action immediate (appel, visite). **{nb_eleve} clients a risque eleve** ({ca_eleve:,.0f} EUR) meritent un suivi renforce. Au total, {nb_critique + nb_eleve} clients representant {ca_critique + ca_eleve:,.0f} EUR de CA sont en zone de danger.", "⚠️")

    # --- Top 15 plus fortes baisses ---
    st.markdown("---")
    st.subheader("Top 15 plus fortes baisses de CA (clients encore actifs)")

    actifs_2ans = clients_f[(clients_f["CA_2024"] > 0) & (clients_f["CA_2025"] > 0)].copy()
    actifs_2ans["Delta"] = actifs_2ans["CA_2025"] - actifs_2ans["CA_2024"]
    top_baisses = actifs_2ans.sort_values("Delta").head(15)

    fig_baisses = px.bar(
        top_baisses, x="Delta", y="Client", orientation="h",
        color_discrete_sequence=[C["danger"]],
        labels={"Delta": "Evolution CA (EUR)", "Client": ""}
    )
    fig_baisses.update_layout(height=500, margin=dict(t=20, b=20, l=300), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_baisses, use_container_width=True)

    ca_top_baisses = top_baisses["Delta"].sum()
    insight(f"Ces 15 clients ont perdu {abs(ca_top_baisses):,.0f} EUR de CA entre 2024 et 2025 tout en restant actifs. Ce sont des clients en desengagement progressif : concentrez vos appels ici avant qu'ils ne partent completement.", "⚠️")

    df_export_baisses = top_baisses[["Client", "Entite", "Ville", "CA_2024", "CA_2025", "Delta", "Nb_Commandes_25"]].copy()
    df_export_baisses.columns = ["Client", "Entite", "Ville", "CA 2024", "CA 2025", "Perte EUR", "Commandes 2025"]
    csv_download(df_export_baisses, "top15_baisses.csv")

    # --- Top 15 plus fortes hausses ---
    st.subheader("Top 15 plus fortes hausses de CA")
    top_hausses = actifs_2ans.sort_values("Delta", ascending=False).head(15)

    fig_hausses = px.bar(
        top_hausses, x="Delta", y="Client", orientation="h",
        color_discrete_sequence=[C["primary"]],
        labels={"Delta": "Evolution CA (EUR)", "Client": ""}
    )
    fig_hausses.update_layout(height=500, margin=dict(t=20, b=20, l=300), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_hausses, use_container_width=True)

    ca_top_hausses = top_hausses["Delta"].sum()
    insight(f"Ces 15 clients ont gagne {ca_top_hausses:,.0f} EUR de CA. Ce sont des reussites a comprendre et reproduire : qu'est-ce qui a motive leur croissance ?")

    # --- Distribution des evolutions ---
    st.markdown("---")
    st.subheader("Repartition des evolutions client (2024 vs 2025)")

    actifs_2ans["Evolution_Tranche"] = pd.cut(
        actifs_2ans["Evolution_PCT"],
        bins=[-float("inf"), -50, -30, -10, 0, 10, 30, 50, float("inf")],
        labels=["< -50%", "-50 a -30%", "-30 a -10%", "-10 a 0%", "0 a +10%", "+10 a +30%", "+30 a +50%", "> +50%"]
    )
    evol_dist = actifs_2ans["Evolution_Tranche"].value_counts().sort_index().reset_index()
    evol_dist.columns = ["Tranche", "Nb clients"]

    fig_dist = px.bar(
        evol_dist, x="Tranche", y="Nb clients",
        color_discrete_sequence=[C["primary"]],
        labels={"Tranche": "Evolution CA (%)", "Nb clients": "Nombre de clients"}
    )
    fig_dist.update_layout(height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig_dist, use_container_width=True)

    nb_en_baisse = (actifs_2ans["Evolution_PCT"] < 0).sum()
    nb_en_hausse = (actifs_2ans["Evolution_PCT"] > 0).sum()
    pct_baisse = nb_en_baisse / len(actifs_2ans) * 100 if len(actifs_2ans) else 0
    insight(f"Parmi les {len(actifs_2ans)} clients actifs les deux annees, {nb_en_baisse} ({pct_baisse:.0f}%) sont en baisse et {nb_en_hausse} ({100-pct_baisse:.0f}%) en hausse.")


# ============================================================
# ONGLET 3 : IMPACT COMMERCIAUX
# ============================================================
elif onglet == "Impact Commerciaux":
    st.title("Impact de la Suppression des Commerciaux")
    st.markdown("*Commerciaux terrain supprimes en juin 2024. Relations desormais par telephone uniquement.*")
    st.markdown("---")

    ventes_impact = ventes_f

    # CA mensuel
    ventes_mois = ventes_impact.groupby("Mois").agg(
        CA=("Total_HT", "sum"),
        Nb_Factures=("Facture", "nunique"),
        Nb_Clients=("Client", "nunique"),
        Nb_Lignes=("Facture", "count")
    ).reset_index()
    ventes_mois = ventes_mois.sort_values("Mois")
    ventes_mois["Panier_Moyen"] = ventes_mois["CA"] / ventes_mois["Nb_Factures"]

    # Periode
    ventes_mois["Periode"] = ventes_mois["Mois"].apply(
        lambda x: "Avant (jan-mai 2024)" if x <= "2024-05"
        else ("Transition (juin-dec 2024)" if x <= "2024-12" else "Apres (2025)")
    )

    avant = ventes_mois[ventes_mois["Periode"] == "Avant (jan-mai 2024)"]
    transition = ventes_mois[ventes_mois["Periode"] == "Transition (juin-dec 2024)"]
    apres = ventes_mois[ventes_mois["Periode"] == "Apres (2025)"]

    # Courbe CA avec ligne verticale
    st.subheader("CA mensuel avec rupture juin 2024")
    fig_ca = px.bar(
        ventes_mois, x="Mois", y="CA",
        color="Periode",
        color_discrete_map={
            "Avant (jan-mai 2024)": C["secondary"],
            "Transition (juin-dec 2024)": C["warning"],
            "Apres (2025)": C["primary"]
        },
        labels={"CA": "CA HT (EUR)", "Mois": ""}
    )
    fig_ca.add_shape(type="line", x0="2024-06", x1="2024-06", y0=0, y1=1, yref="paper", line=dict(dash="dash", color=C["danger"], width=2))
    fig_ca.add_annotation(x="2024-06", y=1, yref="paper", text="Suppression commerciaux", showarrow=False, font=dict(color=C["danger"], size=11), yshift=10)
    fig_ca.update_layout(height=400, margin=dict(t=40, b=20))
    st.plotly_chart(fig_ca, use_container_width=True)

    ca_avant = avant["CA"].mean() if len(avant) else 0
    ca_transition = transition["CA"].mean() if len(transition) else 0
    ca_apres = apres["CA"].mean() if len(apres) else 0
    evol_transition = (ca_transition - ca_avant) / ca_avant * 100 if ca_avant else 0
    evol_apres = (ca_apres - ca_avant) / ca_avant * 100 if ca_avant else 0
    insight(f"Le CA mensuel moyen est passe de {ca_avant:,.0f} EUR (avant) a {ca_transition:,.0f} EUR (transition, {evol_transition:+.1f}%) puis {ca_apres:,.0f} EUR (2025, {evol_apres:+.1f}%). {'La suppression des commerciaux semble avoir eu un impact negatif.' if evol_transition < -5 else 'Le CA se maintient malgre la suppression des commerciaux.' if abs(evol_transition) <= 5 else 'Le CA a progresse apres la suppression des commerciaux.'}")

    # Clients actifs par mois
    st.subheader("Nombre de clients actifs par mois")
    fig_clients = px.line(
        ventes_mois, x="Mois", y="Nb_Clients",
        markers=True, color_discrete_sequence=[C["primary"]],
        labels={"Nb_Clients": "Clients actifs", "Mois": ""}
    )
    fig_clients.add_shape(type="line", x0="2024-06", x1="2024-06", y0=0, y1=1, yref="paper", line=dict(dash="dash", color=C["danger"], width=2))
    fig_clients.add_annotation(x="2024-06", y=1, yref="paper", text="Suppression commerciaux", showarrow=False, font=dict(color=C["danger"], size=11), yshift=10)
    fig_clients.update_layout(height=350, margin=dict(t=40, b=20))
    st.plotly_chart(fig_clients, use_container_width=True)

    clients_avant = avant["Nb_Clients"].mean() if len(avant) else 0
    clients_transition = transition["Nb_Clients"].mean() if len(transition) else 0
    evol_clients = (clients_transition - clients_avant) / clients_avant * 100 if clients_avant else 0
    insight(f"Le nombre moyen de clients actifs par mois est passe de {clients_avant:.0f} a {clients_transition:.0f} apres la suppression des commerciaux ({evol_clients:+.1f}%). {'C\'est un signal d\'erosion du portefeuille.' if evol_clients < -5 else 'Le nombre de clients reste stable.'}")

    # Comparaison avant/apres
    st.subheader("Comparaison avant / apres")

    comp_data = pd.DataFrame({
        "Indicateur": ["CA mensuel moyen", "Clients actifs / mois", "Factures / mois", "Panier moyen"],
        "Avant (jan-mai 24)": [
            avant["CA"].mean(),
            avant["Nb_Clients"].mean(),
            avant["Nb_Factures"].mean(),
            avant["Panier_Moyen"].mean()
        ],
        "Transition (juin-dec 24)": [
            transition["CA"].mean(),
            transition["Nb_Clients"].mean(),
            transition["Nb_Factures"].mean(),
            transition["Panier_Moyen"].mean()
        ],
        "Apres (2025)": [
            apres["CA"].mean(),
            apres["Nb_Clients"].mean(),
            apres["Nb_Factures"].mean(),
            apres["Panier_Moyen"].mean()
        ]
    })

    for col in ["Avant (jan-mai 24)", "Transition (juin-dec 24)", "Apres (2025)"]:
        comp_data[col] = comp_data[col].apply(lambda x: f"{x:,.0f}")

    st.dataframe(comp_data, use_container_width=True, hide_index=True)

    entites_impact_txt = " + ".join(filtre_entite)
    if "EMS" in filtre_entite and "STC" in filtre_entite:
        insight(f"Ce tableau compare les moyennes mensuelles sur 3 periodes ({entites_impact_txt}). Note : 2024 ne contient que STC dans les ventes. Pour une comparaison fiable, selectionnez uniquement STC dans la barre laterale.", "⚠️")
    else:
        insight(f"Ce tableau compare les moyennes mensuelles sur 3 periodes ({entites_impact_txt}).")

    # Top clients impactes
    st.markdown("---")
    st.subheader("Clients les plus impactes (S1 vs S2 2024)")

    ventes_client = ventes_impact[ventes_impact["Annee"] == 2024].copy()
    ventes_client["Semestre"] = ventes_client["Date"].dt.month.apply(lambda m: "S1" if m <= 6 else "S2")

    ca_sem = ventes_client.groupby(["Client", "Semestre"])["Total_HT"].sum().unstack(fill_value=0)
    if "S1" in ca_sem.columns and "S2" in ca_sem.columns:
        ca_sem["Delta"] = ca_sem["S2"] - ca_sem["S1"]
        ca_sem["Evolution"] = np.where(ca_sem["S1"] > 0, ca_sem["Delta"] / ca_sem["S1"] * 100, np.nan)
        ca_sem = ca_sem.sort_values("Delta")

        # Chiffre cle global
        total_delta_s1s2 = ca_sem["Delta"].sum()
        nb_en_baisse_s1s2 = (ca_sem["Delta"] < 0).sum()
        ca_perdu_s1s2 = ca_sem[ca_sem["Delta"] < 0]["Delta"].sum()

        kd1, kd2, kd3 = st.columns(3)
        kd1.metric("CA total perdu S1 → S2", f"{abs(ca_perdu_s1s2):,.0f} EUR", f"{nb_en_baisse_s1s2} clients en baisse")
        kd2.metric("Solde net S1 → S2", f"{total_delta_s1s2:+,.0f} EUR")
        kd3.metric("CA moyen perdu / client", f"{abs(ca_perdu_s1s2) / nb_en_baisse_s1s2:,.0f} EUR" if nb_en_baisse_s1s2 else "—")

        st.markdown("**Clients ayant le plus perdu entre S1 et S2 2024 (apres suppression commerciaux)**")
        top_impact = ca_sem.head(20).reset_index()
        top_impact.columns = ["Client", "CA S1 2024", "CA S2 2024", "Delta EUR", "Evolution %"]
        st.dataframe(top_impact, use_container_width=True, height=400)
        csv_download(top_impact, "clients_impactes_commerciaux.csv")

        insight(f"Au total, **{abs(ca_perdu_s1s2):,.0f} EUR** de CA ont ete perdus entre le S1 et le S2 2024 chez les {nb_en_baisse_s1s2} clients en baisse. C'est la meilleure mesure de l'impact des commerciaux : meme annee, meme perimetre, seul le mode de relation client a change.", "⚠️")



# ============================================================
# ONGLET 4 : PRODUITS & FAMILLES
# ============================================================
elif onglet == "Produits & Familles":
    st.title("Analyse Produits & Familles")
    st.markdown("---")

    annee = st.selectbox("Annee", [2025, 2024])
    ventes_a = ventes_f[ventes_f["Annee"] == annee]
    ventes_comp = ventes_f

    # Top familles
    st.subheader(f"CA par rayon ({annee})")
    ca_rayon = ventes_a.groupby("Rayon")["Total_HT"].sum().reset_index().sort_values("Total_HT", ascending=True)
    ca_rayon = ca_rayon[ca_rayon["Rayon"] != "Autre"]
    fig_rayon = px.bar(
        ca_rayon, x="Total_HT", y="Rayon", orientation="h",
        color_discrete_sequence=[C["primary"]],
        labels={"Total_HT": "CA HT (EUR)", "Rayon": ""}
    )
    fig_rayon.update_layout(height=500, margin=dict(t=20, b=20, l=200))
    st.plotly_chart(fig_rayon, use_container_width=True)

    top_rayon = ca_rayon.iloc[-1]
    top2_rayon = ca_rayon.iloc[-2] if len(ca_rayon) > 1 else None
    pct_top = top_rayon["Total_HT"] / ca_rayon["Total_HT"].sum() * 100
    top_rayon_nom = top_rayon["Rayon"]
    top_rayon_ca = top_rayon["Total_HT"]
    if top2_rayon is not None:
        top2_nom = top2_rayon["Rayon"]
        top2_ca = top2_rayon["Total_HT"]
        txt_rayon = f'Le rayon "{top_rayon_nom}" domine avec {top_rayon_ca:,.0f} EUR ({pct_top:.0f}% du CA). Suivi par "{top2_nom}" ({top2_ca:,.0f} EUR).'
    else:
        txt_rayon = f'Le rayon "{top_rayon_nom}" domine avec {top_rayon_ca:,.0f} EUR ({pct_top:.0f}% du CA).'
    insight(txt_rayon)

    # Top articles
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Top 15 articles par CA ({annee})")
        top_art = ventes_a.groupby(["Code_Article", "Libelle"]).agg(
            CA=("Total_HT", "sum"),
            Qte=("Qte", "sum")
        ).reset_index().sort_values("CA", ascending=False).head(15)
        st.dataframe(top_art, use_container_width=True, hide_index=True)
        csv_download(top_art, f"top15_articles_CA_{annee}.csv")

    with col2:
        st.subheader(f"Top 15 articles par volume ({annee})")
        top_vol = ventes_a.groupby(["Code_Article", "Libelle"]).agg(
            Qte=("Qte", "sum"),
            CA=("Total_HT", "sum")
        ).reset_index().sort_values("Qte", ascending=False).head(15)
        st.dataframe(top_vol, use_container_width=True, hide_index=True)
        csv_download(top_vol, f"top15_articles_volume_{annee}.csv")

    # Evolution familles 2024 vs 2025
    st.markdown("---")
    st.subheader("Evolution par famille (2024 vs 2025)")

    ca_fam_24 = ventes_comp[ventes_comp["Annee"] == 2024].groupby("Libelle_Famille")["Total_HT"].sum()
    ca_fam_25 = ventes_comp[ventes_comp["Annee"] == 2025].groupby("Libelle_Famille")["Total_HT"].sum()
    evol_fam = pd.DataFrame({"CA_2024": ca_fam_24, "CA_2025": ca_fam_25}).fillna(0)
    evol_fam["Evolution_EUR"] = evol_fam["CA_2025"] - evol_fam["CA_2024"]
    evol_fam["Evolution_PCT"] = np.where(evol_fam["CA_2024"] > 0, evol_fam["Evolution_EUR"] / evol_fam["CA_2024"] * 100, np.nan)
    evol_fam = evol_fam.sort_values("Evolution_EUR", ascending=False).reset_index()
    evol_fam.columns = ["Famille", "CA 2024", "CA 2025", "Evolution EUR", "Evolution %"]

    st.dataframe(evol_fam, use_container_width=True, height=500)
    csv_download(evol_fam, "evolution_familles.csv")

    top_hausse = evol_fam.iloc[0] if len(evol_fam) else None
    top_baisse_fam = evol_fam.iloc[-1] if len(evol_fam) else None
    if top_hausse is not None and top_baisse_fam is not None:
        insight(f"Plus forte hausse : \"{top_hausse['Famille']}\" ({top_hausse['Evolution EUR']:+,.0f} EUR). Plus forte baisse : \"{top_baisse_fam['Famille']}\" ({top_baisse_fam['Evolution EUR']:+,.0f} EUR). Les familles en baisse meritent une attention particuliere : desengagement client ou probleme de gamme ?")

    # --- Quels clients achetent quels rayons ? ---
    st.markdown("---")
    st.subheader("Quels clients achetent quels rayons ? (2025)")
    st.markdown("*Pour chaque client, les rayons achetes et ceux qui manquent -utile pour cibler vos appels.*")

    ventes_25 = ventes_f[ventes_f["Annee"] == 2025]
    rayons_par_client = ventes_25.groupby(["Client", "Rayon"])["Total_HT"].sum().unstack(fill_value=0)
    rayons_par_client = rayons_par_client.drop(columns=["Autre"], errors="ignore")
    all_rayons = sorted(rayons_par_client.columns.tolist())

    # Pour chaque client, lister les rayons manquants
    client_rayons_info = []
    for client_name in rayons_par_client.index:
        row = rayons_par_client.loc[client_name]
        achetes = [r for r in all_rayons if row[r] > 0]
        manquants = [r for r in all_rayons if row[r] == 0]
        ca_total = row.sum()
        client_rayons_info.append({
            "Client": client_name,
            "CA 2025": ca_total,
            "Nb rayons": len(achetes),
            "Rayons achetes": ", ".join(achetes),
            "Rayons manquants": ", ".join(manquants) if manquants else "Tous couverts"
        })

    df_rayons_client = pd.DataFrame(client_rayons_info).sort_values("CA 2025", ascending=False)
    st.dataframe(df_rayons_client.head(50), use_container_width=True, height=500, hide_index=True)
    csv_download(df_rayons_client, "clients_rayons_detail.csv", "📥 Telecharger la fiche complete (CSV)")

    moy_rayons_client = df_rayons_client["Nb rayons"].mean()
    insight(f"En moyenne, chaque client achete {moy_rayons_client:.1f} rayons sur {len(all_rayons)} disponibles. La colonne 'Rayons manquants' vous donne directement la fiche d'appel : pour chaque client, vous savez quoi proposer.", "🎯")

    # Effet remise par article (simplifie)
    st.markdown("---")
    st.subheader("Politique de remise par article")
    st.markdown("*Est-ce que vos clients obtiennent un meilleur prix quand ils commandent en plus grande quantite ?*")

    vol_prix = ventes_f[(ventes_f["Qte"] > 0) & (ventes_f["PU_HT"] > 0) & (ventes_f["Total_HT"] > 0)].copy()

    article_counts = vol_prix.groupby("Libelle").agg(
        Nb_Transactions=("Facture", "count"),
        CA=("Total_HT", "sum"),
        PU_Min=("PU_HT", "min"),
        PU_Max=("PU_HT", "max"),
        PU_Moyen=("PU_HT", "mean"),
        Qte_Moy=("Qte", "mean")
    ).reset_index()
    article_counts["Ecart_PU"] = article_counts["PU_Max"] - article_counts["PU_Min"]
    article_counts["Variation_PU_PCT"] = (article_counts["Ecart_PU"] / article_counts["PU_Moyen"] * 100).round(1)
    articles_analysables = article_counts[(article_counts["Nb_Transactions"] >= 10) & (article_counts["Ecart_PU"] > 0)].copy()

    correlations = []
    for art in articles_analysables["Libelle"]:
        data_art = vol_prix[vol_prix["Libelle"] == art]
        if len(data_art) >= 10:
            corr_val = data_art[["Qte", "PU_HT"]].corr().iloc[0, 1]
            correlations.append({"Libelle": art, "Corr": corr_val})

    corr_df = pd.DataFrame(correlations).merge(articles_analysables, on="Libelle")
    corr_df = corr_df.sort_values("Corr")

    # Labels simples
    def label_effet_remise(corr):
        if corr < -0.2:
            return "✅ OUI -remise quantite"
        if corr > 0.2:
            return "⬆️ INVERSE -montee en gamme"
        return "➖ NON -prix fixe"

    corr_df["Effet_Remise"] = corr_df["Corr"].apply(label_effet_remise)

    nb_effet_volume = (corr_df["Corr"] < -0.2).sum()
    nb_pas_effet = (corr_df["Corr"].abs() <= 0.2).sum()
    nb_effet_inverse = (corr_df["Corr"] > 0.2).sum()

    k1, k2, k3 = st.columns(3)
    k1.metric("✅ Remise quantite", f"{nb_effet_volume} articles", "Le prix baisse quand on commande plus")
    k2.metric("➖ Prix fixe", f"{nb_pas_effet} articles", "Prix stable quelle que soit la quantite")
    k3.metric("⬆️ Montee en gamme", f"{nb_effet_inverse} articles", "Le client prend du plus haut de gamme")

    insight(f"Sur {len(corr_df)} articles analyses, {nb_effet_volume} ont une vraie remise quantite (consommables en gros), {nb_pas_effet} ont un prix fixe (equipements unitaires), et {nb_effet_inverse} voient le prix monter (le client prend des formats plus grands ou plus haut de gamme quand il commande plus).")

    # Tableau des articles
    st.markdown("---")

    tab_vol1, tab_vol2, tab_vol3 = st.tabs(["✅ Remise quantite", "➖ Prix fixe", "⬆️ Montee en gamme"])

    with tab_vol1:
        st.markdown("**Articles ou le prix baisse quand la pharmacie commande en grande quantite**")
        df_vol = corr_df[corr_df["Corr"] < -0.2].sort_values("Corr")
        df_show = df_vol[["Libelle", "Effet_Remise", "Nb_Transactions", "CA", "PU_Min", "PU_Max", "PU_Moyen", "Variation_PU_PCT"]].reset_index(drop=True)
        df_show.columns = ["Article", "Effet remise", "Transactions", "CA total", "PU Min", "PU Max", "PU Moyen", "Ecart prix %"]
        st.dataframe(df_show, use_container_width=True, hide_index=True, height=400)

    with tab_vol2:
        st.markdown("**Articles ou le prix reste fixe quelle que soit la quantite**")
        df_stable = corr_df[corr_df["Corr"].abs() <= 0.2].sort_values("CA", ascending=False)
        df_show2 = df_stable[["Libelle", "Effet_Remise", "Nb_Transactions", "CA", "PU_Min", "PU_Max", "PU_Moyen", "Variation_PU_PCT"]].reset_index(drop=True)
        df_show2.columns = ["Article", "Effet remise", "Transactions", "CA total", "PU Min", "PU Max", "PU Moyen", "Ecart prix %"]
        st.dataframe(df_show2, use_container_width=True, hide_index=True, height=400)

    with tab_vol3:
        st.markdown("**Articles ou le client prend du plus haut de gamme quand il commande plus**")
        df_inv = corr_df[corr_df["Corr"] > 0.2].sort_values("Corr", ascending=False)
        df_show3 = df_inv[["Libelle", "Effet_Remise", "Nb_Transactions", "CA", "PU_Min", "PU_Max", "PU_Moyen", "Variation_PU_PCT"]].reset_index(drop=True)
        df_show3.columns = ["Article", "Effet remise", "Transactions", "CA total", "PU Min", "PU Max", "PU Moyen", "Ecart prix %"]
        st.dataframe(df_show3, use_container_width=True, hide_index=True, height=400)

    with st.expander("Comprendre ces resultats"):
        st.markdown(f"""
**En resume :**

- **Les {nb_effet_volume} articles avec remise quantite** sont principalement des **consommables** commandes en lot (masques, compresses, gants, kits nebuliseur). Plus la pharmacie commande, meilleur est le prix. C'est normal.

- **Les {nb_pas_effet} articles a prix fixe** sont surtout des **equipements** (fauteuils, lits, deambulateurs). Ce sont des achats unitaires, donc le prix ne bouge pas.

- **Les {nb_effet_inverse} articles a effet inverse** ne posent pas de probleme. Ce sont des produits ou un meme nom couvre plusieurs tailles/variantes. Quand la pharmacie commande plus, elle prend aussi des formats plus grands, donc le prix moyen monte mecaniquement.
        """)

    # Analyse detaillee par article
    st.markdown("---")
    st.subheader("Detail par article")

    articles_pop = corr_df.sort_values("CA", ascending=False)["Libelle"].head(30).tolist()
    article_sel = st.selectbox("Choisir un article", articles_pop)

    if article_sel:
        art_data = vol_prix[(vol_prix["Libelle"] == article_sel)]
        corr_art = art_data[["Qte", "PU_HT"]].corr().iloc[0, 1]
        effet_art = label_effet_remise(corr_art)

        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
        col_a1.metric("Transactions", len(art_data))
        col_a2.metric("PU min / max", f"{art_data['PU_HT'].min():.2f} / {art_data['PU_HT'].max():.2f}")
        col_a3.metric("Effet remise", effet_art.split("—")[0].strip())
        col_a4.metric("CA total", f"{art_data['Total_HT'].sum():,.0f} EUR")

        # Scatter simple (sans trendline technique)
        fig_art = px.scatter(
            art_data, x="Qte", y="PU_HT",
            color="Annee",
            color_discrete_map=COLORS_ANNEE_INT,
            labels={"Qte": "Quantite commandee", "PU_HT": "Prix unitaire HT (EUR)"}
        )
        fig_art.update_layout(height=400, margin=dict(t=20, b=20))
        st.plotly_chart(fig_art, use_container_width=True)

        # Tableau par tranche de quantite
        art_data_c = art_data.copy()
        art_data_c["Tranche_Qte"] = pd.cut(art_data_c["Qte"], bins=[0, 1, 5, 10, 20, 50, 1000], labels=["1", "2-5", "6-10", "11-20", "21-50", "50+"])
        tranche_stats = art_data_c.groupby("Tranche_Qte", observed=True).agg(
            Nb_Commandes=("Facture", "count"),
            PU_Moyen=("PU_HT", "mean"),
            Qte_Totale=("Qte", "sum"),
            CA=("Total_HT", "sum")
        ).reset_index()
        tranche_stats.columns = ["Tranche Qte", "Nb commandes", "PU moyen", "Qte totale", "CA"]

        st.markdown("**Prix moyen par tranche de quantite**")
        st.dataframe(tranche_stats, use_container_width=True, hide_index=True)

        if corr_art < -0.2:
            ecart_pu = art_data_c.groupby("Tranche_Qte", observed=True)["PU_HT"].mean()
            if len(ecart_pu) >= 2:
                pu_petit = ecart_pu.iloc[0]
                pu_gros = ecart_pu.iloc[-1]
                remise_pct = (pu_petit - pu_gros) / pu_petit * 100 if pu_petit else 0
                insight(f"Remise quantite confirmee : le prix passe de {pu_petit:.2f} EUR (petites quantites) a {pu_gros:.2f} EUR (grosses quantites), soit environ {remise_pct:.0f}% de remise.")
            else:
                insight("Cet article a une remise quantite confirmee.")
        elif corr_art > 0.2:
            insight("Le prix monte avec la quantite : le client prend probablement des variantes plus haut de gamme quand il commande en volume.")
        else:
            insight("Le prix est stable quelle que soit la quantite commandee. Pas de politique de remise volume sur cet article.")


# ============================================================
# ONGLET 5 : OPPORTUNITES
# ============================================================
elif onglet == "Opportunites":
    st.title("Opportunites de Developpement")
    st.markdown("---")

    # --- RECAP GLOBAL DU POTENTIEL ---
    st.subheader("Bilan 2024 → 2025 et potentiel")

    # Pertes brutes
    _ca_reconquete = clients_f[(clients_f["CA_2024"] > 0) & (clients_f["CA_2025"] == 0)]["CA_2024"].sum()
    _mask_baisse = (clients_f["CA_2024"] > 0) & (clients_f["CA_2025"] > 0) & (clients_f["CA_2025"] < clients_f["CA_2024"])
    _ca_rattrapage = (clients_f.loc[_mask_baisse, "CA_2024"] - clients_f.loc[_mask_baisse, "CA_2025"]).sum()
    _pertes_brutes = _ca_reconquete + _ca_rattrapage

    # Gains bruts
    _ca_nouveaux = clients_f[(clients_f["CA_2024"] == 0) & (clients_f["CA_2025"] > 0)]["CA_2025"].sum()
    _mask_hausse = (clients_f["CA_2024"] > 0) & (clients_f["CA_2025"] > 0) & (clients_f["CA_2025"] > clients_f["CA_2024"])
    _ca_gains = (clients_f.loc[_mask_hausse, "CA_2025"] - clients_f.loc[_mask_hausse, "CA_2024"]).sum()
    _gains_bruts = _ca_nouveaux + _ca_gains

    # Delta net = ce qu'on voit dans les KPIs (CA 2025 - CA 2024)
    _delta_net = clients_f["CA_2025"].sum() - clients_f["CA_2024"].sum()

    # Cross-sell = estimation theorique
    _pharma_v = ventes_f[(ventes_f["Annee"] == 2025) & (ventes_f["Client"].str.contains("PHARMACIE", case=False, na=False))]
    _cross_b = (_pharma_v.groupby(["Client", "Rayon"])["Total_HT"].sum().unstack(fill_value=0) > 0).astype(int)
    _cross_b = _cross_b.drop(columns=["Autre"], errors="ignore")
    _nb_ray = _cross_b.sum(axis=1)
    _ca_pharma = _pharma_v.groupby("Client")["Total_HT"].sum()
    _pot_pharma = _nb_ray[_nb_ray <= 2]
    _clients_3plus = _nb_ray[_nb_ray >= 3]
    _ca_par_rayon = _ca_pharma[_clients_3plus.index].sum() / _clients_3plus.sum() if _clients_3plus.sum() > 0 else 0
    _ca_crosssell = len(_pot_pharma) * _ca_par_rayon

    # Affichage : decomposition du delta net
    st.markdown("##### D'ou vient la baisse de CA ?")
    rp1, rp2, rp3 = st.columns(3)
    rp1.markdown(f"""<div style="border:2px solid {C['danger']};border-radius:12px;padding:16px;text-align:center">
        <div style="font-size:11px;color:#9CA3AF;font-weight:600;text-transform:uppercase">Pertes brutes</div>
        <div style="font-size:24px;font-weight:700;color:{C['danger']};margin:6px 0">-{_pertes_brutes:,.0f} EUR</div>
        <div style="font-size:12px;color:#6B7280">{_ca_reconquete:,.0f} clients perdus + {_ca_rattrapage:,.0f} en baisse</div>
    </div>""", unsafe_allow_html=True)
    rp2.markdown(f"""<div style="border:2px solid {C['positive']};border-radius:12px;padding:16px;text-align:center">
        <div style="font-size:11px;color:#9CA3AF;font-weight:600;text-transform:uppercase">Gains bruts</div>
        <div style="font-size:24px;font-weight:700;color:{C['positive']};margin:6px 0">+{_gains_bruts:,.0f} EUR</div>
        <div style="font-size:12px;color:#6B7280">{_ca_nouveaux:,.0f} nouveaux + {_ca_gains:,.0f} en hausse</div>
    </div>""", unsafe_allow_html=True)
    rp3.markdown(f"""<div style="border:2px solid {C['neutral']};border-radius:12px;padding:16px;text-align:center">
        <div style="font-size:11px;color:#9CA3AF;font-weight:600;text-transform:uppercase">= SOLDE NET</div>
        <div style="font-size:28px;font-weight:800;color:{C['danger'] if _delta_net < 0 else C['positive']};margin:6px 0">{_delta_net:+,.0f} EUR</div>
        <div style="font-size:12px;color:#6B7280">= CA 2025 - CA 2024</div>
    </div>""", unsafe_allow_html=True)

    insight(f"Vous avez perdu {_pertes_brutes:,.0f} EUR (clients perdus + clients en baisse) mais recupere {_gains_bruts:,.0f} EUR (nouveaux clients + clients en hausse). Resultat net : **{_delta_net:+,.0f} EUR**, soit exactement l'ecart entre votre CA 2025 et 2024.")

    # --- Potentiel de recuperation : 3 scenarios ---
    st.markdown("")
    st.markdown("##### Combien peut-on recuperer ?")

    # Pessimiste : 20% des perdus reviennent, 30% du delta des en baisse rattrape
    _recup_pessi = _ca_reconquete * 0.20 + _ca_rattrapage * 0.30 + _ca_crosssell
    # Realiste : 40% des perdus reviennent, 50% du delta rattrape
    _recup_realiste = _ca_reconquete * 0.40 + _ca_rattrapage * 0.50 + _ca_crosssell
    # Optimiste : 60% des perdus reviennent, 70% du delta rattrape, cross-sell x2
    _recup_opti = _ca_reconquete * 0.60 + _ca_rattrapage * 0.70 + _ca_crosssell * 2

    sc1, sc2, sc3 = st.columns(3)
    sc1.markdown(f"""<div style="border:2px solid {C['warning']};border-radius:12px;padding:20px;text-align:center">
        <div style="font-size:11px;color:#9CA3AF;font-weight:600;text-transform:uppercase;letter-spacing:0.06em">Pessimiste</div>
        <div style="font-size:28px;font-weight:700;color:{C['warning']};margin:8px 0">+{_recup_pessi:,.0f} EUR</div>
        <div style="font-size:12px;color:#6B7280">20% des perdus reviennent<br>30% de la baisse rattrape<br>+1 rayon / pharmacie</div>
    </div>""", unsafe_allow_html=True)
    sc2.markdown(f"""<div style="border:2px solid {C['primary']};border-radius:12px;padding:20px;text-align:center">
        <div style="font-size:11px;color:#9CA3AF;font-weight:600;text-transform:uppercase;letter-spacing:0.06em">Realiste</div>
        <div style="font-size:28px;font-weight:700;color:{C['primary']};margin:8px 0">+{_recup_realiste:,.0f} EUR</div>
        <div style="font-size:12px;color:#6B7280">40% des perdus reviennent<br>50% de la baisse rattrape<br>+1 rayon / pharmacie</div>
    </div>""", unsafe_allow_html=True)
    sc3.markdown(f"""<div style="border:2px solid {C['positive']};border-radius:12px;padding:20px;text-align:center">
        <div style="font-size:11px;color:#9CA3AF;font-weight:600;text-transform:uppercase;letter-spacing:0.06em">Optimiste</div>
        <div style="font-size:28px;font-weight:700;color:{C['positive']};margin:8px 0">+{_recup_opti:,.0f} EUR</div>
        <div style="font-size:12px;color:#6B7280">60% des perdus reviennent<br>70% de la baisse rattrape<br>+2 rayons / pharmacie</div>
    </div>""", unsafe_allow_html=True)

    with st.expander("Comment c'est calcule ?"):
        st.markdown(f"""
| Levier | Pertes brutes | Pessimiste | Realiste | Optimiste |
|--------|:------------:|:----------:|:--------:|:---------:|
| Reconquete clients perdus | {_ca_reconquete:,.0f} EUR | 20% = {_ca_reconquete*0.20:,.0f} | 40% = {_ca_reconquete*0.40:,.0f} | 60% = {_ca_reconquete*0.60:,.0f} |
| Rattrapage clients en baisse | {_ca_rattrapage:,.0f} EUR | 30% = {_ca_rattrapage*0.30:,.0f} | 50% = {_ca_rattrapage*0.50:,.0f} | 70% = {_ca_rattrapage*0.70:,.0f} |
| Ventes croisees ({len(_pot_pharma)} pharmacies) | -| +1 rayon = {_ca_crosssell:,.0f} | +1 rayon = {_ca_crosssell:,.0f} | +2 rayons = {_ca_crosssell*2:,.0f} |
| **Total** | | **{_recup_pessi:,.0f}** | **{_recup_realiste:,.0f}** | **{_recup_opti:,.0f}** |

*Hypotheses : les % de reconquete sont bases sur des taux de retour client classiques en B2B. Le cross-sell est calcule a {_ca_par_rayon:,.0f} EUR par rayon supplementaire (moyenne des clients 3+ rayons).*
        """)

    insight(f"En scenario realiste, vous pouvez viser **+{_recup_realiste:,.0f} EUR** de CA supplementaire en combinant reconquete telephonique des perdus, rattrapage des clients en baisse, et ventes croisees. Ca ne comblera pas 100% des {_pertes_brutes:,.0f} EUR perdus, mais c'est un objectif atteignable.", "🎯")

    st.markdown("---")

    # --- PROFILS CLIENTS ---
    st.subheader("Profils clients")
    st.markdown("*Regroupement automatique des clients en fonction de leur comportement d'achat (CA, frequence, nombre de familles).*")

    with st.expander("Comment lire cette analyse ?"):
        st.markdown("""
**Chaque client est classe dans un profil** en fonction de 4 criteres :
- Son **CA total** en 2025
- Son **nombre de commandes** (frequence)
- Le **nombre de familles** de produits achetees (diversite)
- Son **panier moyen** par commande

**Les profils :**
- **Champions** : fort CA, achats frequents -vos meilleurs clients, a choyer
- **Reguliers** : CA moyen, achats reguliers -potentiel pour monter en gamme
- **Occasionnels** : achats peu frequents -a developper en augmentant la frequence
- **Petits acheteurs** : faible CA -peuvent etre developpes en elargissant les familles achetees

**Le graphique :**
- Chaque point = un client
- Plus a droite = plus de commandes
- Plus haut = plus de CA
- Taille du point = panier moyen
        """)

    # Features pour le clustering
    client_features = ventes_f[ventes_f["Annee"] == 2025].groupby("Client").agg(
        Nb_Commandes=("Facture", "nunique"),
        Nb_Familles=("Famille", "nunique"),
        Nb_Articles=("Code_Article", "nunique"),
        CA_Total=("Total_HT", "sum"),
        Qte_Total=("Qte", "sum"),
    ).reset_index()

    # Panier moyen proprement
    panier = ventes_f[ventes_f["Annee"] == 2025].groupby(["Client", "Facture"])["Total_HT"].sum().reset_index()
    panier_moyen = panier.groupby("Client")["Total_HT"].mean().reset_index()
    panier_moyen.columns = ["Client", "Panier_Moyen"]

    client_features = client_features.merge(panier_moyen, on="Client", how="left")
    client_features = client_features[client_features["CA_Total"] > 0]

    if len(client_features) >= 10:
        # Clustering fixe a 4 profils
        features_cols = ["Nb_Commandes", "Nb_Familles", "CA_Total", "Panier_Moyen"]
        X = client_features[features_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_clusters = 4
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        client_features["Cluster"] = kmeans.fit_predict(X_scaled)

        # Nommer les clusters par CA moyen
        cluster_order = client_features.groupby("Cluster")["CA_Total"].mean().sort_values(ascending=False).index
        name_map = {}
        cluster_names = ["Champions", "Reguliers", "Occasionnels", "Petits acheteurs"]
        for i, idx in enumerate(cluster_order):
            name_map[idx] = cluster_names[i] if i < len(cluster_names) else f"Groupe {idx}"
        client_features["Profil"] = client_features["Cluster"].map(name_map)

        # Viz
        fig_cluster = px.scatter(
            client_features, x="Nb_Commandes", y="CA_Total",
            color="Profil", hover_name="Client",
            size="Panier_Moyen", size_max=25,
            labels={"Nb_Commandes": "Nombre de commandes", "CA_Total": "CA Total (EUR)"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_cluster.update_layout(height=500, margin=dict(t=20, b=20))
        st.plotly_chart(fig_cluster, use_container_width=True)

        # Stats par cluster
        st.subheader("Detail des profils")
        cluster_stats = client_features.groupby("Profil").agg(
            Nb_Clients=("Client", "count"),
            CA_Moyen=("CA_Total", "mean"),
            CA_Total=("CA_Total", "sum"),
            Commandes_Moy=("Nb_Commandes", "mean"),
            Familles_Moy=("Nb_Familles", "mean"),
            Panier_Moy=("Panier_Moyen", "mean")
        ).reset_index().sort_values("CA_Total", ascending=False)

        # Recommandations par profil
        reco_map = {
            "Champions": "A choyer : fidelisation, avantages, contact regulier",
            "Reguliers": "Potentiel : elargir les familles achetees, monter en gamme",
            "Occasionnels": "A developper : augmenter la frequence d'achat",
            "Petits acheteurs": "A activer : proposer de nouvelles familles de produits"
        }
        cluster_stats["Action recommandee"] = cluster_stats["Profil"].map(reco_map).fillna("")

        for col in ["CA_Moyen", "CA_Total", "Panier_Moy"]:
            cluster_stats[col] = cluster_stats[col].apply(lambda x: f"{x:,.0f}")
        for col in ["Commandes_Moy", "Familles_Moy"]:
            cluster_stats[col] = cluster_stats[col].apply(lambda x: f"{x:.1f}")

        st.dataframe(cluster_stats, use_container_width=True, hide_index=True)

        # Insight clustering
        champions = client_features[client_features["Profil"] == "Champions"]
        if len(champions) > 0:
            pct_ca_champ = champions["CA_Total"].sum() / client_features["CA_Total"].sum() * 100
            pct_nb_champ = len(champions) / len(client_features) * 100
            nb_occ = len(client_features[client_features["Profil"] == "Occasionnels"])
            nb_petits = len(client_features[client_features["Profil"] == "Petits acheteurs"])
            insight(f"Vos {len(champions)} Champions ({pct_nb_champ:.0f}% des clients) generent {pct_ca_champ:.0f}% du CA. Ce sont vos comptes strategiques. A l'inverse, {nb_occ + nb_petits} clients (Occasionnels + Petits acheteurs) representent un vivier de developpement concret.")

        csv_download(
            client_features[["Client", "Profil", "CA_Total", "Nb_Commandes", "Nb_Familles", "Panier_Moyen"]].sort_values("CA_Total", ascending=False),
            "profils_clients.csv"
        )

    # --- CROSS-SELL ---
    st.markdown("---")
    st.subheader("Opportunites de ventes croisees")
    st.markdown("*Quels rayons vos pharmacies n'achetent pas encore ? Voila votre potentiel.*")

    # Matrice client x rayon
    pharma_ventes = ventes_f[(ventes_f["Annee"] == 2025) & (ventes_f["Client"].str.contains("PHARMACIE", case=False, na=False))]
    cross = pharma_ventes.groupby(["Client", "Rayon"])["Total_HT"].sum().unstack(fill_value=0)
    cross = cross.drop(columns=["Autre"], errors="ignore")
    cross_bool = (cross > 0).astype(int)

    # Taux d'adoption par rayon
    adoption = cross_bool.mean().sort_values(ascending=False)

    fig_adopt = px.bar(
        x=adoption.values * 100, y=adoption.index,
        orientation="h", color_discrete_sequence=[C["primary"]],
        labels={"x": "% de pharmacies qui achetent ce rayon", "y": ""}
    )
    fig_adopt.update_layout(height=500, margin=dict(t=20, b=20, l=200))
    st.plotly_chart(fig_adopt, use_container_width=True)

    top_adopt = adoption.index[0] if len(adoption) else ""
    top_adopt_pct = adoption.values[0] * 100 if len(adoption) else 0
    low_adopt = adoption.index[-1] if len(adoption) else ""
    low_adopt_pct = adoption.values[-1] * 100 if len(adoption) else 0
    insight(f"Le rayon le plus repandu est \"{top_adopt}\" ({top_adopt_pct:.0f}% des pharmacies). Le moins repandu est \"{low_adopt}\" ({low_adopt_pct:.0f}%). Les rayons entre 30% et 60% d'adoption sont vos meilleures cibles de vente croisee : assez de pharmacies les achetent pour prouver la demande, mais il reste beaucoup de potentiel.")

    # Opportunites concretes avec rayons manquants
    st.subheader("Pharmacies avec potentiel d'elargissement")
    st.markdown("*Pharmacies qui n'achetent qu'1 ou 2 rayons, avec le detail de ce qui leur manque.*")

    nb_rayons = cross_bool.sum(axis=1).reset_index()
    nb_rayons.columns = ["Client", "Nb_Rayons"]
    ca_client = pharma_ventes.groupby("Client")["Total_HT"].sum().reset_index()
    ca_client.columns = ["Client", "CA_2025"]

    # Rayons manquants par client
    all_rayons_cross = sorted(cross_bool.columns.tolist())
    rayons_manquants_list = []
    for client_name in cross_bool.index:
        row = cross_bool.loc[client_name]
        manquants = [r for r in all_rayons_cross if row[r] == 0]
        rayons_manquants_list.append({"Client": client_name, "Rayons_Manquants": ", ".join(manquants) if manquants else "Tous couverts"})
    df_manquants = pd.DataFrame(rayons_manquants_list)

    pot = nb_rayons.merge(ca_client, on="Client").merge(df_manquants, on="Client")
    pot = pot.sort_values("Nb_Rayons")
    moy_rayons = pot["Nb_Rayons"].mean()

    # CA moyen par rayon (benchmark)
    clients_matures = pot[pot["Nb_Rayons"] >= 3]
    ca_par_rayon_moyen = clients_matures["CA_2025"].sum() / clients_matures["Nb_Rayons"].sum() if clients_matures["Nb_Rayons"].sum() > 0 else 0

    pot_elarg = pot[pot["Nb_Rayons"] <= 2].sort_values("CA_2025", ascending=False).copy()

    # Calcul opportunite
    pot_elarg["Rayons_Manquants_Pessi"] = 1
    rayons_manquants_opti = (moy_rayons - pot_elarg["Nb_Rayons"]).clip(lower=1).round(0).astype(int)
    pot_elarg["Rayons_Manquants_Opti"] = rayons_manquants_opti
    pot_elarg["Rayons_Manquants_Mid"] = ((1 + rayons_manquants_opti) / 2).round(0).astype(int).clip(lower=1)
    pot_elarg["Opportunite_Pessimiste"] = pot_elarg["Rayons_Manquants_Pessi"] * ca_par_rayon_moyen
    pot_elarg["Opportunite_Mid"] = pot_elarg["Rayons_Manquants_Mid"] * ca_par_rayon_moyen
    pot_elarg["Opportunite_Optimiste"] = pot_elarg["Rayons_Manquants_Opti"] * ca_par_rayon_moyen

    total_pessi = pot_elarg["Opportunite_Pessimiste"].sum()
    total_mid = pot_elarg["Opportunite_Mid"].sum()
    total_opti = pot_elarg["Opportunite_Optimiste"].sum()

    st.metric("Moyenne de rayons par pharmacie", f"{moy_rayons:.1f}")
    st.caption(f"CA moyen par rayon (reference clients 3+ rayons) : **{ca_par_rayon_moyen:,.0f} EUR**")

    st.markdown("---")

    # KPIs opportunite
    st.markdown("### Potentiel de ventes croisees estime")

    rayons_mid = ((pot_elarg["Nb_Rayons"].mean() + moy_rayons) / 2)
    col_p, col_m, col_o = st.columns(3)
    with col_p:
        st.markdown(f"""<div style="border:1px solid rgba(0,0,0,0.1);border-radius:12px;padding:20px;text-align:center">
            <div style="font-size:11px;color:#9CA3AF;font-weight:600;text-transform:uppercase;letter-spacing:0.06em">Pessimiste</div>
            <div style="font-size:28px;font-weight:700;color:#0A0A0A;margin:8px 0">{total_pessi:,.0f} EUR</div>
            <div style="font-size:13px;color:#6B7280">+1 rayon par pharmacie</div>
        </div>""", unsafe_allow_html=True)
    with col_m:
        st.markdown(f"""<div style="border:1px solid rgba(0,0,0,0.1);border-radius:12px;padding:20px;text-align:center">
            <div style="font-size:11px;color:#9CA3AF;font-weight:600;text-transform:uppercase;letter-spacing:0.06em">Realiste</div>
            <div style="font-size:28px;font-weight:700;color:#0A0A0A;margin:8px 0">{total_mid:,.0f} EUR</div>
            <div style="font-size:13px;color:#6B7280">Passage a ~{rayons_mid:.0f} rayons par pharmacie</div>
        </div>""", unsafe_allow_html=True)
    with col_o:
        st.markdown(f"""<div style="border:1px solid rgba(0,0,0,0.1);border-radius:12px;padding:20px;text-align:center">
            <div style="font-size:11px;color:#9CA3AF;font-weight:600;text-transform:uppercase;letter-spacing:0.06em">Optimiste</div>
            <div style="font-size:28px;font-weight:700;color:#0A0A0A;margin:8px 0">{total_opti:,.0f} EUR</div>
            <div style="font-size:13px;color:#6B7280">Passage a {moy_rayons:.0f} rayons (la moyenne)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    with st.expander("Comment est calcule le potentiel ?"):
        st.markdown(f"""
**Methode :**

1. On identifie les pharmacies qui n'achetent que **1 ou 2 rayons** ({len(pot_elarg)} pharmacies)
2. On regarde le **CA moyen par rayon** chez les pharmacies qui en achetent 3 ou plus = **{ca_par_rayon_moyen:,.0f} EUR/rayon**
3. On estime le CA supplementaire si chaque pharmacie ajoute des rayons

| Scenario | Hypothese |
|----------|-----------|
| **Pessimiste** | Chaque pharmacie ajoute **1 seul rayon** |
| **Realiste** | Chaque pharmacie passe a **~{rayons_mid:.0f} rayons** |
| **Optimiste** | Chaque pharmacie monte a la **moyenne** ({moy_rayons:.1f} rayons) |
        """)

    st.markdown("")
    st.markdown(f"**{len(pot_elarg)} pharmacies** ciblees -avec le detail des rayons manquants :")

    df_pot = pot_elarg[["Client", "Nb_Rayons", "CA_2025", "Rayons_Manquants", "Opportunite_Pessimiste", "Opportunite_Mid", "Opportunite_Optimiste"]].reset_index(drop=True)
    df_pot.columns = ["Client", "Rayons actuels", "CA 2025", "Rayons manquants (a proposer)", "Pessimiste", "Realiste", "Optimiste"]
    st.dataframe(df_pot.head(30), use_container_width=True, hide_index=True)
    csv_download(df_pot, "opportunites_cross_sell.csv", "📥 Telecharger la fiche d'appel (CSV)")

    insight(f"L'opportunite de ventes croisees pour {len(pot_elarg)} pharmacies : {total_pessi:,.0f} EUR (pessimiste), {total_mid:,.0f} EUR (realiste) ou {total_opti:,.0f} EUR (optimiste). La colonne 'Rayons manquants' vous donne directement la fiche d'appel pour chaque pharmacie.", "🎯")
