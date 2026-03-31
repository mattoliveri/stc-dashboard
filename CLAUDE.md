# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Dashboard Streamlit d'analyse du portefeuille clients pour **STC Services Pharma** (equipement medical, Marseille). Deux entites : STC (185 clients) et EMS (83 clients, rachete en 2023). Donnees 2024-2025.

## Commands

```bash
# Lancer le dashboard
streamlit run app.py

# Lancer sur un port specifique
streamlit run app.py --server.port 8501
```

Dependencies : `streamlit`, `pandas`, `plotly`, `numpy`, `scikit-learn`. Pas de requirements.txt — installer manuellement si necessaire.

## Architecture

**`app.py`** — Application Streamlit monofichier (~1300 lignes), 5 onglets :
1. Vue Globale (KPIs, CA, evolution mensuelle)
2. Sante Portefeuille (segmentation clients, score de risque, reconquete)
3. Impact Commerciaux (avant/apres suppression commerciaux juin 2024)
4. Produits & Familles (CA par rayon, effet volume/prix)
5. Opportunites (clustering K-Means, cross-sell)

Palette de couleurs centralisee dans le dict `C` en haut du fichier. Segments clients dans `COLORS_SEG`. Fonction `insight()` pour les encadres d'interpretation.

## Donnees

```
DATASETS/
  clients/
    clients_STC.csv     185 clients, cols: Client payeur, CP, Ville, ..., CA Loc/Vte 2024/2025
    clients_EMS.csv      83 clients, meme structure mais colonnes decalees (agent commercial en plus)
  ventes/
    Extraction_Ventes_2024.csv   STC uniquement, detail factures (10K lignes)
    Extraction_Ventes_2025.csv   STC + EMS, detail factures (12K lignes)
    Comparaison.csv              Recap article par article 2024 vs 2025
    familles.csv                 86 familles produit, 16 rayons
```

**Asymetrie critique** : les ventes 2024 ne contiennent PAS EMS, les ventes 2025 l'incluent. Toute comparaison directe 2024/2025 sur les fichiers ventes doit mentionner cette limite.

Les colonnes STC et EMS ne sont pas au meme index — le chargement dans `load_data()` utilise `columns[i]` par position.

## Contexte metier

- Client = pharmacies de Marseille et alentours
- Activite = vente + location d'equipement medical (lits, fauteuils, cannes, etc.)
- Commerciaux terrain supprimes en juin 2024 (relations par telephone uniquement depuis)
- Ne pas analyser marges/prix de revient (donnees non fiables selon le client)
- Objectif : identifier clients a risque, quantifier opportunites de reconquete, fidelisation

Voir `notes/contexte_client.md` pour le brief complet.

## Segmentation clients

Criteres dans la fonction `segmenter()` de l'onglet 2 :
- **Perdu** : CA 2024 > 0 et CA 2025 = 0
- **A risque** : baisse > 30% ET > 500 EUR
- **En baisse** : baisse > 5% ET > 200 EUR
- **En croissance** : hausse > 15% ET > 200 EUR
- **Stable** : entre -5% et +15%
- **Nouveau** : CA 2024 = 0 et CA 2025 > 0

Score de risque 0-100 : combine evolution CA, recence derniere commande, frequence 2025.
