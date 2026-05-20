# csma-tp — TP CSMA : méthode Arlequin pour poutres hétérogènes

[![CI](https://github.com/basileMarchand/csma_junior_2026/actions/workflows/ci.yml/badge.svg)](https://github.com/basileMarchand/csma_junior_2026/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-%E2%89%A53.11-blue)
![License](https://img.shields.io/badge/license-GPL--3.0-blue)

Projet pédagogique illustrant la méthode **Arlequin** pour le couplage d'un
modèle homogénéisé et d'un modèle hétérogène (VER) sur une poutre 1D en
traction. Le code repose sur une petite bibliothèque éléments finis 1D
maison (`pybeam`).

## Structure

```
pybeam/         Bibliothèque EF 1D (maillage, éléments, opérateurs,
                conditions de Dirichlet, couplage Arlequin)
tests/          Tests pytest
  cases/        Cas de référence (.py + .toml + .ref.csv)
workflows/      Pilotage des études paramétriques
scripts/        Agrégation et tracés (utilisés par Snakemake)
Snakefile       Pipeline d'étude de convergence vs taille de VER
results/        Sorties (CSV, figures)
```

## Prérequis

- Python ≥ 3.11
- [`uv`](https://docs.astral.sh/uv/) (gestion de l'environnement)

## Installation

```bash
uv sync
```

## Utilisation

- Exécuter tous les cas de `tests/cases/` (écrit `<name>.csv` à côté de chaque `<name>.py`) :

  ```bash
  for f in tests/cases/*.py; do uv run python "$f"; done
  ```

- Lancer la suite pytest :

  ```bash
  uv run pytest tests/ -v
  ```

- Générer les figures de comparaison (PNG dans `results/figures/`) :

  ```bash
  for f in tests/cases/*.py; do CSMA_PLOT=1 uv run python "$f"; done
  ```

- Régénérer les oracles `.ref.csv` (à faire une seule fois après un changement physique délibéré, puis commiter) :

  ```bash
  for f in tests/cases/*.py; do
      uv run python "$f"
      stem="${f%.py}"
      mv "${stem}.csv" "${stem}.ref.csv"
  done
  ```

- Nettoyer les sorties et caches :

  ```bash
  rm -rf results/parametric/ .snakemake/ tests/cases/*.csv
  find . -type d -name __pycache__ -exec rm -rf {} +
  ```

## Étude de convergence

Le `Snakefile` enchaîne, pour plusieurs tailles de VER, le calcul Arlequin,
l'agrégation des erreurs L2 et le tracé en log-log :

```bash
uv run snakemake --cores all
```

La figure finale est produite dans `results/parametric/convergence_plot.png`.

## Licence

Distribué sous licence **GNU GPL v3.0**. Voir le fichier [`LICENSE`](LICENSE).
