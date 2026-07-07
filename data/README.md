# Données

## Dataset d'entraînement / évaluation

Le projet utilise le **RSNA Pneumonia Detection Challenge** (Kaggle), radiographies thoraciques frontales au format DICOM avec annotations de boîtes englobantes. Dataset externe, sous licence propre : il n'est **pas redistribué** dans ce dépôt (voir `.gitignore` : `stage_2_*`). Pour reproduire les notebooks, ajouter le dataset directement sur Kaggle (Add Input → *RSNA Pneumonia Detection Challenge*) ou le télécharger depuis la compétition.

## `maira2_interface.csv`

Banque d'images pré-analysées par la piste MAIRA-2 + GPT-OSS (générée par `notebooks/04_maira2_alternative_pipeline.ipynb`), consommée directement par le Mode 2 de `app/app.py`. Colonnes principales :

- `patientId` : identifiant de l'image RSNA.
- `classe_finale`, `classe_regles`, `classe_LLM` : classe retenue, classe par règles (mots-clés), classe par GPT-OSS.
- `boites_predites`, `boites_reelles` : boîtes MAIRA-2 et vérité terrain RSNA, en JSON, pixels sur une image 1024×1024.
- `ref_size` : taille de référence des boîtes.
- `rapport_maira` : rapport radiologique brut généré par MAIRA-2.

## Poids YOLO (`best.pt`)

Non versionné (trop volumineux, voir `.gitignore`). Produit par `notebooks/02_yolo_training.ipynb` sur Kaggle, à télécharger et pointer depuis la barre latérale de `app/app.py`.

## Modèle MedGemma (Ollama)

Le Mode 1 de l'app appelle MedGemma via un serveur [Ollama](https://ollama.com) local (`ollama pull <modele>`, voir `README.md`). Aucun poids n'est stocké dans ce dépôt.
