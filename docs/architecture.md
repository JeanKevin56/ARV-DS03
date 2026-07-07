# Architecture livrée

## Mode 1 — analyse en direct : YOLO26 (décideur) + MedGemma (rédacteur)

```text
Radio DICOM
   │  prétraitement : Min-Max 8 bits → CLAHE (clip=2.0, grid=8x8) → RGB, 640×640
   ▼
YOLO26 (fine-tuné, notebooks/02_yolo_training.ipynb)
   │  détecte les boîtes d'opacité, shrink 0.85, NMS iou=0.4
   │  bande d'incertitude sur la confiance des boîtes :
   │    confiance ≥ seuil haut  → suspected_opacity
   │    seuil bas ≤ confiance < seuil haut → uncertain
   │    aucune détection ≥ seuil bas → normal
   ▼
MedGemma 1.5 4B (via Ollama, prompts/medgemma_report_writer_prompt.txt)
   │  reçoit l'image + le verdict + les zones anatomiques (PAS les coordonnées brutes)
   │  rédige une description clinique cohérente avec le verdict (jamais de méta-langage
   │  du type "the detection model")
   ▼
JSON structuré + journal_analyses.csv (app/app.py)
```

Composant clé : `app/app.py` — `build_medgemma_prompt()`, `predict_boxes()`, `dicom_to_rgb()`, `log_analysis()`.

## Mode 2 — banque pré-analysée : MAIRA-2 + GPT-OSS

Calculé à l'avance sur Kaggle (`notebooks/04_maira2_alternative_pipeline.ipynb`), lu tel quel par l'app :

```text
Radio DICOM (RSNA, déjà en banque)
   ▼
MAIRA-2 (encodeur RAD-DINO gelé + Vicuna-7B) — par prompting, aucun fine-tuning
   │  génère un rapport ancré : liste de (phrase, boîtes englobantes)
   ▼
Classification dérivée par règles (mots-clés radiologiques)
   │  puis affinée par GPT-OSS-120B (OpenRouter, prompts/gptoss_maira2_classifier_prompt.txt)
   │  qui lit le rapport MAIRA-2 et tranche normal / suspected_opacity / uncertain
   ▼
data/maira2_interface.csv (classe finale, boîtes prédites/réelles, rapport texte)
   ▼
app/app.py (mode 2) : lookup par patientId, affichage boîtes rouge=MAIRA-2 / vert=vérité RSNA
```

## Piste de référence (baseline) — classification directe par prompting

`notebooks/01_baseline_medgemma_prompting.ipynb` : un VLM (MedGemma/PaliGemma) reçoit directement l'image et un prompt structuré, sans détecteur intermédiaire, et renvoie classe + `box_2d` + justification en un seul appel. C'est la baseline "Must have" contre laquelle le pipeline détecteur+rédacteur (mode 1) et la piste MAIRA-2 (mode 2) sont comparés.

## Sortie JSON commune

Voir `prompts/json_schema.md` pour le schéma détaillé et les règles de validation.

## Pistes bonus (Could have)

- `notebooks/05_gemma4_e2b_qlora_classification_bonus.ipynb` : fine-tuning QLoRA de Gemma 4 E2B pour la classification 3 classes, plus une localisation qualitative par quadrant.
- `notebooks/06_gemma4_e4b_lora_finetuning_bonus.ipynb` : fine-tuning LoRA de Gemma 4 E4B pour produire directement un JSON avec `box_2d`, à partir des descriptions générées par MedGemma.

## Objectifs d'intégration

- 100 % des sorties avec `warning` non clinique.
- 100 % des analyses journalisées (`app/journal_analyses.csv`, non versionné).
- Classe `uncertain` déclenchée par la bande d'incertitude YOLO ou par un JSON MedGemma non parsable.
