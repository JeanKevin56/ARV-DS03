# ARV-DS03 — Assistant radiologue virtuel responsable

> **École :** EFREI — Solution Delivery, filière Data · **Année académique :** 2025-2026
> Mastercamp — projet ARV-DS03

## Contexte

Prototype pédagogique d'IA médicale multimodale : détection de pneumonie sur radiographies thoraciques frontales (dataset RSNA Pneumonia Detection Challenge), avec deux pistes explorées et comparées.

---

> **Position non clinique.** Ce dépôt n'est pas un dispositif médical. Il ne doit jamais être utilisé pour diagnostiquer, trier ou orienter un patient. Toute sortie doit rester un résultat expérimental, vérifié par un professionnel qualifié.

---

## Contrat du projet

| Élément | Cadrage |
|---|---|
| Entrée | Une radiographie thoracique frontale (DICOM) |
| Sorties | `normal`, `suspected_opacity`, `uncertain` |
| Preuve minimale | JSON valide, warning, logs, métriques, cas d'erreur |
| Données | RSNA Pneumonia Detection Challenge (externe, licence propre, non redistribué) |
| Finalité | Prototype éducatif de data/IA, pas aide au diagnostic réelle |

## Architecture livrée

Deux pistes ont été développées et comparées (voir `notebooks/`) ; l'application (`app/app.py`) expose les deux :

```text
Mode 1 — analyse en direct (cascade YOLO26 + MedGemma)
  Radio DICOM → prétraitement (CLAHE) → YOLO26 détecte et DÉCIDE la classe
  (bande d'incertitude sur la confiance) → MedGemma (via Ollama) REDIGE une
  description clinique cohérente avec ce verdict → JSON + logs CSV

Mode 2 — banque pré-analysée (MAIRA-2 + GPT-OSS)
  Radiographies déjà traitées sur Kaggle par MAIRA-2 (rapport ancré + boîtes),
  classe dérivée par règles puis affinée par GPT-OSS-120B → data/maira2_interface.csv
  → lu directement par l'app (pas de calcul en direct dans ce mode)
```

Le principe directeur du mode 1 : **YOLO décide, MedGemma décrit** — MedGemma ne recalcule jamais la classe, il justifie le verdict de manière médicalement crédible (voir `prompts/medgemma_report_writer_prompt.txt`).

La progression du projet (baseline par prompting direct → pipeline détecteur+rédacteur → piste MAIRA-2 → expérimentations LoRA) est documentée dans `notebooks/`, numérotés dans l'ordre de lecture.

## Démarrage rapide

Prérequis :
- [Ollama](https://ollama.com) installé et lancé, avec le modèle MedGemma tiré (`ollama pull <modele>`) — voir `ollama list` pour le nom exact.
- Les poids YOLO entraînés `best.pt` (produits par `notebooks/02_yolo_training.ipynb`, non versionnés car trop volumineux).
- `data/maira2_interface.csv` (déjà fourni dans ce dépôt) pour le mode 2.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

Dans la barre latérale de l'app : renseigner le chemin vers `best.pt` et le nom exact du modèle Ollama. Le mode MAIRA-2 fonctionne dès le lancement (CSV déjà fourni) ; le mode YOLO+MedGemma nécessite les deux ressources ci-dessus.

## Smoke test du dépôt

Avant une soutenance, un push ou une livraison, lancer le contrôle court :

```bash
pip install -r requirements-test.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m compileall -q app tests
```

Ce smoke test vérifie la structure du dépôt, la présence des prompts et notebooks, la validité JSON des notebooks, la présence de l'avertissement non clinique et la compilation Python.

## Organisation

```text
ARV-DS03-main/
├── README.md
├── docs/          # appel d'offre, architecture, éthique, protocole d'évaluation
├── data/          # maira2_interface.csv (banque mode 2) + doc du dataset externe
├── prompts/       # prompts réellement utilisés (MedGemma, GPT-OSS) + schéma JSON
├── app/           # application Streamlit (app.py), les deux modes
├── notebooks/     # baseline, entraînement YOLO, pipelines, piste MAIRA-2, bonus LoRA
├── eval/          # registre d'erreurs (template à remplir)
└── tests/         # smoke tests du dépôt
```

## Correspondance avec le barème (Must / Should / Could)

Le détail complet du barème est dans `docs/appel_offre.md`. Ce que ce dépôt livre concrètement :

| Niveau | Attendu | Livré dans ce dépôt |
|---|---|---|
| **Must have** | Baseline par prompting, JSON structuré, 3 classes, warning, interface web, logs | `notebooks/01_baseline_medgemma_prompting.ipynb` (classification directe par MedGemma) + `app/app.py` (interface Streamlit, journalisation CSV) |
| **Should have** | Comparaison baseline vs amélioration, métriques, registre d'erreurs | `notebooks/02_yolo_training.ipynb` + `03_yolo_medgemma_pipeline.ipynb` (détecteur + rédacteur, amélioration mesurée vs baseline) ; `notebooks/04_maira2_alternative_pipeline.ipynb` (piste alternative comparée) ; `eval/error_register_template.csv` |
| **Could have** | LoRA/QLoRA, localisation visuelle, classifieur auxiliaire | `notebooks/05_gemma4_e2b_qlora_classification_bonus.ipynb` (QLoRA + localisation par quadrant) ; `notebooks/06_gemma4_e4b_lora_finetuning_bonus.ipynb` (fine-tuning LoRA E4B) |

La classe `uncertain` est un garde-fou méthodologique volontaire, pas un échec du modèle : savoir ne pas conclure sur une image ambiguë fait partie de la qualité attendue.

## Références techniques

Les pistes avancées restent expérimentales, traçables et justifiées. En particulier, un groupe qui mobilise Gemma, MedGemma, MAIRA-2 ou RSNA doit citer la source exacte, la version, les conditions d'accès et les limites d'usage.

| Ressource | Usage dans ce projet | Référence à citer |
|---|---|---|
| MedGemma 1.5 4B | Description clinique (mode 1), classification directe (baseline) | [Model card Hugging Face](https://huggingface.co/google/medgemma-4b-pt) |
| Gemma 4 (E2B/E4B) + Unsloth | Fine-tuning LoRA/QLoRA expérimental (bonus) | [Guide Gemma 4](https://unsloth.ai/docs/models/gemma-4/train), [blog Unsloth](https://unsloth.ai/blog) |
| MAIRA-2 (Microsoft) | Génération de rapport ancré par prompting (piste alternative) | [Model card Hugging Face](https://huggingface.co/microsoft/maira-2) |
| Ultralytics YOLO26 | Détection des zones d'opacité | [Documentation Ultralytics](https://docs.ultralytics.com) |
| RSNA Pneumonia Detection Challenge | Dataset d'entraînement/évaluation, accès Kaggle | [Kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) |

## Points de vigilance

- Ne pas inventer d'information clinique absente de l'image.
- Ne pas supprimer la classe `uncertain` ; elle est un garde-fou, pas un échec.
- Ne pas afficher uniquement des réussites en soutenance.
- Ne jamais commiter de données patient réelles, identifiantes ou ambiguës.
- Ne pas présenter le prototype comme validé médicalement.

## Licence et sources externes

Le code pédagogique du dépôt est publié sous licence MIT.

**Les datasets externes, modèles et bibliothèques utilisés conservent leurs licences propres** : vérifier et documenter les droits d'usage avant toute expérimentation (RSNA Pneumonia, MedGemma, MAIRA-2 et Gemma 4 sont notamment soumis à des conditions d'accès spécifiques sur Hugging Face/Kaggle).
