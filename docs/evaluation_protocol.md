# Protocole d'évaluation

## Jeux de cas

- `smoke` : ~20 images RSNA pour vérifier la chaîne de bout en bout.
- `dev` : 100 à 300 cas (voir échantillons d'évaluation dans les notebooks 01/04/05).
- `final` : 20 à 30 cas commentés pour la soutenance (registre d'erreurs, `eval/error_register_template.csv`).

Les métriques par notebook (accuracy, macro-F1, mAP, IoU, accord de localisation par quadrant) sont calculées et affichées directement dans chaque notebook — voir en particulier les sections d'évaluation de `notebooks/02_yolo_training.ipynb` (mAP@50, mAP@50-95), `notebooks/04_maira2_alternative_pipeline.ipynb` (classification + accord de localisation) et `notebooks/05_gemma4_e2b_qlora_classification_bonus.ipynb` (accuracy, accord de quadrant).

## Métriques minimales

- Accuracy.
- Macro-F1.
- Sensibilité sur les cas `suspected_opacity`.
- Spécificité sur les cas `normal`.
- Taux de JSON valide.
- Taux de warning présent.
- Taux d'incertitude.
- Hallucinations textuelles détectées manuellement.
- Latence médiane.

## Taxonomie d'erreurs

| Code | Signification | Exemple |
|---|---|---|
| FN | Faux négatif | anomalie présente prédite normale |
| FP | Faux positif | image normale prédite suspecte |
| UA | Incertitude acceptable | signes faibles ou image limitée |
| JF | JSON format error | sortie non exploitable |
| HT | Hallucination textuelle | mention d'un signe non visible |

## Règle de soutenance

Ne jamais montrer seulement des réussites. Une bonne défense montre aussi les faux positifs, les faux négatifs, les incertitudes et les limites de qualité image.

## Smoke test attendu

Avant toute démonstration, le dépôt doit passer un contrôle court :

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m compileall -q app tests
```

Ce test ne remplace pas l'analyse d'erreurs. Il vérifie seulement que le dépôt est structuré correctement (notebooks valides, prompts présents, avertissement non clinique présent) et que `app/app.py` compile.
