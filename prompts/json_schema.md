# Sortie JSON — schéma réel

Champs produits par MedGemma (mode 1, `app/app.py`) et par le pipeline MAIRA-2 (mode 2) :

```json
{
  "image_quality": "good | limited | poor",
  "predicted_class": "normal | suspected_opacity | uncertain",
  "confidence": 0.0,
  "visual_evidence": ["string"],
  "justification": "string",
  "limitations": ["string"],
  "warning": "Prototype pedagogique. Non destine au diagnostic medical.",
  "opacities": [{"box_2d": [1, 1, 1, 1]}]
}
```

## Règles de validation

- `predicted_class` doit être l'une de : `normal`, `suspected_opacity`, `uncertain`.
- `confidence` est un nombre entre 0 et 1.
- `warning` doit toujours être présent.
- `opacities[].box_2d` : `[y1, x1, y2, x2]`, entiers normalisés entre 0 et 1024 (convention héritée de PaliGemma/MedGemma).
- Dans `app/app.py`, `predicted_class` n'est pas décidé par MedGemma : YOLO décide la classe en amont (bande d'incertitude sur la confiance des boîtes), MedGemma ne fait que rédiger une description cohérente avec ce verdict (voir `medgemma_report_writer_prompt.txt`).
- Si le JSON renvoyé par le modèle n'est pas parsable, le code (`medgemma_describe_ollama` dans `app/app.py`) retombe sur un JSON minimal avec `predicted_class` forcé au verdict YOLO et `limitations: ["parsing failed"]`.
