

Assistant Radiologique Virtuel - État de l'Art
Thomas DURIAUD – Paul FONTAINE – Louis-Marie FORT – Ivana GARCIA – Jules GRALL –
Adam GREZE – Florian GIRAUD












## ÉTAT DE L'ART
## Assistant Radiologique Virtuel
ARVI-RX (Assistant Radiologique Virtuel Intelligent pour Radiographies X)
Thomas DURIAUD – Paul FONTAINE – Louis-Marie FORT – Ivana GARCIA – Jules GRALL –
Adam GREZE – Florian GIRAUD
## EFREI 2025-2026




























Assistant Radiologique Virtuel - État de l'Art
Thomas DURIAUD – Paul FONTAINE – Louis-Marie FORT – Ivana GARCIA – Jules GRALL –
Adam GREZE – Florian GIRAUD
## Introduction
Le projet «  Assistant  Radiologue  Virtuel »  vise  à  construire  un prototype  pédagogique  capable
d'analyser des radiographies thoraciques frontales et de les classer en trois catégories : Normal,
Suspicion d'opacité et Incertain. Ce document dresse l'état de l'art des technologies, modèles et
bases de données mobilisés dans ce type de système, en situant les choix du projet dans leur
contexte scientifique.
L'IA appliquée à la radiologie thoracique est aujourd'hui l'un des champs de recherche les plus
actifs en IA médicale. Les progrès récents des grands modèles de vision et de langage (Vision-
Language Models, VLMs) ouvrent la voie à des systèmes capables non seulement de classifier
une image, mais aussi de justifier leur résultat en langage naturel, ce qui constitue le cœur de ce
projet.

- Contexte clinique : la radiographie thoracique
1.1 Importance diagnostique
La  radiographie  thoracique  frontale  (Chest  X-Ray,  CXR)  est  l'un  des  examens  d'imagerie
médicale  les  plus  réalisés  au  monde.  Elle  permet  de  détecter  de  nombreuses  pathologies
pulmonaires : pneumonie, pneumothorax, épanchement pleural, atélectasie, masses tumorales,
cardiomégalie, etc. Sa disponibilité, son faible coût et sa rapidité d'acquisition en font un examen
de première intention dans de nombreux protocoles.
Cependant,  l'interprétation  d'une  CXR  requiert  une  expertise  radiologique  significative.  La
variabilité inter-lecteurs (désaccord entre deux radiologues) est documentée dans la littérature,
notamment pour les opacités subtiles ou les images de qualité réduite. C'est précisément dans
ce contexte que des outils d'aide à la décision basés sur l'IA présentent un intérêt pédagogique
et potentiellement clinique.
1.2 Limitations et enjeux
Les principales difficultés de l'analyse automatisée de CXR incluent : la faible résolution spatiale
des lésions subtiles, la superposition des structures anatomiques (côtes, cœur, vaisseaux), la
diversité  des  appareils  d'acquisition  et  des  protocoles  de  numérisation,  et  les  artefacts  liés  au
patient  (rotation,  inspiration  incomplète,  corps  étrangers).  Ces  éléments  justifient  l'introduction
d'une classe « Incertain » dans le projet, plutôt qu'une classification binaire réductrice.








- Évolution de l'IA en imagerie médicale

Assistant Radiologique Virtuel - État de l'Art
Thomas DURIAUD – Paul FONTAINE – Louis-Marie FORT – Ivana GARCIA – Jules GRALL –
Adam GREZE – Florian GIRAUD
2.1 Ère des réseaux de neurones convolutifs (CNN)
Dès 2017, Wang et al. (NIH) ont montré que des CNN entraînés sur de grandes bases de données
pouvaient  égaler  ou  surpasser  les  radiologues  sur  certaines  tâches  de  classification  de  CXR
(CheXNet,  Stanford,  2017).  Ces  approches  reposent  sur  des  architectures comme  DenseNet,
ResNet  ou  EfficientNet,  entraînées  de  manière  supervisée  sur  des  labels  extraits  de  rapports
radiologiques.
Ces  modèles  présentent  néanmoins  des  limites  importantes  :  ils  ne  produisent  qu'un  score  de
classification  sans  explication,  leur  généralisation  à  d'autres  hôpitaux  ou  appareils  est  souvent
médiocre (biais de distribution), et leur calibration, c'est-à-dire la fiabilité du score de confiance —
est régulièrement mise en défaut.
2.2 Vision Transformers (ViT) et modèles hybrides
Depuis  2020,  les  Vision  Transformers  (ViT)  ont  progressivement  remplacé  les  CNN  dans  les
benchmarks  de  pointe.  Leur  mécanisme  d'attention  permet  de  capturer  des  relations  longue
distance dans l'image, potentiellement plus adaptées à la détection de lésions diffuses comme
les opacités bilatérales de pneumonie. Des variantes comme BioViL-T ou CheXagent combinent
encodeur visuel et décodeur textuel, permettant de générer des descriptions radiologiques.
2.3 L'émergence des Vision-Language Models (VLMs)
La  génération  actuelle  de  VLMs — LLaVA,  GPT-4o,  Gemini,  Flamingo,  puis  leurs  variantes
médicales — représente un saut qualitatif majeur. Ces modèles multimodaux acceptent en entrée
une  image  et  un  texte  (prompt)  et  produisent  une  réponse  textuelle  structurée.  Appliqués  à  la
radiologie, ils permettent pour la première fois de combiner classification, localisation des signes
et justification clinique en langage naturel dans un seul système.

- Modèles clés pour le projet
3.1 MedGemma 4B — Baseline recommandée
MedGemma  est  un  modèle  Vision-Language  développé  par  Google,  spécialisé  pour  les
applications médicales. Basé sur l'architecture Gemma (famille PaliGemma / Gemma 3), il a été
pré-entraîné sur de larges corpus d'images médicales annotées, incluant des CXR. Sa version
4B (4 milliards de paramètres) offre un excellent équilibre entre performance et accessibilité sur
GPU modeste.
MedGemma est disponible sur Hugging Face (google/medgemma-4b-pt) sous conditions d'accès.
Il  supporte  nativement  les  images  en  entrée  et  peut  répondre  en  format  structuré  JSON  si  le
prompt l'indique. C'est le choix recommandé pour la baseline de ce projet : aucun entraînement
n'est requis, seul un prompting soigneux suffit pour obtenir des résultats exploitables.
3.2 Gemma 4 E2B/E4B + Unsloth — Voie de fine-tuning
Pour  aller  au-delà  du  prompting,  le  projet  prévoit  une  phase  optionnelle  de  fine-tuning  via
l'approche LoRA (Low-Rank Adaptation). Unsloth est une bibliothèque Python qui optimise le fine-
tuning de modèles Gemma en réduisant drastiquement la consommation mémoire GPU (jusqu'à

Assistant Radiologique Virtuel - État de l'Art
Thomas DURIAUD – Paul FONTAINE – Louis-Marie FORT – Ivana GARCIA – Jules GRALL –
Adam GREZE – Florian GIRAUD
4x par rapport au fine-tuning classique). Les variantes E2B (2 milliards de paramètres) et E4B (4
milliards) sont multimodales et supportent les images en entrée avant le texte.
LoRA consiste à n'entraîner qu'une petite fraction des paramètres du modèle (matrices de rang
bas insérées dans les couches d'attention), ce qui permet d'adapter un grand modèle avec peu
de données et peu de ressources. C'est la méthode PEFT (Parameter-Efficient Fine-Tuning) la
plus répandue en 2024-2025 pour les VLMs médicaux.
3.3 MAIRA-2 — Référence recherche
MAIRA-2  (Microsoft  Research,  2024)  est  un  modèle  spécialisé  pour  la  génération  de  rapports
radiologiques à partir de CXR. Très performant sur plusieurs benchmarks (RadGraph F1 : 27.2),
il  reste  réservé  à  la  recherche  avancée  en  raison  de  sa  complexité  d'intégration  et  de  ses
exigences computationnelles. Il sert ici de référence pour situer les ambitions du projet.
3.4 CNN / ViT légers — Classifieurs de support
Des  modèles  CNN légers  (ResNet-18,  EfficientNet-B0)  ou  des  ViT de petite  taille peuvent  être
utilisés en complément du VLM comme classifieurs rapides. Leur rôle dans l'architecture du projet
est  de  fournir  un  score  de  confiance  complémentaire  permettant  de déclencher  la  classe  «
Incertain » en cas d'ambiguïté, renforçant ainsi les garde-fous du système.

Modèle Type Paramètres Usage dans le
projet
## Performance
repère
MedGemma 4B VLM médical 4B Baseline
## (prompting)
## VQA-RAD 70.2%
## Gemma 4
## E2B/E4B
VLM généraliste 2-4B Fine-tuning LoRA Macro-F1 ~89.5
## MIMIC
MAIRA-2 VLM spécialisé
## CXR
## Large Référence
recherche
RadGraph F1 27.2
CNN / ViT léger Classifieur visuel <100M Support
confiance
Variable selon tâche





- Bases de données de référence
4.1 RSNA Pneumonia Detection Challenge — Choix principal
Le dataset RSNA Pneumonia (Kaggle, 2018) regroupe environ 30 000 radiographies thoraciques
frontales annotées avec des bounding boxes pour les zones de pneumonie et des labels de classe
(Normal  /  Pneumonie  /  Non-pneumonie).  Issu  d'un  challenge  compétitif  organisé  par  la
Radiological Society of North America, il est public, bien documenté et de qualité contrôlée. C'est
le dataset  recommandé pour  ce projet  étudiant en raison de  sa  taille accessible,  de  ses  labels
fiables et de l'absence de restriction d'accès.

Assistant Radiologique Virtuel - État de l'Art
Thomas DURIAUD – Paul FONTAINE – Louis-Marie FORT – Ivana GARCIA – Jules GRALL –
Adam GREZE – Florian GIRAUD
4.2 CheXpert — Extension
CheXpert (Stanford AIMI, 2019) contient 224 316 radiographies thoraciques avec 14 pathologies
étiquetées  à  partir  de  rapports  radiologiques  via  NLP.  Sa  richesse  en  annotations  en  fait  une
référence incontournable de la littérature. Cependant, les labels sont extraits automatiquement et
présentent du bruit, notamment pour les cas incertains (labels U-ones / U-zeros). Il est utilisé dans
ce projet pour les discussions scientifiques comparatives.
4.3 MIMIC-CXR — Niveau avancé
MIMIC-CXR (PhysioNet, 2019) est l'un des plus grands datasets publics de CXR : 377 110 images
associées à 227 835 rapports radiologiques de patients du Beth Israel Deaconess Medical Center.
Son accès est contrôlé (inscription PhysioNet + formation éthique obligatoire) et les données ne
sont pas redistribuables. Sa richesse clinique (rapports complets, métadonnées patients) en fait
la référence pour les systèmes de génération de rapports. Il est cité dans ce projet comme repère
de littérature.
4.4 NIH ChestX-ray14 — Contexte
Publié par le NIH en 2017, ce dataset de 112 000 images avec 14 labels a lancé la dynamique
de  la  recherche  en  classification  automatique  de  CXR.  Il  souffre  de  biais  de  labellisation
documentés  (extraction  NLP  imparfaite)  mais  reste  une  référence  historique  importante  pour
comprendre l'évolution du domaine.

Dataset Taille Labels Accès Rôle dans le projet
## RSNA
## Pneumonia
~30 000 3 classes +
bboxes
Public (Kaggle) Dataset principal
CheXpert 224 316 14 pathologies Public (Stanford) Discussion /
extension
MIMIC-CXR 377 110 Rapports
complets
## Contrôlé
(PhysioNet)
Référence littérature
NIH ChestX-
ray14
## ~112
## 000
14 pathologies Public (NIH) Contexte historique

- Techniques de prompting pour VLMs médicaux
5.1 Importance du prompt engineering
Pour  les  VLMs  utilisés  sans  fine-tuning,  la  qualité  du  prompt  est  déterminante.  Des  travaux
récents (Wei et al., 2022 ; Singhal et al., 2023, Med-PaLM) montrent que des techniques comme
le Chain-of-Thought prompting (raisonnement étape par étape), le few-shot prompting (exemples
dans le prompt) ou le prompting structuré (sortie JSON forcée) améliorent significativement les
performances et la fiabilité des sorties.
5.2 Contrat de sortie structuré

Assistant Radiologique Virtuel - État de l'Art
Thomas DURIAUD – Paul FONTAINE – Louis-Marie FORT – Ivana GARCIA – Jules GRALL –
Adam GREZE – Florian GIRAUD
Une pratique émergente dans les systèmes d'IA médicale est le « contrat de sortie » : forcer le
modèle à produire une réponse dans un format JSON strictement défini incluant la classe prédite,
le niveau de confiance, les preuves visuelles observées, la justification clinique, les limitations et
un avertissement obligatoire. Cette approche réduit les hallucinations et garantit la traçabilité des
décisions, deux exigences centrales du cahier des charges de ce projet.
L'objectif du projet est d'atteindre ≥ 95% de JSON valides en sortie, contre environ 75% pour un
prompt naïf, et de réduire le taux d'hallucination textuelle à 0%.
5.3 Gestion de l'incertitude
La littérature sur la fiabilité des modèles médicaux (Kompa et al., 2021 ; Begoli et al., 2019) insiste
sur la nécessité d'exprimer l'incertitude plutôt que de produire une réponse assurée incorrecte.
Le mécanisme de classe « Incertain » retenu dans ce projet s'inscrit dans cette tradition : lorsque
la confiance est inférieure à un seuil (0.60) ou que la qualité de l'image est insuffisante, le système
doit déclarer son incertitude plutôt que de forcer une classification.

- Fine-tuning et adaptation de domaine
6.1 Transfer learning en imagerie médicale
Le  transfer  learning, adapter  un  modèle  pré-entraîné  sur  de  grandes  données  générales  à  un
domaine spécifique avec peu de données, est la stratégie dominante en IA médicale depuis 2018.
Les  modèles  pré-entraînés  sur  ImageNet  ou  sur  des  corpus  médicaux  (comme  MedGemma)
capturent  des  représentations  visuelles  génériques  que  l'on  peut  spécialiser  avec  quelques
centaines d'exemples annotés.
6.2 LoRA et PEFT — État de l'art du fine-tuning efficace
Low-Rank Adaptation (LoRA, Hu et al., 2021) est aujourd'hui la méthode PEFT de référence pour
adapter les grands modèles de langage et de vision. Elle consiste à injecter des matrices de faible
rang dans les couches d'attention, réduisant le nombre de paramètres entraînés de 99% tout en
conservant  des  performances  proches  du  fine-tuning  complet.  QLoRA  (Dettmers  et  al.,  2023)
pousse plus loin en quantifiant le modèle de base en 4 bits, permettant le fine-tuning sur GPU de
## 16 Go.
Dans  le  contexte  de  ce  projet,  Unsloth  +  Gemma  4  E2B/E4B  constitue  la  voie  rapide
recommandée pour un fine-tuning expérimental : multimodal, compatible FastVisionModel, avec
finetune_vision_layers=False   au   début   pour   préserver   les   représentations   visuelles pré-
entraînées.
6.3 Risques du fine-tuning
Le  fine-tuning  sur  peu  de  données  présente  des  risques  documentés  en  médecine  :  sur-
apprentissage  (overfitting)  sur  les  biais  du  dataset  d'entraînement,  catastrophic  forgetting  des
connaissances médicales générales, et dégradation de la calibration. C'est pourquoi le cahier des
charges impose de valider la baseline avant tout fine-tuning, une bonne pratique qui reflète l'état
de l'art en IA médicale responsable.

Assistant Radiologique Virtuel - État de l'Art
Thomas DURIAUD – Paul FONTAINE – Louis-Marie FORT – Ivana GARCIA – Jules GRALL –
Adam GREZE – Florian GIRAUD

- Évaluation et métriques en IA médicale
7.1 Métriques standard
L'évaluation d'un système de classification médicale requiert des métriques adaptées aux classes
déséquilibrées et aux enjeux asymétriques (faux négatif = pathologie manquée, faux positif = sur-
alerte). Les métriques retenues pour ce projet sont :
- Accuracy globale (≥ 0.68) : proportion de cas correctement classifiés
- Macro-F1 (≥ 0.70) : moyenne des F1-scores par classe, pénalisant les classes
minoritaires sous-performantes
- Sensibilité (recall) élevée : priorité à ne pas manquer les opacités
- Spécificité équilibrée : limiter les fausses alertes

7.2 Calibration et fiabilité
La  calibration  d'un  modèle  mesure  la  corrélation  entre  le  score  de  confiance  affiché  et  la
probabilité  réelle  d'être  correct.  Un  modèle  bien  calibré  qui  affiche  80%  de  confiance  doit  être
correct dans 80% des cas. Cette propriété est essentielle en médecine. Des métriques comme
l'Expected Calibration Error (ECE) permettent de la quantifier, même si elles ne sont pas exigées
dans ce prototype pédagogique.
7.3 Analyse d'erreurs
Au-delà  des  métriques  agrégées,  l'état  de  l'art  en  IA  médicale  responsable  exige  une  analyse
qualitative des erreurs  :  catégorisation  des  faux positifs  et  faux négatifs,  identification  des  cas-
limites, détection des hallucinations textuelles. Cette démarche, prévue à l'étape 6 du projet, est
conforme aux recommandations des organismes de régulation (FDA, CE) pour les logiciels d'aide
au diagnostic.


- Architecture système et traçabilité
8.1 Pipeline de traitement d'images médicales
Un  système  d'analyse  d'images  médicales  robuste  requiert  un  pipeline  de  prétraitement
standardisé  :  normalisation  des  valeurs  de  pixels  (DICOM  →  PNG/JPEG  normalisé),
redimensionnement  à  la  résolution  d'entrée  du  modèle,  amélioration  optionnelle  du  contraste
(CLAHE),  et  anonymisation  des  métadonnées  DICOM  (suppression  des  informations  patient).
Ces étapes garantissent la reproductibilité et la conformité RGPD.
8.2 Journalisation et traçabilité
Les systèmes d'IA en santé sont soumis à des exigences de traçabilité croissantes (règlement
européen  2017/745  sur  les  dispositifs  médicaux,  futur  AI  Act).  Dans  ce  projet  pédagogique,  la

Assistant Radiologique Virtuel - État de l'Art
Thomas DURIAUD – Paul FONTAINE – Louis-Marie FORT – Ivana GARCIA – Jules GRALL –
Adam GREZE – Florian GIRAUD
journalisation  via  SQLite  (résultats,  logs  d'inférence,  métadonnées)  reflète  cette  exigence.
Chaque prédiction doit être enregistrée avec son horodatage, le prompt utilisé, la réponse brute
et le JSON parsé.
8.3 Garde-fous et IA responsable
L'état  de  l'art  en  IA  médicale  responsable  recommande  l'intégration  de  garde-fous  explicites  :
refus   de   produire   un   diagnostic   définitif,   classe   d'incertitude   obligatoire,   avertissement
systématique  à  l'utilisateur,  et  interdiction  de remplacer  l'avis  d'un professionnel  de  santé.  Ces
principes, formalisés dans le cahier des charges du projet (0 hallucination tolérée, 100% de sorties
avec warning), s'inscrivent dans les lignes directrices publiées par l'OMS (2021) et la Commission
Européenne sur l'IA digne de confiance.

- Positionnement du projet dans l'état de l'art
Ce tableau situe le projet par rapport aux systèmes de référence de la littérature :

Dimension Systèmes de pointe (littérature) Ce projet (objectif pédagogique)
Dataset MIMIC-CXR (377K), CheXpert (224K) RSNA Pneumonia (~30K, 100-150 cas)
Modèle MAIRA-2, CheXagent, Med-PaLM 2 MedGemma 4B (baseline prompting)
Fine-tuning Full fine-tuning sur GPU A100 LoRA optionnel (Unsloth, E2B/E4B)
Macro-F1 Jusqu'à 89.5 (MIMIC top-5) Cible ≥ 0.70
Sortie Rapport structuré, localisation JSON 3 classes + justification
Traçabilité Variable selon les systèmes SQLite obligatoire, logs complets
Incertitude Souvent ignorée en recherche Classe dédiée, garde-fous stricts



- Solutions envisagées
Sur la base de l'état de l'art dressé dans les sections précédentes, deux solutions ont été retenues
et seront comparées dans ce projet : MedGemma 4B, utilisé en mode baseline par prompting, et
Gemma 4 E2B, adapté via fine-tuning LoRA. Ces deux approches représentent les deux grandes
stratégies  actuelles d'exploitation  des  VLMs en  IA médicale — l'une  sans entraînement,  l'autre
avec adaptation légère de domaine.

10.1 Solution 1 — MedGemma 4B
## Présentation

Assistant Radiologique Virtuel - État de l'Art
Thomas DURIAUD – Paul FONTAINE – Louis-Marie FORT – Ivana GARCIA – Jules GRALL –
Adam GREZE – Florian GIRAUD
MedGemma  4B  est  un  Vision-Language  Model  développé  par  Google  DeepMind  et  spécialisé
pour les applications médicales. Pré-entraîné sur de larges corpus d'images médicales annotées,
incluant  des  radiographies  thoraciques,  il  dispose  d'une  représentation  visuelle  médicale
intrinsèque  sans  nécessiter  d'entraînement  supplémentaire.  Il  est  accessible  librement  sur
Hugging Face (google/medgemma-4b-pt) sous conditions d'accès académiques.
Dans cette solution, le modèle est utilisé tel quel, en mode inférence pure. La stratégie repose
exclusivement sur la qualité du prompt : un prompt structuré fournit au modèle le contexte clinique,
le  schéma  de  sortie  JSON  attendu,  les  trois  classes  possibles  et  les  consignes  de  prudence
(avertissement obligatoire, classe Incertain si doute). Aucun GPU d'entraînement n'est requis, un
GPU de 8 à 16 Go suffit pour l'inférence.
## Avantages
- Mise en œuvre immédiate : pas d'entraînement, déploiement en quelques heures
- Connaissances médicales intégrées : pré-entraîné sur des données radiologiques, il
comprend la terminologie et les signes cliniques
- Itérations rapides : améliorer les résultats revient à affiner le prompt, sans cycle
d'entraînement
- Faible risque d'overfitting : aucune adaptation sur les données du projet, donc pas de
sur-apprentissage
- Reproductibilité : la même inférence avec le même prompt produit des résultats stables
## Limites
- Absence de spécialisation sur le dataset cible (RSNA Pneumonia) : le modèle ne connaît
pas les spécificités de ce corpus
- Dépendance forte à la qualité du prompt : un prompt mal conçu dégrade
significativement les performances
- Calibration non contrôlée : le score de confiance produit n'est pas nécessairement bien
calibré sur cette tâche précise
- Latence d'inférence variable selon le matériel disponible



10.2 Solution 2 — Gemma 4 E2B
## Présentation
Gemma  4  E2B  est  un  modèle  Vision-Language  généraliste  de  2  milliards  de  paramètres
développé par Google. Sa variante E2B (Efficient 2B) est multimodale : elle accepte une image
et un texte en entrée et produit une réponse textuelle. Par rapport à MedGemma, elle ne dispose
pas  de  pré-entraînement  médical  spécifique,  mais  sa  taille  réduite  la  rend  particulièrement
adaptée à un fine-tuning rapide via LoRA sur des ressources GPU modestes.
La stratégie retenue est le fine-tuning LoRA avec Unsloth, un framework Python qui optimise la
mémoire GPU (réduction jusqu'à 4x par rapport au fine-tuning standard). Le modèle est adapté
sur un sous-ensemble annoté du dataset RSNA Pneumonia (100 à 150 cas de développement),
avec les paramètres de vision gelés au départ (finetune_vision_layers=False) pour préserver les
représentations visuelles pré-entraînées. Seules les couches d'attention du décodeur textuel sont
adaptées via les matrices LoRA de rang bas.

Assistant Radiologique Virtuel - État de l'Art
Thomas DURIAUD – Paul FONTAINE – Louis-Marie FORT – Ivana GARCIA – Jules GRALL –
Adam GREZE – Florian GIRAUD
## Avantages
- Spécialisation sur le dataset cible : le modèle apprend les spécificités visuelles du
corpus RSNA Pneumonia
- Format de sortie maîtrisé : le fine-tuning sur des exemples JSON valides améliore la
conformité de sortie (cible ≥ 95%)
- Réduction des hallucinations : l'entraînement sur des cas annotés par des experts limite
l'invention de termes cliniques
- Efficacité mémoire : LoRA + Unsloth permettent de fine-tuner sur un GPU de 16 Go
- Export GGUF possible : le modèle adapté peut être exporté pour un déploiement léger et
local
## Limites
- Risque d'overfitting sur peu de données (100-150 cas) : une régularisation et une
validation rigoureuse sont nécessaires
- Absence de pré-entraînement médical : Gemma 4 E2B part de représentations
générales, sans connaissance médicale intrinsèque comme MedGemma
- Cycle plus long : chaque itération nécessite un cycle d'entraînement avant évaluation
- Dépendance aux conditions d'accès Unsloth et à la disponibilité du GPU












10.3 Comparaison des deux solutions
Critère MedGemma 4B (prompting) Gemma 4 E2B (LoRA)
Type d'approche Zero-shot / few-shot
prompting
Fine-tuning supervisé léger
Pré-entraînement médical Oui (Google DeepMind) Non (généraliste)
Entraînement requis Non Oui (LoRA, ~1-2h sur GPU
## 16 Go)
GPU inférence 8-16 Go 8-16 Go
Spécialisation RSNA Non (connaissance générale) Oui (100-150 cas ciblés)
Conformité JSON attendue ~75% (baseline) → ≥95%
(prompt amélioré)
≥95% (appris par fine-tuning)
Risque d'hallucination Modéré (prompt à contrôler) Faible (si données annotées)

Assistant Radiologique Virtuel - État de l'Art
Thomas DURIAUD – Paul FONTAINE – Louis-Marie FORT – Ivana GARCIA – Jules GRALL –
Adam GREZE – Florian GIRAUD
Risque d'overfitting Nul (pas d'entraînement) Modéré (peu de données)
Vitesse d'itération Très rapide (modifier le
prompt)
Lente (cycle entraînement)
Latence inférence < 10 s (cible) < 10 s (cible)
Export / déploiement Via API ou pipeline HF Export GGUF possible
Complexité d'intégration Faible Moyenne
Rôle dans le projet Baseline obligatoire Amélioration optionnelle

10.4 Comparaison des datasets
Le choix du dataset d'entraînement et d'évaluation est une décision structurante pour tout projet
d'IA  médicale.  Il  conditionne  la  qualité  des  labels,  la  diversité  des  cas,  la  facilité  d'accès  et  la
pertinence par rapport à la tâche cible. Quatre datasets de référence ont été analysés et comparés
: RSNA Pneumonia, CheXpert, MIMIC-CXR et NIH ChestX-ray14.

Présentation des datasets
RSNA Pneumonia Detection Challenge
Publié en 2018 par la Radiological Society of North America (RSNA) dans le cadre d'un challenge
Kaggle, ce dataset contient environ 30 000 radiographies thoraciques frontales. Chaque image
est annotée par des radiologues certifiés avec un label de classe (Normal ou Pneumonie) et, pour
les cas positifs, des bounding boxes délimitant précisément les zones d'opacité pulmonaire. Le
périmètre est volontairement restreint à la détection de pneumonie, ce qui en fait un dataset ciblé,
cohérent et parfaitement adapté à un projet à tâche unique.

CheXpert
Développé  par  Stanford  University  (AIMI)  et  publié  en  2019,  CheXpert  regroupe  224  316
radiographies thoraciques issues de l'hôpital Stanford, associées à 14 labels de pathologies. Les
labels  sont  extraits  automatiquement  des  rapports  radiologiques  par  un  pipeline  NLP,  ce  qui
introduit du bruit, notamment pour les cas incertains, représentés par des labels dits U-ones ou
U-zeros. Sa richesse en volume et en pathologies en fait une référence majeure de la littérature,
mais sa complexité le rend moins adapté à un projet pédagogique focalisé sur une tâche unique.
## MIMIC-CXR
MIMIC-CXR (PhysioNet, 2019) est le dataset public le plus complet en imagerie thoracique : 377
110 radiographies issues du Beth Israel Deaconess Medical Center, associées à 227 835 rapports
radiologiques complets rédigés par des radiologues. Son accès est strictement contrôlé et requiert
une inscription sur la plateforme PhysioNet, une formation à l'éthique de la recherche médicale et
la signature d'un accord de non-redistribution. Cette richesse clinique en fait la référence pour les
systèmes de génération de rapports, mais les contraintes d'accès le rendent inadapté à un projet
étudiant de sept semaines.

Assistant Radiologique Virtuel - État de l'Art
Thomas DURIAUD – Paul FONTAINE – Louis-Marie FORT – Ivana GARCIA – Jules GRALL –
Adam GREZE – Florian GIRAUD
NIH ChestX-ray14
Publié par le National Institutes of Health en 2017, ce dataset de 112 120 radiographies avec 14
labels a été le premier grand dataset public de CXR et a joué un rôle fondateur dans la recherche
en classification automatique. Ses labels sont issus d'extraction NLP sur des rapports médicaux,
avec   des   biais   de   labellisation   documentés   dans   la   littérature.   Il   constitue   aujourd'hui
principalement une référence historique et de contexte.

Tableau comparatif

Critere RSNA
## Pneumonia
CheXpert MIMIC-CXR NIH ChestX-ray14
## Volume ~30 000 224 316 377 110 ~112 000
## Pathologies Pneumonie
uniquement
## 14
pathologies
## 14+
pathologies
14 pathologies
Type de
labels
## Expert
radiologue
## NLP
automatique
## NLP +
rapports
NLP automatique
Qualite des
labels
## Elevee (bboxes) Moyenne
## (bruit)
## Elevee
## (rapports
complets)
Moyenne (biais NLP)
## Bounding
boxes
## Oui Non Non Non
## Rapports
textuels
## Non Partiel Oui
## (complets)
## Non
Acces Public (Kaggle) Public
(Stanford)
## Controle
(PhysioNet)
Public (NIH)
## Formalites
d'acces
## Aucune Inscription
simple
## Formation
ethique
requise
## Aucune
## Tache
adaptee
Oui (3 classes) Partiel (trop
large)
## Non (trop
complexe)
## Partiel
## Complexite
integration
## Faible Moyenne Elevee Moyenne
## Projet
etudiant
## Oui Extension
possible
Non Contexte uniquement

Justification du choix : RSNA Pneumonia
Au terme de cette comparaison, le dataset RSNA Pneumonia est retenu comme dataset principal
du projet. Ce choix repose sur cinq arguments convergents.


Assistant Radiologique Virtuel - État de l'Art
Thomas DURIAUD – Paul FONTAINE – Louis-Marie FORT – Ivana GARCIA – Jules GRALL –
Adam GREZE – Florian GIRAUD
- Un périmètre aligné avec la tâche
Le projet classe les radiographies en trois catégories : Normal, Suspicion d'opacité et Incertain.
RSNA Pneumonia est le seul dataset dont le périmètre est nativement centré sur cette tâche : il
se  concentre  exclusivement  sur  la  pneumonie,  sans  dispersion sur  d'autres  pathologies.  Les
labels Normal et Pneumonie correspondent directement aux deux classes principales du projet,
tandis que la classe Incertain est gérée par les garde-fous du système.
- Des labels fiables annotés par des radiologues
Contrairement à CheXpert et NIH ChestX-ray14, dont les labels sont extraits automatiquement
par NLP avec du bruit documenté, les annotations RSNA ont été produites par des radiologues
certifiés. Cette fiabilité est essentielle pour entraîner et évaluer un prototype dont l'un des objectifs
est d'atteindre 0% d'hallucination et des métriques d'évaluation solides.
- Des bounding boxes disponibles
RSNA Pneumonia est le seul dataset de cette comparaison à fournir des bounding boxes sur les
zones d'opacité. Bien que non requises dans la version de base du projet, ces annotations ouvrent
la voie à des extensions expérimentales, validation visuelle des prédictions, localisation des zones
suspectes, sans nécessiter de dataset supplémentaire.
- Un accès libre et immédiat
Disponible sur Kaggle sans formalité particulière, RSNA Pneumonia peut être téléchargé et utilisé
dès le démarrage du projet. À l'opposé, MIMIC-CXR impose une inscription sur PhysioNet, une
formation  éthique  et  un  accord  de  non-redistribution,  des  contraintes  incompatibles  avec  un
calendrier de sept semaines.
- Une taille adaptée aux ressources
Avec environ 30 000 images, RSNA Pneumonia offre suffisamment de données pour développer
et  évaluer  un  modèle  robuste,  tout  en  restant  gérable  en  termes  de  stockage  et  de  temps  de
traitement. Le sous-ensemble utilisé dans ce projet, 100 à 150 cas de développement et 30 cas
finaux  commentés, est  extrait de  ce  corpus,  garantissant  la  représentativité  sans  surcharger le
pipeline.

Choix  retenu  :  RSNA  Pneumonia  comme  dataset  principal.  CheXpert  et  MIMIC-CXR  sont
mobilisés uniquement comme références de littérature pour la mise en perspective scientifique
des résultats. NIH ChestX-ray14 est cité à titre de contexte historique.

## Conclusion
L'état de l'art en analyse automatisée de radiographies thoraciques est dominé par des modèles
multimodaux  de  grande  taille  (VLMs),  entraînés  sur  des  datasets  de  centaines  de  milliers
d'images avec des techniques d'adaptation de domaine efficaces (LoRA, QLoRA). Les meilleurs
systèmes atteignent des Macro-F1 supérieurs à 0.85 sur des benchmarks comme MIMIC-CXR.
Ce projet pédagogique ne cherche pas à atteindre ces performances, mais à en comprendre les
fondements : choisir et évaluer une baseline (MedGemma 4B par prompting), concevoir un contrat
de sortie structuré et traçable, implémenter des garde-fous responsables, et conduire une analyse

Assistant Radiologique Virtuel - État de l'Art
Thomas DURIAUD – Paul FONTAINE – Louis-Marie FORT – Ivana GARCIA – Jules GRALL –
Adam GREZE – Florian GIRAUD
d'erreurs  rigoureuse.  Ces  compétences  sont  directement  transférables  aux  systèmes  de
production en IA médicale.
Les choix technologiques du projet, MedGemma comme baseline, Unsloth+Gemma comme voie
de fine-tuning, RSNA Pneumonia comme dataset principal, sont cohérents avec l'état de l'art et
adaptés aux contraintes d'un projet étudiant de 7 semaines.

Références bibliographiques
Wang, X. et al. (2017). ChestX-ray8 : Hospital-scale Chest X-ray Database and Benchmarks. CVPR 2017.
Irvin, J. et al. (2019). CheXpert : A Large Chest Radiograph Dataset with Uncertainty Labels and Expert
Comparison. AAAI 2019.
Johnson, A. et al. (2019). MIMIC-CXR : A large publicly available database of labeled chest radiographs.
PhysioNet.
Hu, E. et al. (2021). LoRA : Low-Rank Adaptation of Large Language Models. ICLR 2022.
Dettmers, T. et al. (2023). QLoRA : Efficient Finetuning of Quantized LLMs. NeurIPS 2023.
Singhal, K. et al. (2023). Large Language Models Encode Clinical Knowledge. Nature, 620, 172-180.
Google Health. (2024). MedGemma : Medical Vision-Language Model. Hugging Face Model Card.
Bannur, S. et al. (2024). MAIRA-2 : Grounded Radiology Report Generation. Microsoft Research.
Unsloth AI. (2024). Gemma 4 Fine-tuning Guide. https://unsloth.ai/docs/models/gemma-4/train
RSNA. (2018). RSNA Pneumonia Detection Challenge. Kaggle / Radiological Society of North America.