ARVI-RX (Assistant Radiologique Virtuel Intelligent pour Radiographies X)

Thème:
IA médicale multimodale, vision-language models et prototype web traçable pour l’aide pédagogique à l’analyse d’images médicales.

Mot clés:
IA médicale multimodale, Radiographie thoracique, Vision-Language Models

Problème / défi de l'appel d'offre:
L’interprétation d’images médicales est un domaine où les modèles d’IA multimodale progressent rapidement, mais où le risque d’erreur, de surconfiance et d’hallucination reste critique. Le défi proposé n’est donc pas de créer un outil de diagnostic clinique, mais de concevoir un assistant pédagogique capable d’analyser de manière prudente une radiographie thoracique frontale et de produire une sortie structurée, traçable et explicable.
Le projet abordera trois difficultés principales. Premièrement, transformer un modèle vision-langage en système logiciel mesurable, avec entrée image, prétraitement, prompt, sortie JSON, garde-fous, logs et interface web. Deuxièmement, apprendre à évaluer une IA médicale autrement qu’à travers une démonstration impressionnante : métriques, faux positifs, faux négatifs, incertitude, hallucinations et analyse d’erreurs. Troisièmement, former les étudiants à une posture responsable : aucune conclusion clinique définitive, avertissement systématique, validation humaine et documentation des limites.
Le besoin est donc double : construire un prototype fonctionnel, mais surtout apprendre à produire une preuve d’ingénierie fiable dans un contexte sensible.

Objectif de l'appel d'offre:
L’objectif est de développer un prototype web d’assistant radiologique virtuel à finalité pédagogique. Le système devra recevoir une radiographie thoracique frontale, l’analyser à l’aide d’un modèle multimodal ou d’un pipeline hybride, puis retourner une réponse structurée comprenant la qualité de l’image, une classe prédite, un score de confiance contrôlé, des éléments visuels observés, une justification courte, des limites et un avertissement clair.
Le périmètre fonctionnel sera volontairement limité à trois sorties : normal, suspicion d’opacité, ou incertain. La classe “incertain” devra être considérée comme une décision de sécurité lorsque la qualité de l’image est faible ou que les signes visuels sont insuffisants.
La solution devra comparer une baseline par prompting à une amélioration légère : prompt renforcé, ensemble de prompts, garde-fou, classifieur auxiliaire ou adaptation LoRA si le groupe dispose des ressources nécessaires. La priorité sera donnée à la reproductibilité : notebook baseline, scripts d’évaluation, base SQLite ou CSV de logs, dashboard de métriques, registre d’erreurs et démonstration web.
L’objectif final est de livrer un prototype fiable, documenté et défendable, non un dispositif médical.

Contexte et tendance dans le domaine:
Les modèles image-texte, aussi appelés vision-language models, permettent de fournir une image et une instruction textuelle en entrée afin de produire une réponse textuelle structurée. Cette famille technologique est adaptée au projet, car elle permet de combiner radiographie, prompt de contrôle, justification et format de sortie exploitable.
Dans le domaine médical, MedGemma constitue une référence pertinente pour un projet étudiant : Google le présente comme une famille de modèles entraînés pour la compréhension médicale texte-image, avec des variantes multimodales utilisables comme base de développement. Il faut toutefois conserver une position prudente : Google indique que ces modèles ne doivent pas informer directement un diagnostic ou une décision clinique sans vérification indépendante.
Côté données, le contexte est favorable car plusieurs bases publiques existent. RSNA Pneumonia contient environ 30 000 radiographies thoraciques frontales issues du NIH CXR8, avec vues PA/AP et annotations liées à la détection de pneumonie/opacités. CheXpert contient 224 316 radiographies de 65 240 patients avec rapports associés, et MIMIC-CXR contient 377 110 images correspondant à 227 835 études radiographiques.
La tendance est donc claire : les outils existent, mais l’enjeu pédagogique est de savoir les encadrer, les mesurer et les limiter.

Quels sont les principaux résultats de l'appel d'offre pour réaliser cette plus-value:
Les résultats attendus sont organisés en livrables vérifiables.
Le premier résultat est un dépôt GitHub structuré, contenant le README, les notebooks, les scripts d’inférence, les prompts, les fichiers d’évaluation, le schéma SQLite, les templates d’interface et la documentation éthique.
Le deuxième résultat est une baseline fonctionnelle : un modèle vision-langage reçoit une radiographie et un prompt fixe, puis retourne une réponse JSON contenant classe, confiance, justification, limites et avertissement.
Le troisième résultat est une amélioration mesurée : prompt renforcé, ensemble de prompts, garde-fou d’incertitude, classifieur léger ou LoRA expérimental. Cette amélioration devra être comparée à la baseline avec des métriques explicites : accuracy, macro-F1, sensibilité, spécificité, validité JSON, taux d’incertitude, hallucinations et latence.
Le quatrième résultat est une application web simple : upload d’image, affichage de la radiographie, analyse, sortie structurée, warning et sauvegarde du run.
Le cinquième résultat est un rapport critique avec 20 à 30 cas commentés : réussites, faux positifs, faux négatifs, incertitudes acceptables, erreurs de format, hallucinations et actions correctives.

Développemen de la solution. Must/Should/Could:
--Must have
Une application web permettant de déposer une radiographie thoracique frontale, d’exécuter une analyse IA, d’afficher une classe parmi normal / suspicion d’opacité / incertain, de fournir une justification courte, un score de confiance, des limites et un avertissement non clinique. Une baseline par prompting doit être disponible, accompagnée d’un fichier de résultats et d’une première évaluation. Les sorties doivent être structurées en JSON et journalisées.
--Should have
Une comparaison baseline vs amélioration légère : prompt renforcé, vote entre prompts, seuil d’incertitude, classifieur léger ou garde-fous. Un dashboard doit afficher les métriques principales : accuracy, macro-F1, sensibilité, spécificité, validité JSON, latence, taux d’incertitude et erreurs. Une base SQLite doit stocker les cas, prompts, modèles, sorties et évaluations.
--Could have 
Une expérimentation LoRA sur Gemma 4 multimodal via Unsloth ou une adaptation MedGemma via PEFT/QLoRA, uniquement après stabilisation de la baseline. Un module de localisation visuelle ou d’explication graphique peut être ajouté en bonus, mais ne doit pas remplacer l’évaluation.

Exigences fonctionnels:
La solution proposée devra inclure les fonctionnalités suivantes :
-dépôt d’une image de radiographie thoracique frontale ;
-affichage de l’image uploadée ;
-prétraitement minimal : format, taille, normalisation, contrôle qualité basique ;
-inférence via modèle vision-langage ou pipeline hybride ;
-sortie JSON structurée avec : qualité image, classe prédite, confiance, -éléments visuels, justification, limites, avertissement ;
-trois classes obligatoires : normal, suspicion d’opacité, incertain ;
-règle d’incertitude en cas d’image pauvre, signes faibles, JSON invalide ou confiance insuffisante ;
-sauvegarde des prédictions, prompts, versions de modèle, latence et commentaires ;
-dashboard d’évaluation ;
-comparaison baseline vs version améliorée ;
-registre d’erreurs avec faux positifs, faux négatifs, incertitudes acceptables et hallucinations ;
-rapport final avec 20 à 30 cas commentés.
Fonctionnalité interdite : présenter la sortie comme un diagnostic médical. Le système doit toujours être décrit comme un prototype pédagogique.

Exigences techniques:
La solution devra être développée principalement en Python. La stack recommandée est :
-Python 3.11 ;
-PyTorch ;
-Hugging Face Transformers ;
-Pillow, OpenCV, pydicom pour le traitement image ;
-Pandas, scikit-learn pour les métriques ;
-Gradio ou Streamlit pour une première démonstration rapide ;
-FastAPI + Uvicorn + Pydantic pour une API plus propre ;
-SQLite pour les logs et l’évaluation ;
-GitHub pour le versioning ;
-Docker en bonus ;
-notebooks Colab/Jupyter pour les expérimentations.
Gradio est adapté à une première interface rapide, car sa documentation le présente comme un moyen de construire et partager des démos ML en quelques lignes de Python. FastAPI est recommandé pour exposer /predict, car il s’agit d’un framework Python moderne et performant fondé sur les type hints.
Pour l’optimisation, Unsloth peut être utilisé sur Gemma 4 multimodal : la documentation indique que les variantes E2B/E4B sont les principales variantes multimodales, qu’il faut charger le modèle avec FastVisionModel, commencer avec finetune_vision_layers = False, et privilégier LoRA avant fine-tuning complet.

Sécurité:
La solution devra être conçue comme un prototype pédagogique non clinique. Elle ne devra jamais manipuler de données personnelles réelles ni de dossiers patients identifiables. Les images utilisées devront provenir de datasets publics autorisés, de cas synthétiques ou de données explicitement dé-identifiées. Aucun nom, identifiant patient, date de naissance ou information clinique personnelle ne devra être collecté.
Sur le plan applicatif, l’upload devra être limité aux formats autorisés, les fichiers devront être validés avant traitement, et les résultats devront être stockés sans information nominative. Les logs conserveront uniquement les identifiants de cas, chemins de fichiers, versions de prompts, modèles, prédictions, métriques et commentaires techniques.
Sur le plan médical, la sécurité reposera sur quatre garde-fous : avertissement visible, classe “incertain”, interdiction de diagnostic définitif, et obligation de validation humaine. Le système devra afficher explicitement : “Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise.”
Sur le plan IA, la solution devra contrôler les hallucinations par prompts structurés, format JSON, vérification automatique, analyse d’erreurs et refus des conclusions non justifiées par l’image.

Critères d'évaluation:
Les propositions seront évaluées selon les critères suivants :
-Clarté du périmètre : 15 %
Un seul cas d’usage, trois classes, limites non cliniques clairement définies.
-Baseline fonctionnelle : 15 %
Notebook reproductible, prompt fixe, résultats sauvegardés.
-Amélioration mesurée : 20 %
Comparaison baseline vs amélioration, justification des choix, gain mesurable ou compromis explicite.
-Application web : 15 %
Upload, affichage, analyse, score, justification, avertissement, logs.
-Évaluation et analyse d’erreurs : 20 %
Métriques, matrice de confusion, 20 à 30 cas commentés, faux positifs, faux négatifs, incertitudes, hallucinations.
-Sécurité, éthique et limites : 10 %
Pas de diagnostic, données non identifiantes, warning, validation humaine, prudence.
-Qualité de la soutenance : 5 %
Clarté du discours, preuves, démonstration stable, capacité à expliquer les limites.
Bonus : LoRA léger, ablation systématique de prompts, dashboard avancé, localisation visuelle.

Références:
Références recommandées à indiquer dans le formulaire ou dans le README du dépôt :
-Github : Assistant radiologue virtuel.
-Hugging Face : Image-text-to-text / Vision-Language Models.
-Google : MedGemma / MedGemma model card.
-Unsloth : Gemma 4 Fine-tuning Guide.
-Unsloth : Model Catalog.
-RSNA : Pneumonia Detection Challenge.
-Stanford AIMI : CheXpert Chest X-rays.
-PhysioNet : MIMIC-CXR.
-Gradio documentation.
-FastAPI documentation.
Le dépôt GitHub devra aussi citer explicitement les conditions d’usage des modèles et datasets. Les ressources Unsloth sont particulièrement utiles pour accélérer les expérimentations Gemma 4, tandis que MedGemma doit être traité comme base médicale à valider rigoureusement et non comme garantie de performance clinique.

Impact positif du projet:
Le projet peut avoir un impact pédagogique et social positif en formant les étudiants à une IA médicale responsable. Il ne vise pas à remplacer les professionnels de santé, mais à apprendre comment construire, mesurer et limiter un système d’aide à l’analyse d’image dans un domaine sensible.
L’impact principal est éducatif : les étudiants apprendront à combiner IA multimodale, développement web, évaluation expérimentale, sécurité des données, gestion de l’incertitude et éthique. Ils seront sensibilisés au fait qu’une réponse médicale fluide peut être fausse, et qu’un système fiable doit être jugé par ses preuves, ses erreurs et ses limites.
Le projet peut aussi favoriser une meilleure compréhension des technologies d’aide au diagnostic, sans confusion avec un dispositif médical. Il encourage une culture de prudence : données dé-identifiées, avertissement obligatoire, validation humaine, analyse des faux négatifs et documentation des hallucinations.
L’impact environnemental sera maîtrisé en privilégiant d’abord le prompting, les petits sous-ensembles de données, les modèles légers, les expérimentations LoRA et la limitation des entraînements coûteux. Le fine-tuning complet ne sera envisagé qu’en bonus.

Commenatires:
Cet appel d’offre est volontairement conçu comme un projet d’ingénierie responsable. Le cœur du sujet n’est pas de produire le modèle le plus impressionnant, mais de construire un système défendable : périmètre limité, baseline claire, amélioration mesurée, sortie structurée, garde-fous, logs, métriques et analyse d’erreurs.
Le projet est adapté à des étudiants ingénieurs car il combine plusieurs compétences : IA multimodale, prompt engineering, data engineering, backend, interface web, évaluation, documentation scientifique et réflexion éthique. Il permet également d’introduire des outils actuels comme MedGemma, Gemma 4, Unsloth, Gradio/FastAPI et SQLite, sans imposer un niveau de recherche clinique.
La réussite attendue n’est pas une démonstration parfaite. Une très bonne proposition devra montrer ce que le système sait faire, ce qu’il ne sait pas garantir, et comment les étudiants ont pris leurs décisions. L’analyse d’erreurs sera donc considérée comme un livrable central, pas comme une annexe.

https://github.com/BTajini/assistant-radiologue-virtuel
