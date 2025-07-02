Classification de la maturité des fraise (Strawberry Ripeness classification)

Le projet vise à améliorer la production industrielle des fraises.
On cherche à classer les fraises en quatre groupes de maturation : non-mûre, semi-mûre, presque mûre et mûre.

La première méthode dans le dossier `Clustering` se base sur le clustering K-means dans différents espaces de couleurs : RGB, HSV et Lab. On classifie alors soit en utilisant la moyenne des couleurs soit l'histogramme 2d sur les canaux appropriés.

La deuxième méthode dans le dossier `CNN` utilise des réseaux convolutifs pour la classification. Plus précisément un fine-tuning est effectué sur resnet18 préentrainé pour correspondre à notre tâche.

Les évaluations et les résultats peuvent être observés dans les différents notebooks. 
















