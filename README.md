## Objectifs du projet
Le projet ici est de récupérer un modèle de réseau de neurones de convolution de segmentation. On va prendre une image en input, extraire les features les plus importantes de l'image et la reconstruire avec des pixels qui vont correspondre à différentes classe (Principe de l'auto encodeur avec de la classification). Une fois cela fait on va envoyer ce serveur sur une instance ec2, et le but est d'avoir un client web pour communiquer et utiliser le modèle.

Avant de faire cela, on doit entrainer ce modèle sur la métrique de notre choix (IoU étant la métrique la plus pertinente et la plus punitif mais la plus efficace). Et envoyer nos métrique directement dans un bucket S3. 

J'ai donc télécharger le modèle directement sur le Pytorch https://github.com/pytorch/vision/blob/main/torchvision/models/detection/mask_rcnn.py Que j'ai analyser dans un premier temps avec un fichier Jupyter ipynb et ensuite entrainer. J'ai aussi créer un fichier qui va prendre en argument en CLI le path d'une image et qui va donnée en sortie l'image segmenté avec reconnaissance d'image.
Pour entrainer ce modèle pré-entrainé et ensuite évaluer mon modèle, j'ai décidé d'utiliser la dataset : **oxford-pet** qui contient des photos d'animaux et de faire mon entrainement sur toutes les métriques vu en cours et voici les résultats obtenus: 

```
Métriques: 
- Test Phase Date du test: 2024-04-02 Métriques globales: 
	- IoU Global: 0.75 
	- Précision Globale: 0.85 
	- Rappel Global: 0.87
	- F1-score Global: 0.85 
- Métriques par classe: Classe 1: Chat 
	- IoU: 0.78 
	- Précision: 0.88 
	- Rappel: 0.90 
	- F1-score: 0.89 
- Classe 2: Chien 
	- IoU: 0.72 
	- Précision: 0.82 
	- Rappel: 0.85 
	- F1-score: 0.83 
```

Une fois cela fait j'envoyé ces métriques directement sur un bucket s3.
![[Pasted image 20240402102835.png]]

