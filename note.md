# Note sur le problème de détection de grain de café

# A lancer

- Modèle avec les différentes classes
- Rajouter unsure et voir
- Filtre à la mano les mauvais labels


## TODO


- Regarder si c'est possible d'obtenir un count avec deux images d'un côté et de l'autre de la branche.
- Gérer les différentes classes dans le jeu de données
- Faire un jeu de test avec LabelImg, avoir un jeu de test qui permettent d'avoir des vrais scores:
    - Crop de la branche considérée
    - Un count (à la crowd counting, avec un point par cerise pour pouvoir vérifier plus facilement)


## Possible amélio

- Dans les données d'entrées, certaines images sont mal labélisées (on voit qu'il manque beaucoup de bbox, à voir à quel point ça impactent)
- Voir carrément pas alignés comme : 
voir "image louche"

- Rajouter des données d'autres jeux, centrés agri : Wheat Challenge, Mango Detection par exemple


# Communication sur ce qui a été fait 

Transformation et uniformisation des images 
261 / 438 ne sont pas en 1024 par 1024 
Les infos du XML correspondent pas toujours (width et height inversées, ou juste à zéro)
Gérer le jeu de données de grain de café, pipé pour que ça rentre dans YOLO (s'inspirer de ce que a fait Juan Miguel, mais pas sûr que ça prenne moins de temps)
Lancer un premier modèle juste avec les images en 1024, voir ce que ça donne
Splitter les images plus grosses en images 1024x1024
(Voir plus petit en 500x500)
Faire en sorte que : ça tourne sur collab ou éventuellement Google Cloud si trop lent
Avoir une idée du score avec juste un COCO finetuné sur directement le café, sans data augmentation
data augmentation, rotation, flips => Automatique dans Ultralytics, à la volée
- Les images sont énormes :
    - Option 1 : splitting et virer celles où il y a pas de label
    - Option 2 : padding and downsizing => resoud le problème des images de taille différentes et réduit considérablement la taille des images
                    mieux pour la batch size etc...



# Plus long terme 

- Avoir une mini-appli d'avoir le detecteur en embarqué, pour voir en temps réel 
- Avoir une vidéo

À discuter dans le rapport, faire des recherches dessus 

Segment anything de Meta, à voir, ça a l'air chanmé.
https://ai.facebook.com/research/publications/segment-anything/


# Image louche

J'ai viré :
1615379843027 # bbox mal placées
"1615401570714.jpg" #bbox mal placées


1615385610373 # manque beacoup de label
1615385635949 # Small but ok


# Premiers résultats

dans predict3/ images du modèle juste entraîné sur les 1024x1024,
ça a l'air de bien marcher. Si l'image est flou, c'est pas dingue, mais ça a l'air de bien marcher au global

Avec le modèle entrainé sur plus d'images (avec des images downsize en 640x480) ça a l'air mieux

image technique : 
2 (grain super sombre)
3 (giga floue)
19 (fond bleu)
20 (ombre, pas ouf non plus)
29 (un peu à l'ombre, en vrai, faudrait être meilleur là dessus)
30 (surexposé)
55 (sombre)
77 (super exposé et floue bizarre)
89 (grain rouge)
91 (super exposée)


# Call Romain 

Comment isoler le premier plan de manière automatique 
Crop à la main 

Dans l'application : Un cropping à la main 


# check image post process

1615562134510  # Il a annoté l'ombre du soleil ou des conneries comme ça
1615400791606 # Feuille ultra eloignée annotée ..
1615300838785 # Feuille

# Count chelou

1682086833877.jpg 36 count, pred : 75