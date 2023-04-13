# Note sur le problème de détection de grain de café

## TODO urgent

Lancer un premier modèle juste avec les images en 1024, voir ce que ça donne => En cours
Splitter les images plus grosses en images 1024x1024
(Voir plus petit en 500x500)


## TODO

- Gérer le jeu de données de grain de café, pipé pour que ça rentre dans YOLO (s'inspirer de ce que a fait Juan Miguel, mais pas sûr que ça prenne moins de temps)
- Faire en sorte que : ça tourne sur collab ou éventuellement Google Cloud si trop lent
- Avoir une idée du score avec juste un COCO finetuné sur directement le café, sans data augmentation
- Gérer les différentes classes dans le jeu de données


## Possible amélio

- Dans les données d'entrées, certaines images sont mal labélisées (on voit qu'il manque beaucoup de bbox, à voir à quel point ça impactent)
- Voir carrément pas alignés comme : 
voir "image louche"

- data augmentation, rotation, flips (attention au labels, à voir comment on gère ça, en amont ou à la volée ?)
- Rajouter des données d'autres jeux, centrés agri : Wheat Challenge, Mango Detection par exemple
- Les images sont énormes :
    - Option 1 : splitting et virer celles où il y a pas de label
    - Option 2 : padding and downsizing => resoud le problème des images de taille différentes et réduit considérablement la taille des images
                    mieux pour la batch size etc...


# Communication sur ce qui a été fait 

Transformation et uniformisation des images 
261 / 438 ne sont pas en 1024 par 1024 
Les infos du XML correspondent pas toujours (width et height inversées, ou juste à zéro)


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


1615385610373 # rotation facile à régler
1645385635949 # Small but ok
