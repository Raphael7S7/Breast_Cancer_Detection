# Breast_Cancer_Detection
Cette analyse se base sur les données de detection du cancer du sein et provienne de l'université du Wisconsin.
(https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)

Elles comportent une serie de mesure effectuée sur 569 radiographies mamaires labélisées.
L'objectif est de construire un modèle capable de reconnaitre le type de cancer sur une radio non labélisée.

Le notebook comporte une analyse détaillé des données avec l'utilisation de tests statistique pour 
déterminer quelle variable influence le plus le type de cancer. On trouve également une comparaison des différents
modèles populaires en classification avec amélioration de leurs paramètres dans le but de bien classer les observations.

Enfin une application d'aide à la décision est disponible au format streamlit. Pour la déployer veuillez rentrer
les commandes suivantes dans votre terminal : 

            pip install streamlit
            streamlit run https://github.com/Raphael7S7/Breast_Cancer_Detection/blob/main/Cancer_Sein.py
            
            (Vous avez besoin de la dernière version de python pour lire l'application pip3 --version puis entrer)
            
  ![alt text](https://github.com/Raphael7S7/Breast_Cancer_Detection/blob/main/Screenshot%202021-01-13%20at%2016.23.04.png)          
