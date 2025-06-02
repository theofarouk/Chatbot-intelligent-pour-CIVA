# Chatbot-intelligent-pour-CIVA
Construire un chatbot intelligent alimenté par GraphRAG pour interroger un corpus documentaire hétérogène tout en maintenant la cohérence et le contexte des réponses. Le produit sera à expérimenter sur un ensemble de document de la Communauté d'Intérêt des Véhicules Autonomes (CIVA)


# Installation :

## 1) créer un environnement (en python 3.11)

```
name=chatbot
version="python=3.11"
# conda remove --name chatbot --all
conda create -n $name $version
```

## 2) Charger les librairies (sur le datalab)

```
# Pensez à se positionner avant sur le repertoire du repository
# Exemple : 
conda activate chatbot
pip install -r requirements.txt --retries 20