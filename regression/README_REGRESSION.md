Comment utiliser le code pour entraîner les modèles qu'on a utilisé et générer des prédictions:
Au début, il faut exécuter le fichier train.py, cela va entraîner les trois modèles de base qu'on a utilisé: catboostregressor, LGBMregressor et un kernel.
Notre modèle final est un LGBMregressor qui prend comme inputs les prédictions des trois modèles de base. Pour entraîner ce modèle il faut exécuter le fichier train_stack.py.
Finalement, pour générer les prédictions finales il faut exécuter le fichier predict_Stack.py

Vous voyez qu'il y a un fichier nommé cache_functions.py, cela c'est parce que on garde les modèles qu'on entraîne en cache.

Au début du fichier config.py on doit écrire où on a les fichiers de data du data challenge.
