import os
import numpy as np
from PrinipalCNNModel import load_dataset, evaluate_model, accuracy_summary, loss_summary

#Ce fichier contient le code principal pour l'entraînement d'un modèle de réseau de neurones
#convolutionnel (CNN) pour la classification d('images. Il inclut les fonctions pour charger '
#les données, prétraiter les images, définir l''')architecture du CNN,
# évaluer le modèle avec une validation croisée, et visualiser les résultats d('entraînement. De plus, '
# 'il recherche la meilleure valeur pour le nombre de plis dans la validation croisée.)

# Charger les données en utilisant la fonction load_dataset du fichier principal
folder_path = "E:\\data"
trainX, trainY, testX, testY = load_dataset(folder_path)

# Liste des valeurs de n_folds à tester
n_folds_list = [3, 5, 7, 10]

# Initialisation des scores
best_score = 0
best_n_folds = None

# Recherche de la meilleure valeur de n_folds
for n_folds in n_folds_list:
    scores, _ = evaluate_model(trainX, trainY, n_folds)
    avg_score = np.mean(scores)
    print(f"Average score for {n_folds}-fold cross-validation: {avg_score}")

    # Mise à jour du meilleur score et du meilleur n_folds
    if avg_score > best_score:
        best_score = avg_score
        best_n_folds = n_folds

# Entraînement final avec la meilleure valeur de n_folds
print(f"Best number of folds: {best_n_folds}")
final_scores, final_histories = evaluate_model(trainX, trainY, best_n_folds)
accuracy_summary(final_histories)
loss_summary(final_histories)
