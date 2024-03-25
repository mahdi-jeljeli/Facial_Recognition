import numpy as np
from keras.models import load_model
from PIL import Image

def load_image(filename):
    # Load the image in color
    img = Image.open(filename)
    # Convert to grayscale
    img = img.convert('L')
    # Resize the image to 28x28
    img = img.resize((28, 28))
    # Convert to array
    img_array = np.array(img)
    # Reshape into a single sample with 1 channel
    img_array = img_array.reshape(1, 28, 28, 1)
    # Prepare pixel data
    img_array = img_array.astype('float32')
    img_array = img_array / 255.0
    return img_array

# Chargez le modèle après l'appel à la fonction final()
model = load_model('E:\M1_ISIDS\Semestre_2\Porgrammation python av\projet reconnaissance faciale\my_model.keras')

# Créer un dictionnaire associant les classes prédites à des noms ou des identifiants de personnes
classes_dict = {
    0: "Personne 1",
    1: "Personne 2",
    2: "Personne 3",
    3: "Personne 4",
    4: "Personne 5",
    5: "Personne 6",
    6: "Personne 7",
    7: "Personne 8",
    8: "Personne 9",
    9: "Personne 10",
    10: "Personne 11",
    11: "Personne 12",
    12: "Personne 13",
    13: "Personne 14",
    14: "Personne 15",
    15: "Personne 16",
    16: "Personne 17",
    17: "Personne 18",
    18: "Personne 19",
    19: "Personne 20",
    20: "Personne 21",
    21: "Personne 22",
    22: "Personne 23",
    23: "Personne 24",
    24: "Personne 25",
    25: "Personne 26",
    26: "Personne 27",
    27: "Personne 28",
    28: "Personne 29",
    29: "Personne 30",
    30: "Personne 31",
    31: "Personne 32",
    32: "Personne 33",
    33: "Personne 34",
    34: "Personne 35",
    35: "Personne 36",
    36: "Personne 37",
    37: "Personne 38",
    38: "Personne 39",
    39: "Personne 40",
}
# Faites une prédiction avec votre modèle CNN sur une image spécifique
img = load_image('E:\\data\\35\\1.jpg')
# Faire une prédiction avec votre modèle CNN
predictions = model.predict(img)

# Obtenir l'indice de la classe avec la probabilité la plus élevée
predicted_class_index = np.argmax(predictions)

# Afficher le nom ou l'identifiant correspondant à la classe prédite
if predicted_class_index in classes_dict:
    print("La personne prédite est :", classes_dict[predicted_class_index])
else:
    print("Classe prédite non reconnue")