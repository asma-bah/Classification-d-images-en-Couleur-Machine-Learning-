import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

#chargement du data set
with open("dataset_images_train", 'rb') as fo:
    data = pickle.load(fo, encoding='bytes')

X_train, X_valid, y_train, y_valid = train_test_split(
    data["data"], data["target"], test_size=0.2, random_state=42
)

############################### Visualisations des données ########################

#afficher la premiere image de la base d'entrainement
image = data["data"][0]
image = image.reshape(3,32,32).transpose(1,2,0)
plt.figure()
plt.imshow(image.astype("uint8"))
#plt.show()

#Q2
def print_image_classe(k):
    nb = 0

    for i in range(len(data["data"])):
        if data["target"][i] == k:
            plt.figure()
            img = data["data"][i].reshape(3, 32, 32).transpose(1, 2, 0)
            plt.imshow(img.astype("uint8"))
            plt.title(f"classe {k}")
            plt.show()
            nb += 1
        if nb == 10:
            break


#print_image_classe(4)

#Q3 Methode TSNE pour une projection
plt.figure()

X_subs = X_train[:3000]
y_subs = y_train[:3000]

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_subs)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_subs)
plt.colorbar()
plt.show()

#Constat :
########## la projection montre que les classes ne sont pas directement separé, elles sont toutes melangé

###########################################################################################
#######################"    Algorithme des k plus proches voisins   ########################"

def square_euclidian_distance(v1, v2):
    return np.sum((v1 - v2) ** 2, axis=1)

#selectionner le voisin
def select_neighborhoods(X_train, y_train, x_valid, k):
    distances = square_euclidian_distance(X_train, x_valid)  #calcul de toutes les distances
    k_min_index = np.argsort(distances)[:k]
    return y_train[k_min_index]


#fonction de prediction
def predict_knn(X_train, y_train, x_valid, k):
    neighbors = select_neighborhoods(X_train, y_train, x_valid, k)
    values, counts = np.unique(neighbors, return_counts=True)
    return values[np.argmax(counts)]  # classe la plus fréquente

#fonction d'evaluation
def evaluate_knn(X_train,  y_train, X_valid, y_valid, k):
    predictions = []

    for i in range(X_valid.shape[0]):
        pred = predict_knn(X_train, y_train, X_valid[i], k)
        predictions.append(pred)

    accuracy = np.mean(np.array(predictions) == y_valid)
    return accuracy


#acc = evaluate_knn(X_train[:5000], y_train[:5000], X_valid[:200], y_valid[:200], k=5)
#print(f"Accuracy : {acc:.4f}")
print(X_train.dtype)   # doit être float ou int
print(y_train.dtype)   # doit être int
print(np.unique(y_train))  # doit afficher [0 1 2 3 4 5 6 7 8 9]
print(X_train.min(), X_train.max())  # doit être entre 0 et 255


k_values = list(range(1,21,2))

# Normalisation des données
X_train_norm = X_train / 255.0
X_valid_norm = X_valid / 255.0

#  PCA (permet de reduire le nombre de dimensions et le temps de calcul)
pca = PCA(n_components=50)
X_train = pca.fit_transform(X_train_norm)
X_valid = pca.transform(X_valid_norm)

list_accuracies = []
for k in k_values:
    acc = evaluate_knn(X_train, y_train, X_valid, y_valid, k)
    list_accuracies.append(acc)
    #print(f"k={k} -> Accuracy: {acc}")

# tracer de la courbe
plt.figure()
plt.plot(k_values, list_accuracies)
plt.xlabel("Nombre de voisins k")
plt.ylabel("Precision")
plt.title("Precision du KNN en fonction de k")
plt.show()

best_k = k_values[np.argmax(list_accuracies)] #choix du k plus proche voisins
print(f"Best k={best_k} -> Accuracy: {max(list_accuracies)}")

# Charger le données de test
with open("data_images_test", 'rb') as fo:
    data_test = pickle.load(fo, encoding='bytes')


X_test = data_test["data"]
X_test = pca.transform(X_test / 255.0) #normalisation et pca

predictions = []
for i in range(X_test.shape[0]):
    predict = predict_knn(X_train, y_train, X_test[i], best_k)
    predictions.append(predict)

    if i % 100 == 0:
        print(f"Progression :  {i} / {len(X_test)}")

np.savetxt("images_test_predictions.csv", predictions)
print("Predictions sauvegardées ! ")

#premier score 0.2138, deuxieme score apres ajout de pca et normalisation le score est de 0.3568
