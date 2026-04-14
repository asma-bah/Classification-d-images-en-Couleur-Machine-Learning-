import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import torch as th
import torch.optim as optim
from torch.nn import functional as F
from  tqdm import tqdm
from sklearn.decomposition import PCA


#chargement du data set
with open("dataset_images_train", 'rb') as fo:
    data = pickle.load(fo, encoding='bytes')

X_train, X_valid, y_train, y_valid = train_test_split(
    data["data"], data["target"], test_size=0.2, random_state=42
)

# normalisation
X_train_norm = X_train / 255.0
X_valid_norm = X_valid / 255.0

pca = PCA(n_components=500)
X_train = pca.fit_transform(X_train_norm)
X_valid = pca.transform(X_valid_norm)



##====================================================================================##"
#==================== Reseau de neurones avec des couches linéaires ============================ "


##====================================================================================#"


d = X_train.shape[1]
k = 10

print(d, k)


#prediction a partir des sorties du modele
def predict(f):
    return th.argmax(f, 1)

#calcul du taux d'erreur en comparant les y predits avec les y reels
def error_rate(y_pred, y):
    return ((y_pred != y).sum().float()) / y_pred.size()[0]


#modele de reseau de neurones avec des couches linéaires
class neural_network_classif(th.nn.Module):
    def __init__(self, d, k,h1,h2, h3):
        super(neural_network_classif, self).__init__()
        self.layer1 = th.nn.Linear(d,h1)
        self.layer2 = th.nn.Linear(h1,h2)
        self.layer3 = th.nn.Linear(h2, h3)
        self.layer4 = th.nn.Linear(h3, k)

        self.dropout = th.nn.Dropout(p=0.6)

        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.layer3.reset_parameters()
        self.layer4.reset_parameters()

    def forward(self, x):
        phi1 = F.relu(self.layer1(x))
        phi1 = self.dropout(phi1)
        phi2 = F.relu(self.layer2(phi1))
        phi2 = self.dropout(phi2)
        phi3 = F.relu(self.layer3(phi2))
        phi3 = self.dropout(phi3)
        return self.layer4(phi3)


device = "cpu"
nnet = neural_network_classif(d, k, 1024,512,256).to(device)

#conversion des données en tenseur pytorch
X_train = th.from_numpy(X_train).float().to(device)
y_train = th.from_numpy(y_train).long().to(device)

X_valid = th.from_numpy(X_valid).float().to(device)
y_valid = th.from_numpy(y_valid).long().to(device)

optimizer = optim.Adam(nnet.parameters(), lr=0.001)

criterion = th.nn.CrossEntropyLoss()

#
train_errors = []
valid_errors = []

nb_epoch = 500
pbar = tqdm(range(nb_epoch))


for epoch in pbar:
    nnet.train()

    optimizer.zero_grad()
    f_train = nnet(X_train)
    loss = criterion(f_train, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        nnet.eval()  # ← AJOUTER pour désactiver le dropout
        with th.no_grad():  # ← AJOUTER pour ne pas calculer les gradients
            y_pred_train = predict(f_train)
            error_train = error_rate(y_pred_train, y_train)

            f_valid = nnet(X_valid)
            error_valid = error_rate(predict(f_valid), y_valid)

            train_errors.append(error_train.item())
            valid_errors.append(error_valid.item())

            pbar.set_postfix(iter=epoch,loss=loss.item(),error_train=error_train.item(), error_valid=error_valid.item())
        nnet.train()  # ← remettre en mode train après évaluation


# Courbe d'apprentissage pour visualiser
plt.figure()
plt.plot(train_errors, label='Train')
plt.plot(valid_errors, label='Validation')
plt.xlabel("Epochs (x100)")
plt.ylabel("Erreur")
plt.title("Courbe d'apprentissage MLP")
plt.legend()
plt.show()

with open("data_images_test", 'rb') as fo:
    data_test = pickle.load(fo, encoding='bytes')

X_test = data_test["data"] / 255.0
X_test = pca.transform(X_test)
X_test_th = th.from_numpy(X_test).float().to(device)

nnet.eval()
with th.no_grad():
    predictions = predict(nnet(X_test_th)).numpy()

np.savetxt("images_test_predictions.csv", predictions)
print("Prédictions sauvegardées !")

#Score 0.5054 en 7min avec 500 epochs dropout à 0.6
# et 0.49 avec 5000 epoch et dropout à 0.5 en 1h27min du coup j'ai dû revenir en arriere