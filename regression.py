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

# ✅ normaliser avant PCA
X_train_norm = X_train / 255.0
X_valid_norm = X_valid / 255.0

pca = PCA(n_components=200)
X_train = pca.fit_transform(X_train_norm)
X_valid = pca.transform(X_valid_norm)



##====================================================================================##"
#==================== Regression logistique multivariée ============================ "


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



#modele
class reg_log_multi(th.nn.Module):
    def __init__(self,d, k):
        super(reg_log_multi,self).__init__()
        self.layer = th.nn.Linear(d, k)

        self.layer.reset_parameters()

    def forward(self,x):
        return F.softmax(self.layer(x), 1)


device = "cpu"
model = reg_log_multi(d, k).to(device)

#conversion des données en tenseur pytorch
X_train = th.from_numpy(X_train).float().to(device)
y_train = th.from_numpy(y_train).long().to(device)

X_valid = th.from_numpy(X_valid).float().to(device)
y_valid = th.from_numpy(y_valid).long().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion = th.nn.CrossEntropyLoss()
# 3. Courbe d'apprentissage pour visualiser
train_errors = []
valid_errors = []

nb_epoch = 100000
pbar = tqdm(range(nb_epoch))


for epoch in pbar:
    optimizer.zero_grad()
    f_train = model(X_train)
    loss = criterion(f_train, y_train)
    loss.backward()
    optimizer.step()


    if epoch % 100 == 0:
        y_pred_train = predict(f_train)
        error_train = error_rate(y_pred_train, y_train)

        f_valid = model(X_valid)
        error_valid = error_rate(predict(f_valid), y_valid)

        train_errors.append(error_train.item())
        valid_errors.append(error_valid.item())

        pbar.set_postfix(iter=epoch,loss=loss.item(),error_train=error_train.item(),error_valid=error_valid.item())


# Courbe après la boucle
plt.figure()
plt.plot(train_errors, label='Train')
plt.plot(valid_errors, label='Validation')
plt.xlabel("Epochs (x100)")
plt.ylabel("Erreur")
plt.title("Courbe d'apprentissage")
plt.legend()
plt.show()

# Prédiction sur le jeu de test
with open("data_images_test", 'rb') as fo:
    data_test = pickle.load(fo, encoding='bytes')

X_test = data_test["data"] / 255.0
X_test = pca.transform(X_test)
X_test = th.from_numpy(X_test).float().to(device)

model.eval()

with th.no_grad():
    predictions = predict(model(X_test)).numpy()


np.savetxt("images_test_predictions.csv", predictions)
print("Prédictions sauvegardées !")

#Score 0.3856