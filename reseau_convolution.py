import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import torch as th
import torch.optim as optim
from torchvision import transforms
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Chargement du dataset
with open("dataset_images_train", 'rb') as fo:
    data = pickle.load(fo, encoding='bytes')

X_train, X_valid, y_train, y_valid = train_test_split(
    data["data"], data["target"], test_size=0.2, random_state=42
)

# Normalisation
X_train_norm = X_train / 255.0
X_valid_norm = X_valid / 255.0

##====================================================================================##
#==================== Dataset avec augmentation =====================================##
##====================================================================================##

class CIFARDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Reshape en image (32, 32, 3) pour PIL
        image = (self.X[idx] * 255).astype("uint8").reshape(3, 32, 32).transpose(1, 2, 0)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, self.y[idx]

# Augmentation pour le train
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# Pas d'augmentation pour la validation
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

train_dataset = CIFARDataset(X_train_norm, y_train, transform=train_transforms)
valid_dataset = CIFARDataset(X_valid_norm, y_valid, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

##====================================================================================##
#==================== Réseau de convolution =========================================##
##====================================================================================##

class reseau_convolution(th.nn.Module):
    def __init__(self):
        super(reseau_convolution, self).__init__()
        self.conv1 = th.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = th.nn.BatchNorm2d(32)

        self.conv2 = th.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = th.nn.BatchNorm2d(64)

        self.conv3 = th.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = th.nn.BatchNorm2d(128)

        self.pool = th.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = th.nn.Dropout(p=0.4)

        self.fc1 = th.nn.Linear(128 * 4 * 4, 256)
        self.fc2 = th.nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32→16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16→8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8→4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

device = "cpu"
cnn = reseau_convolution().to(device)

criterion = th.nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

# Scheduler — réduit le lr automatiquement
scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

nb_epochs = 15
pbar = tqdm(range(nb_epochs))

train_losses = []
valid_accuracies = []
best_acc = 0

# Boucle d'entraînement
for epoch in pbar:
    cnn.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()  # mise à jour du lr
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Evaluation
    cnn.eval()
    correct = 0
    total = 0
    with th.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = cnn(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    valid_accuracies.append(acc)

    if best_acc < acc:
        best_acc = acc
        th.save(cnn.state_dict(), "best_model.pth")

    pbar.set_postfix(epoch=epoch, loss=avg_loss, acc=acc, best=best_acc)

# Courbes
plt.figure()
plt.plot(train_losses)
plt.title("Loss d'entraînement")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.figure()
plt.plot(valid_accuracies)
plt.title("Précision validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# Prédiction sur le jeu de test
with open("data_images_test", 'rb') as fo:
    data_test = pickle.load(fo, encoding='bytes')

X_test_norm = data_test["data"] / 255.0

class CIFARTestDataset(Dataset):
    def __init__(self, X, transform=None):
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = (self.X[idx] * 255).astype("uint8").reshape(3, 32, 32).transpose(1, 2, 0)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image

test_dataset = CIFARTestDataset(X_test_norm, transform=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Charger le meilleur modèle
cnn.load_state_dict(th.load("best_model.pth"))
cnn.eval()

predictions = []
with th.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = cnn(images)
        preds = outputs.argmax(dim=1)
        predictions.extend(preds.cpu().numpy())

np.savetxt("images_test_predictions.csv", predictions)
print(f"Prédictions sauvegardées ! Meilleur accuracy : {best_acc:.4f}")

#score 0.6394 avec 15 epochs et 0.01 loss, temps d'apprentissage 2min avant ajout des BatchNorm
#score 0.7208 avec 25 epochs et 0.001 loss, temps d'apprentissage 6min