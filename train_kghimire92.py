# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from pathlib import Path
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% ----------------------------------- Hyper Parameters and Constants ------------------------------------------------
DATA_DIR = (str(Path(__file__).parents[1]) + '/train/')
IMG_WIDTH = 1600
IMG_HEIGHT = 1200
NUM_CLASSES = 7
# LR = 5e-2
N_EPOCHS = 25
BATCH_SIZE = 32
DROPOUT = 0.5

# %% ----------------------------------- MalariaDataset Class ----------------------------------------------------------
class MalariaDataset(Dataset):
    """init() : initial processes like reading a csv file, assigning transforms, ..."""

    def __init__(self, data_dir, output_classes, img_width, img_height, transform=None):
        self.data_dir = data_dir  # path to the data directory
        self.transform = transform  # a transform object
        self.img_width = img_width
        self.img_height = img_height

        file_ids = MalariaDataset.get_file_ids(data_dir)
        file_labels_list = MalariaDataset.map_file_to_labels_list(data_dir, file_ids)
        file_encoded_targets_list = MalariaDataset.map_file_to_encoded_targets(file_labels_list, output_classes)
        self.data = file_encoded_targets_list  # list of (image_name, target) tuples

    """len() : return the size of input data"""

    def __len__(self):
        return len(self.data)

    """ getitem() : return data and label at arbitrary index"""

    def __getitem__(self, index):
        # img_name, img_label = self.data[index][0:]
        img_name, img_label = self.data[index]
        img_file_name = img_name + '.png'
        img_path = os.path.join(self.data_dir, img_file_name)
        pil_img = Image.open(img_path)
        resized_img = pil_img.resize((self.img_width, self.img_height))

        # Transform image
        if self.transform is not None:
            image = self.transform(resized_img)
        label = torch.from_numpy(img_label)

        return image, label

    @staticmethod
    def get_file_ids(data_dir):
        image_files = list()
        text_files = list()
        json_files = list()
        file_ids = list()

        for file in os.listdir(data_dir):
            if file.endswith('.png'):
                f_name = file.split('.')
                image_files.append(f_name[0])
            if file.endswith('.txt'):
                f_name = file.split('.')
                text_files.append(f_name[0])
            if file.endswith('.json'):
                f_name = file.split('.')
                json_files.append(f_name[0])

        for file in image_files:
            if file in text_files and file in json_files:
                file_ids.append(file)
        return file_ids

    @staticmethod
    def map_file_to_labels_list(data_dir, files_list):
        file_labels_list = list()

        for file in files_list:
            with open(data_dir + file + '.txt') as fileobj:
                content = []
                for row in fileobj:
                    content.append(row.rstrip('\n'))
                file_labels_tuple = (file, content)
                file_labels_list.append(file_labels_tuple)

        return file_labels_list

    @staticmethod
    def map_file_to_encoded_targets(file_labels_list, classes_list):
        file_encoded_targets_list = list()

        for item in file_labels_list:
            target = []
            labels_list = item[1]
            file_id = item[0]

            for malaria_class in classes_list:
                if malaria_class in labels_list:
                    target.append(1)
                else:
                    target.append(0)
            target_numpy = np.asarray(target)
            file_target_tuple = (file_id, target_numpy)
            file_encoded_targets_list.append(file_target_tuple)
        return file_encoded_targets_list


# %% -------------------------------------- CNN Class ------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, (5, 5), stride=5)  # output (n_examples, 16, 240, 320)
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2, 2))  # output (n_examples, 16, 120, 160)
        self.conv2 = nn.Conv2d(16, 32, (5, 5), stride=5)  # output (n_examples, 32, 24, 32)
        self.convnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((2, 2))  # output (n_examples, 32, 12, 16)
        self.linear1 = nn.Linear(32 * 12 * 16, 400)  # input will be flattened to (n_examples, 32 * 12 * 16)
        self.linear1_bn = nn.BatchNorm1d(400)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(400, NUM_CLASSES)
        self.act1 = torch.relu
        self.act2 = torch.sigmoid

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act1(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act1(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act1(self.linear1(x.view(len(x), -1)))))
        return self.act2(self.linear2(x))


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
malaria_classes = ["red blood cell", "difficult", "gametocyte", "trophozoite",
                   "ring", "schizont", "leukocyte"]
img_transforms = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(30),
    transforms.ToTensor()])

malariadata = MalariaDataset(data_dir=DATA_DIR, output_classes=malaria_classes, img_width=IMG_WIDTH,
                             img_height=IMG_HEIGHT, transform=img_transforms)
# print(malariadata)

train_size = round(0.85 * len(malariadata))
test_size = len(malariadata) - train_size
train_dataset, test_dataset = random_split(malariadata, [train_size, test_size])
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
# If you need to move a model to GPU via .cuda(), do so before constructing optimizers for it. Parameters of a
# model after .cuda() will be different objects with those before the call.

model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Training loop...")
for epoch in range(N_EPOCHS):

    loss_train = 0  # Initializes train loss which will be added up after going forward on each batch
    model.train()  # Activates Dropout and makes BatchNorm use the actual training data to compute the mean and std
    # (this is the default behaviour but will be changed later on the evaluation phase)

    for i, (x_train, y_train) in enumerate(train_data_loader):
        
        x_train, y_train = x_train.cuda(), y_train.cuda()
        y_train = y_train.float()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()  # updates weights
        loss_train += loss.item()

    print("Epoch {} | Train Loss {:.5f}".format(
        epoch, loss_train))
   
torch.save(model.state_dict(), "model_kghimire92.pt") 

print("Testing loop...")
loss_test = 0
model.load_state_dict(torch.load("model_kghimire92.pt"))
model.eval()  # Deactivates Dropout and makes BatchNorm use mean and std estimates computed during training
with torch.no_grad():  # The code inside will run without Autograd, which reduces memory usage, speeds up
    # computations and makes sure the model can't use the test data to learn
    for i, (x_test, y_test) in enumerate(train_data_loader):
        x_test, y_test = x_test.cuda(), y_test.cuda()
        y_test = y_test.float()
        y_test_pred = model(x_test)
        loss = criterion(y_test_pred, y_test)
        loss_test += loss.item()
print("Test Loss {:.5f}".format(loss_test))
