# %% --------------------------------------- Imports -------------------------------------------------------------------
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% ----------------------------------- Hyper Parameters and Constants ------------------------------------------------
DROPOUT = 0.5
NUM_CLASSES = 7
IMG_WIDTH = 1600
IMG_HEIGHT = 1200
# BATCH_SIZE = 3
BATCH_SIZE = 21


# %% ----------------------------------- MalariaDataset Class ----------------------------------------------------------
class MalariaDataset(Dataset):
    """init() : initial processes like reading a csv file, assigning transforms, ..."""

    def __init__(self, data_path_list, img_width, img_height, transform=None):
        self.data_path_list = data_path_list  # list of data paths
        self.transform = transform  # a transform object
        self.img_width = img_width
        self.img_height = img_height
        self.data = data_path_list

    """len() : return the size of input data"""

    def __len__(self):
        return len(self.data)

    """ getitem() : return data and label at arbitrary index"""

    def __getitem__(self, index):
        img_path = self.data[index]
        pil_img = Image.open(img_path)
        resized_img = pil_img.resize((self.img_width, self.img_height))
        # Transform image
        if self.transform is not None:
            image = self.transform(resized_img)
        return image


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


# %% -------------------------------------- Predict Function -----------------------------------------------------------

def predict(data_path_list):
    # Data Prep
    img_transforms = transforms.Compose([transforms.ToTensor()])
    data = MalariaDataset(data_path_list, img_width=IMG_WIDTH,
                          img_height=IMG_HEIGHT, transform=img_transforms)
    # print(len(data))
    data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)

    tensors_list = list()

    # Model
    model = CNN().cuda()
    model.load_state_dict(torch.load("model_kghimire92.pt"))
    model.eval()
    with torch.no_grad():
        for i, (x) in enumerate(data_loader):
            x = x.cuda()
            y_pred = model(x)
            tensors_list.append(y_pred)
    y_pred_tensor = torch.cat(tensors_list).round()
    return y_pred_tensor.cpu()
