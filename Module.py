import torch.nn as nn
from torchvision import transforms

# Define a custom fully connected network for hair density estimation
class HairDensityModel(nn.Module):
    def __init__(self):
        super(HairDensityModel, self).__init__()
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 1)

        # Define transforms for image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        #x = torch.sigmoid(self.fc1(x))

        #x = torch.sigmoid(self.fc2(x))
        x = self.fc2(x)
        return x