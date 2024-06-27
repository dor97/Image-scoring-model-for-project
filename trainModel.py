import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from Module import HairDensityModel
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Read annotations from CSV file
        self.annotations = pd.read_csv(os.path.join(root_dir, 'annotations.csv'))
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'images', self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        hair_density = float(self.annotations.iloc[idx, 1])
        
        if self.transform:
            image = self.transform(image)
        
        return image, hair_density


# Function to train the model
def train_model(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")


if __name__ == "__main__":
    
    # # Define transforms for image preprocessing
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    # Load pre-trained ResNet model
    pretrained_model = models.resnet50(pretrained=True)

    # Replace the final fully connected layer with a custom one for hair density estimation
    model = HairDensityModel()
    pretrained_model.fc = model

    transform = model.transform

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pretrained_model.to(device)

    # Freeze all layers except the final layers for fine-tuning
    for param in pretrained_model.parameters():
        param.requires_grad = False
    for param in pretrained_model.fc.parameters():
        param.requires_grad = True

    # Load dataset and create data loader
    train_dataset = CustomDataset("dataSet/", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Initialize the optimizer and loss function
    optimizer = optim.Adam(pretrained_model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    # Train the model
    train_model(pretrained_model, train_loader, optimizer, criterion, epochs=15)


    # Set the model to evaluation mode
    pretrained_model.eval()

    # Function to predict hair density score for a single image
    def predict_hair_density(image_path, model, transform):
        # Load and transform the image
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = model(image)

        # Extract predicted hair density score
        predicted_density = output.item()

        return predicted_density

    results = []
    # Test the model on new images
    image_paths = ['test/image1.jpg', 'test/image2.jpg', 'test/image3.jpg', 'test/image7.webp', 'test/image8.jpeg', 'test/image10.jpeg']  # List of paths to test images
    for image_path in image_paths:
        predicted_density = predict_hair_density(image_path, pretrained_model, transform)
        print(f"Image: {image_path}, Predicted Density: {predicted_density:.4f}")
        results.append(predicted_density)


    def calc_vaies(results: list[float]) -> float:
        min_result = min(results)
        max_result = max(results)

        return min_result, max_result - min_result

    baies_minus, baies_div = calc_vaies(results)

    print(f"baies_minus: {baies_minus}, baies_div: {baies_div}")

    baies_tosave = {
        "baies_minus" : baies_minus,
        "baies_div" : baies_div
    }

    torch.save(pretrained_model.state_dict(), "model.pth")
    torch.save(baies_tosave, "baies.pth")
