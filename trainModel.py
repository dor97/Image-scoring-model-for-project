# import cv2

# image1 = cv2.imread("image1.jpg")
# image2 = cv2.imread("image1.jpg")

# # Convert images to grayscale (if they are not already)
# gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# # Initialize SIFT detector
# sift = cv2.SIFT_create()

# # Find keypoints and descriptors in both images
# keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
# keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# # Initialize Brute-Force matcher
# bf = cv2.BFMatcher()

# # Match descriptors
# matches = bf.match(descriptors1, descriptors2)

# # Sort matches by distance
# matches = sorted(matches, key = lambda x:x.distance)

# # Draw matches
# matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# # Display matched image
# cv2.imshow('Matched Image', matched_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import os

# Function to calculate hair density in an image after background removal
# def calculate_hair_density(image):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Show the image
#     cv2.imshow("Image", gray)
#     cv2.waitKey(0)
    
#     # Apply thresholding to remove background
#     _, thresholded = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # Adjust threshold as needed
    
#     # Apply edge detection to identify hair regions
#     edges = cv2.Canny(thresholded, 50, 150)

#     # Show the image
#     cv2.imshow("Image", edges)
#     cv2.waitKey(0)
    
#     # Calculate hair density by counting non-zero pixels
#     hair_density = np.count_nonzero(edges) / (image.shape[0] * image.shape[1])
    
#     return hair_density

# # Function to rate images based on hair density
# def rate_images(image_folder):
#     # List to store image filenames and their corresponding hair densities
#     image_ratings = []
    
#     # Iterate through images in the folder
#     for filename in os.listdir(image_folder):
#         # Load the image
#         image = cv2.imread(os.path.join(image_folder, filename))
        
#         # Calculate hair density
#         hair_density = calculate_hair_density(image)
        
#         # Append image filename and hair density to the list
#         image_ratings.append((filename, hair_density))
    
#     # Sort images based on hair density
#     image_ratings.sort(key=lambda x: x[1], reverse=True)
    
#     # Print the sorted image ratings
#     for filename, density in image_ratings:
#         print(f"{filename}: Hair Density = {density:.2f}")


# # Function to calculate hair density in an image using contour detection
# def calculate_hair_density(image):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Show the image
#     cv2.imshow("Image", gray)
#     cv2.waitKey(0)
    
#     # Perform edge detection using Canny
#     edges = cv2.Canny(gray, 20, 150)

#     # Show the image
#     cv2.imshow("Image", edges)
#     cv2.waitKey(0)
    
#     # Find contours in the edge-detected image
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Calculate the total length of contours (hair strands)
#     total_contour_length = sum([cv2.arcLength(contour, True) for contour in contours])
    
#     # Calculate hair density as the ratio of total contour length to image area
#     hair_density = total_contour_length / (image.shape[0] * image.shape[1])
    
#     return hair_density

# # Function to rate images based on hair density
# def rate_images(image_folder):
#     # List to store image filenames and their corresponding hair densities
#     image_ratings = []
    
#     # Iterate through images in the folder
#     for filename in os.listdir(image_folder):
#         # Load the image
#         image = cv2.imread(os.path.join(image_folder, filename))
        
#         # Calculate hair density
#         hair_density = calculate_hair_density(image)
        
#         # Append image filename and hair density to the list
#         image_ratings.append((filename, hair_density))
    
#     # Sort images based on hair density
#     image_ratings.sort(key=lambda x: x[1], reverse=True)
    
#     # Print the sorted image ratings
#     for filename, density in image_ratings:
#         print(f"{filename}: Hair Density = {density:.2f}")


# # Example usage
# image_folder = "images/"
# rate_images(image_folder)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
import copy
from Module import HairDensityModel
import numpy as np
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

torch.save(pretrained_model.state_dict(), "model2.pth")
torch.save(baies_tosave, "baies2.pth")
