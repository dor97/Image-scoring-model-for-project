import torch
from torchvision import models
from Module import HairDensityModel

from PIL import Image

class trained_model:
    def __init__(self) -> None:
        self.pretrained_model = models.resnet50()
        self.pretrained_model.fc = HairDensityModel()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.pretrained_model.load_state_dict(torch.load("model.pth", map_location=device))

        baies = torch.load("baies.pth")
        self.baies_minus = baies["baies_minus"]
        self.baies_div = baies["baies_div"]

        # Set the model to evaluation mode
        self.pretrained_model.eval()

    # Function to predict hair density score for a single image
    def predict_hair_density_image(self, image, model, transform) -> float:
        # Load and transform the image
        image = Image.open(image).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = model(image)

        # Extract predicted hair density score
        predicted_density = output.item()

        return (predicted_density - self.baies_minus) / self.baies_div

    # Function to predict hair density score for a single image
    def predict_hair_density(self, image_path, model, transform) -> float:
        # Load and transform the image
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = model(image)

        # Extract predicted hair density score
        predicted_density = output.item()

        return (predicted_density - self.baies_minus) / self.baies_div
    
    def test_one_image(self, image) -> float:
        # Test the model on new image
        predicted_density = self.predict_hair_density_image(image, self.pretrained_model, self.pretrained_model.fc.transform)
        print(f"Predicted Density: {predicted_density:.4f}")

        return predicted_density

    
    def test_one(self, image_path: str) -> float:
        # Test the model on new image
        predicted_density = self.predict_hair_density(image_path, self.pretrained_model, self.pretrained_model.fc.transform)
        print(f"Image: {image_path}, Predicted Density: {predicted_density:.4f}")

        return predicted_density

    def test(self, image_paths: list[str]) -> list[float]:
        # Test the model on new images
        result = []

        for image_path in image_paths:
            predicted_density = self.predict_hair_density(image_path, self.pretrained_model, self.pretrained_model.fc.transform)
            print(f"Image: {image_path}, Predicted Density: {predicted_density:.4f}")
            result.append(predicted_density)

        return result


#model = trained_model()
#mage_paths = ['test/image1.jpg', 'test/image2.jpg', 'test/image3.jpg', 'test/image7.webp', 'test/image8.jpeg', 'test/image10.jpeg']  # List of paths to test images

#model.test(image_paths)


# pretrained_model = models.resnet50()
# # Replace the final fully connected layer with a custom one for hair density estimation
# pretrained_model.fc = HairDensityModel()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# pretrained_model.load_state_dict(torch.load("model.pth", map_location=device))

# # Set the model to evaluation mode
# pretrained_model.eval()

# # Function to predict hair density score for a single image
# def predict_hair_density(image_path, model, transform):
#     # Load and transform the image
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0)  # Add batch dimension

#     # Perform inference
#     with torch.no_grad():
#         output = model(image)

#     # Extract predicted hair density score
#     predicted_density = output.item()

#     return predicted_density

if __name__ == "__main__":
    # Test the model on new images
    pretrained_model = trained_model()
    image_paths = ['test/image1.jpg', 'test/image2.jpg', 'test/image3.jpg', 'test/image7.webp', 'test/image8.jpeg', 'test/image10.jpeg']  # List of paths to test images
    pretrained_model.test(image_paths)
    # image_paths = ['test/image1.jpg', 'test/image2.jpg', 'test/image3.jpg', 'test/image7.webp', 'test/image8.jpeg', 'test/image10.jpeg']  # List of paths to test images
    # for image_path in image_paths:
    #     predicted_density = pretrained_model.predict_hair_density(image_path, pretrained_model, pretrained_model.fc.transform)
    #     print(f"Image: {image_path}, Predicted Density: {predicted_density:.4f}")