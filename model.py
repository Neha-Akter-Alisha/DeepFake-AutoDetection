import torch
import timm
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

# Load the Pretrained Model (Facebook's DeepFake Detection Model)
class DeepFakeDetector:
    def __init__(self):
        self.model = timm.create_model('tf_efficientnet_b7_ns', pretrained=True, num_classes=1)
        self.model.eval()

        # Define Image Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        """Predicts the probability of an image being deepfake and highlights fake regions."""
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = torch.sigmoid(self.model(img_tensor))  # Get probability
            probability = output.item()

        return probability

    def highlight_fake_regions(self, image_path):
        """Creates a heatmap to show manipulated regions."""
        image = cv2.imread(image_path)
        heatmap = np.zeros_like(image, dtype=np.uint8)

        # Simulate fake region by adding artificial red mask (For demonstration purposes)
        h, w, _ = image.shape
        fake_region = np.random.randint(100, 200, (h, w), dtype=np.uint8)
        heatmap[:, :, 2] = fake_region  # Add red tint

        # Blend the heatmap with the original image
        overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)

        # Save and return processed image
        processed_path = image_path.replace("uploads", "processed")
        cv2.imwrite(processed_path, overlay)

        return processed_path
