import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Load ResNet18 (same as training)
model = models.resnet18(pretrained=False)

# Replace final fully connected layer with 10 output classes (same as training)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # 10 animal classes

# Move model to device
model = model.to(device)

# Load trained weights
state_dict = torch.load("animal9.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Image preprocessing (same normalization used during training)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

# Class labels for 10 classes (make sure this matches your train_data.classes order)
class_names = ["Dog", "Horse", "Elephant", "Butterfly", "Chicken", "Cat", "Cow", "Sheep", "Spider", "Squirrel"]


# Prediction function
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        class_name = class_names[predicted.item()]
    print(f"Prediction: {class_name}")
    return class_name

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py path_to_image")
        sys.exit(1)
    
    image_path = sys.argv[1]
    predict(image_path)

