import torch
from torchvision import transforms
from PIL import Image
from cnn import SimpleCNN  # Import your model class

# Load the trained model
model = SimpleCNN(num_classes=2)
model_path = 'saved_model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load and preprocess the input image
image_path = 'data/bread/2023-12-10 16:36:48.272170.png'
image_path = 'data/chicken/2023-12-10 16:34:06.128226.png'
input_image = Image.open(image_path).convert('RGB')
input_tensor = transform(input_image)
input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

# Check if a GPU is available and move the input tensor to the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_batch = input_batch.to(device)

# Make the prediction
with torch.no_grad():
    output = model(input_batch)

print(output)

# Get the predicted class
_, predicted_class = torch.max(output, 1)

# Map the predicted class index to the class label
class_index = predicted_class.item()
classes = ["bread", "chicken"]
class_label = classes[class_index]

print(f'The predicted class is: {class_label}')


# import torch
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
# from cnn import SimpleCNN  # Import your model class
# from PIL import Image
# 
# # Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 
# # Define transform for inference
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
# ])
# 
# # Load the trained model
# model_path = 'simple_cnn_model_2023-12-10 22:35:02.283354.pth'
# model = SimpleCNN(num_classes=2)  # Make sure to replace NUM_CLASSES
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.eval()
# model = model.to(device)
# 
# # Create a function to predict the class of an input image
# def predict_image(image_path):
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0).to(device)
#     
#     with torch.no_grad():
#         output = model(image)
#     
#     print(output)
#     # _, predicted_class = torch.max(output, 1)
#     predicted_class = torch.argmax(output, dim=1)
#     print(predicted_class)
#     return predicted_class.item()
# 
# # Example usage
# image_path = 'data/bread/2023-12-10 16:36:48.272170.png'
# predicted_class = predict_image(image_path)
# 
# print(f'The predicted class is: {predicted_class}')
# 
