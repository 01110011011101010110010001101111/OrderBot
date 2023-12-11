import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

def test(model, test_loader, criterion, device):
    model.eval()
    all_labels = []
    all_preds = []
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return running_loss / len(test_loader), accuracy

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset = datasets.ImageFolder(root='data', transform=transform)
    print(dataset.classes)

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Create the model
    num_classes = len(dataset.classes)
    model = SimpleCNN(num_classes=num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Training Loss: {train_loss:.4f}, '
              f'Test Loss: {test_loss:.4f}, '
              f'Test Accuracy: {100 * test_accuracy:.2f}%')

    # Save the trained model
    from datetime import datetime
    torch.save(model.state_dict(), f'models/saved_model_{datetime.now()}.pth')

    # # Test a sample image
    # model.eval()
    # sample_image_path = 'data/chicken/2023-12-10 15:37:44.961748.png'  # Replace with the path to your sample image
    # sample_image = Image.open(sample_image_path).convert('RGB')
    # sample_input = transform(sample_image).unsqueeze(0).to(device)

    # with torch.no_grad():
    #     output = model(sample_input)

    # _, predicted_class = torch.max(output, 1)
    # class_index = predicted_class.item()
    # class_label = dataset.classes[class_index]

    # print(f'The predicted class for the sample image is: {class_label}')


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms, datasets
# from sklearn.model_selection import train_test_split
# import os
# from PIL import Image
# 
# # Define the CNN architecture
# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
# 
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
# 
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.relu3 = nn.ReLU()
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
# 
#         self.fc1 = nn.Linear(64 * 8 * 8, num_classes)
# 
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)
# 
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.pool2(x)
# 
#         x = self.conv3(x)
#         x = self.relu3(x)
#         x = self.pool3(x)
# 
#         x = x.view(-1, 64 * 8 * 8)  # Flatten before fully connected layer
#         x = self.fc1(x)
# 
#         return x
# 
# # Define a custom dataset class
# class CustomDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.classes = os.listdir(root_dir)
#         self.class_to_index = {cls: i for i, cls in enumerate(self.classes)}
#         self.images = self.get_images()
# 
#     def get_images(self):
#         all_images = []
#         for i, cls in enumerate(self.classes):
#             class_path = os.path.join(self.root_dir, cls)
#             class_images = [(os.path.join(class_path, img), i) for img in os.listdir(class_path)]
#             all_images.extend(class_images)
#         return all_images
# 
#     def __len__(self):
#         return len(self.images)
# 
#     def __getitem__(self, idx):
#         img_path, label = self.images[idx]
#         image = Image.open(img_path).convert('RGB')
# 
#         if self.transform:
#             image = self.transform(image)
# 
#         return image, label
# 
# if __name__ == "__main__":
#     # Set random seed for reproducibility
#     torch.manual_seed(42)
# 
#     # Define data transformations
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.ToTensor(),
#     ])
# 
#     # Load dataset and split into training and validation sets
#     dataset = CustomDataset(root_dir="data", transform=transform)
# 
#     # Split dataset into training and validation sets
#     train_size = int(0.6 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
# 
#     # Create DataLoader
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
# 
#     # Initialize the model, loss function, and optimizer
#     print(dataset.classes)
#     model = SimpleCNN(num_classes=len(dataset.classes))
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
# 
#     # Training loop
#     num_epochs = 50
#     for epoch in range(num_epochs):
#         model.train()
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
# 
#         # Validation loop
#         model.eval()
#         val_loss = 0.0
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
# 
#                 _, predicted = outputs.max(1)
#                 total += labels.size(0)
#                 correct += predicted.eq(labels).sum().item()
# 
#         avg_val_loss = val_loss / len(val_loader)
#         accuracy = correct / total * 100
# 
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')
# 
#     from datetime import datetime
#     # Save the trained model
#     torch.save(model.state_dict(), f'simple_cnn_model_{datetime.now()}.pth')
# 
