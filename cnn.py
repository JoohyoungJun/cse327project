import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform FashionMNIST to 3-channel RGB and resize to 32x32
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize each channel
])

# Use FashionMNIST w/o preprocessing
#transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.5,), (0.5,))
#])

# Load FashionMNIST dataset
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load pretrained models
model_dict = {
    #'AlexNet': lambda: models.alexnet(weights=models.AlexNet_Weights.DEFAULT),
    'VGG11': lambda: models.vgg11(weights=models.VGG11_Weights.DEFAULT),
    'GoogLeNet': lambda: models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT, aux_logits=True, transform_input=False),
    'ResNet18': lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
    'MobileNetV2': lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
    #'EfficientNetB0': lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
}

results = {}

# Training and evaluation function
def train_and_evaluate(model_name, epochs=10):
    print(f"Training {model_name}")
    model = model_dict[model_name]()

    # Modify final classifier layer to match 10 output classes

    # 1. Option for AlexNet and VGG11: the final output layer is the last element in the 'classifier' list
    if model_name in ["AlexNet", "VGG11"]:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
    # 2. Option for ResNet18 and GoogLeNet: the final layer is named 'fc'
    elif model_name in ["ResNet18", "GoogLeNet"]:
        model.fc = nn.Linear(model.fc.in_features, 10)
    # 3. Option for EfficientNetB0 and MobileNetV2: the output layer is at index 1 in the 'classifier' list
    elif model_name in ["EfficientNetB0", "MobileNetV2"]:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)


    # used in 1-ch
    #if model_name == "AlexNet":
    #    model.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
    #elif model_name == "VGG11":
    #    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    #    model.features = model.features[:-2]
    #elif model_name == "ResNet18":
    #    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    #    model.maxpool = nn.Identity()   
    #elif model_name == "GoogLeNet":
    #    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    #    model.maxpool1 = nn.Identity()
    #elif model_name == "MobileNetV2":
    #    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
    #elif model_name == "EfficientNetB0":
    #    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)



    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Training loop with tqdm (progress bar)
        for images, labels in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # logic for GoogleNet
            # Use outputs.logits if exists, else use outputs directly
            if hasattr(outputs, "logits"): 
                main_output = outputs.logits
            else:
                main_output = outputs

            loss = criterion(main_output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                # logic for GoogleNet
                # Use outputs.logits if exists, else use outputs directly
                if hasattr(outputs, "logits"): 
                    main_output = outputs.logits
                else:
                    main_output = outputs

                _, predicted = torch.max(main_output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        test_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, Accuracy={accuracy:.4f}")

    # Save results
    results[model_name] = {
        'loss': train_losses,
        'acc': test_accuracies,
        'final_accuracy': test_accuracies[-1]
    }

# Train and evaluate all models
for model_name in model_dict.keys():
    start = time.time()
    train_and_evaluate(model_name)
    print(f"{model_name} done in {time.time() - start:.1f}s\n")

# Plot training loss and test accuracy
plt.figure(figsize=(14, 6))

# Plot training loss
plt.subplot(1, 2, 1)
for name, data in results.items():
    plt.plot(data["loss"], label=name)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Plot test accuracy
plt.subplot(1, 2, 2)
for name, data in results.items():
    plt.plot(data["acc"], label=name)
plt.title("Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("cnn_loss_accuracy.png")
plt.close()
