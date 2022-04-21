import sys
import os
import torch
import torchvision
import torch.utils.data as data
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
import torchvision.models as models
from torch import nn
from torch import optim
import numpy as np
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="usage info")
    parser.add_argument('-data_dir', '--data_dir', help="training data full directory", type=str, required=True)
    parser.add_argument('-model_save_dir', '--model_save_dir', help="model_save_dir", type=str, required=True)
    parser.add_argument('-n_epochs', '--n_epochs', type=int)
    args = parser.parse_args()
    global data_dir, checkpoint_path, n_epochs
    data_dir = args.data_dir
    checkpoint_path = args.model_save_dir
    n_epochs = args.n_epochs
    print(args.data_dir, args.data_dir)



data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
                                    ])
#data_dir = r'D:\tencent_electronics'
dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=data_transform)
# Split dataset with  stratify into train and validation
train_indices, val_indices = train_test_split(list(range(len(dataset.targets))), test_size=0.2, stratify=dataset.targets)
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

# Create DataLoader
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)


model = models.vgg19_bn(pretrained=True)
num_classes = 4
n_inputs = model.classifier[6].in_features
model.classifier[6] = nn.Sequential(
                      nn.Linear(n_inputs, 256),
                      nn.ReLU(),
                      nn.Dropout(0.4),
                      nn.Linear(256, num_classes),
                      nn.LogSoftmax(dim=1))

for param in model.parameters():
    param.requires_grad = False

for param in model.classifier[6].parameters():
    param.requires_grad = True


# model = models.resnet18(pretrained=True)
# model = models.resnext101_32x8d(pretrained=True)
# n_inputs = model.fc.in_features
# model.fc = nn.Sequential(
#                       nn.Linear(n_inputs, 256),
#                       nn.ReLU(),
#                       nn.Dropout(0.4),
#                       nn.Linear(256, num_classes),
#                       nn.LogSoftmax(dim=1))
#
# for param in model.parameters():
#     param.requires_grad = False
#
# for param in model.fc.parameters():
#     param.requires_grad = True


# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')


criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

checkpoint_path = os.path.join(os.path.dirname(__file__), 'vgg19_bn_model.pt')




def train(model, optimizer, criterion, trainloader, validloader, checkpoint_path, n_epochs=1):
    min_val_loss = np.Inf
    n_epochs = n_epochs
    # Main loop
    for epoch in range(n_epochs):
        total_trues_predicts = 0
        all_predicts=0
        # Initialize validation loss for epoch
        val_loss = 0
        running_loss = 0
        model.train()
        # Training loop
        for data, targets in trainloader:

            optimizer.zero_grad()
            # Generate predictions
            out = model(data)
            # Calculate loss
            loss = criterion(out, targets)
            running_loss += loss
            # Backpropagation
            loss.backward()
            # Update model parameters
            optimizer.step()
            print('batch loss: ', loss)
        print("Training Loss: {:.6f}".format(running_loss/len(trainloader)))
        # Validation loop
        with torch.no_grad():
            model.eval()
            for data, targets in validloader:
                # Generate predictions
                out = model(data)
                # Calculate loss
                loss = criterion(out, targets)
                val_loss += loss

                total_trues_predicts += torch.sum(targets==out.argmax(dim=1)).item()
                all_predicts += len(targets)
                print("batch_accuracy_score: {:.6f}".format(total_trues_predicts / all_predicts))
            print("validation Loss: {:.6f}".format(val_loss / len(validloader)))
            print("accuracy_score: {:.6f}".format(total_trues_predicts / all_predicts))
            # Average validation loss
            val_loss = val_loss / len(validloader)
            # If the validation loss is at a minimum
            if val_loss < min_val_loss:
                # Save the model
                torch.save(model.state_dict(), checkpoint_path)
                min_val_loss = val_loss

#the training call
train(model, optimizer, criterion, train_data_loader, val_data_loader, checkpoint_path, n_epochs=n_epochs)
