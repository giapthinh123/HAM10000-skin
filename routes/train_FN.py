import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


class CustomCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1 = self.create_conv_block(3, 16)
        self.conv2 = self.create_conv_block(16, 32)
        self.conv3 = self.create_conv_block(32, 64)
        self.conv4 = self.create_conv_block(64, 64)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 13 * 13, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes),
        )
        
        self.global_avg_pool = nn.AvgPool2d(3, stride=2)

    def create_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)

        x = self.global_avg_pool(x)
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = self.fc2(x) 
        x = self.fc3(x)
        return x

num_epochs = 20
batch_size = 16
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN().to(device)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([4.39, 2.79, 1.30, 12.32, 1.28, 0.21, 10.23]).to(device)) 
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.classes = sorted(dataframe['dx'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx]['image_id'] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = self.class_to_idx[self.dataframe.iloc[idx]['dx']]

        if self.transform:
            image = self.transform(image)
        return image, label

train_df = pd.read_csv('data_split/train/metadata.csv')
val_df = pd.read_csv('data_split/val/metadata.csv')
test_df = pd.read_csv('data_split/test/metadata.csv')

# Tạo Dataset từ thư mục tương ứng
train_dataset = HAM10000Dataset(train_df, 'data_split/train/', transform=train_transforms)
val_dataset = HAM10000Dataset(val_df, 'data_split/val/', transform=val_transforms)
test_dataset = HAM10000Dataset(test_df, 'data_split/test/', transform=val_transforms)

# Tạo DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    # Đánh giá trên tập validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # Cập nhật learning rate
    scheduler.step(val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model_finetuned_v4.pth")
        print(f"✅ Lưu model fine-tuned tốt nhất (Val Acc: {val_acc:.2f}%)")