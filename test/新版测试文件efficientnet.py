import os
import sys
import time
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
from torchvision.models import efficientnet_b0
import csv
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
import random
from PIL import Image

def get_image_paths_and_labels(root_dir):
    image_paths = []
    labels = []
    class_to_idx = {}
    idx = 0
    for class_name in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        if class_name not in class_to_idx:
            class_to_idx[class_name] = idx
            idx += 1
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            if os.path.isfile(img_path):
                image_paths.append(img_path)
                labels.append(class_to_idx[class_name])
    return image_paths, labels, class_to_idx

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def random_hyperparams():
    lr = random.choice([1e-4, 5e-5, 3e-5])
    weight_decay = random.choice([0.01, 0.05, 0.1])
    batch_size = random.choice([32, 64])
    return lr, weight_decay, batch_size

def main():
    model_name = "EfficientNet"
    dataset_root =  '甲骨文手写体+甲骨文拓片转手写diffusion_增广'  
    DateFrom =  '甲骨文手写体+甲骨文拓片转手写diffusion_增广'
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    all_image_paths, all_labels, class_to_idx = get_image_paths_and_labels(os.path.join(dataset_root, "train"))
    num_classes = len(class_to_idx)
    print("num_classes: ", num_classes)

    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    with open(f'{model_name}_{DateFrom}_kfold_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['fold', 'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'f1_score', 'auc', 'top5_acc', 'lr', 'weight_decay', 'batch_size',
                     'test_loss', 'test_acc', 'test_f1', 'test_auc', 'test_top5_acc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for fold, (train_idx, val_idx) in enumerate(skf.split(all_image_paths, all_labels)):
            print(f'Fold {fold+1}/{k_folds}')
            lr, weight_decay, batch_size = random_hyperparams()
            print(f'Fold {fold+1} hyperparams: lr={lr}, weight_decay={weight_decay}, batch_size={batch_size}')
            nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
            train_paths = [all_image_paths[i] for i in train_idx]
            train_labels = [all_labels[i] for i in train_idx]
            val_paths = [all_image_paths[i] for i in val_idx]
            val_labels = [all_labels[i] for i in val_idx]
            train_dataset = CustomDataset(train_paths, train_labels, transform=data_transform)
            val_dataset = CustomDataset(val_paths, val_labels, transform=data_transform)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

            net = efficientnet_b0(weights=None)
            net.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
            net.to(device)

            loss_function = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

            best_acc = 0.0
            early_stop_patience = 100  # 容忍的epoch数
            early_stop_counter = 0
            save_path = f'./{model_name}_{DateFrom}_fold{fold+1}.pth'
            epochs = 100
            for epoch in range(epochs):
                net.train()
                running_loss = 0.0
                train_acc = 0.0
                train_bar = tqdm(train_loader, file=sys.stdout)
                for step, data in enumerate(train_bar):
                    images, labels = data
                    optimizer.zero_grad()
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    loss = loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    predict_y = torch.max(outputs, dim=1)[1]
                    train_acc += torch.eq(predict_y, labels).sum().item()
                    train_bar.desc = f"train epoch[{epoch+1}/{epochs}] loss:{loss:.3f} acc:{train_acc/len(train_dataset):.3f}"

                net.eval()
                val_loss = 0.0
                acc = 0.0
                all_labels_fold = []
                all_preds = []
                all_probs = []
                top5_correct = 0
                with torch.no_grad():
                    val_bar = tqdm(val_loader, file=sys.stdout)
                    for val_data in val_bar:
                        val_images, val_labels_ = val_data
                        val_images, val_labels_ = val_images.to(device), val_labels_.to(device)
                        outputs = net(val_images)
                        loss = loss_function(outputs, val_labels_)
                        val_loss += loss.item()
                        probs = torch.softmax(outputs, dim=1)
                        all_probs.append(probs.cpu().numpy())
                        all_labels_fold.append(val_labels_.cpu().numpy())
                        all_preds.append(torch.argmax(probs, dim=1).cpu().numpy())
                        acc += torch.eq(torch.argmax(probs, dim=1), val_labels_).sum().item()
                        _, top5 = torch.topk(probs, 5, dim=1)
                        top5_correct += sum([val_labels_.cpu().numpy()[i] in top5.cpu().numpy()[i] for i in range(len(val_labels_))])
                val_accurate = acc / len(val_dataset)
                all_labels_fold = np.concatenate(all_labels_fold)
                all_preds = np.concatenate(all_preds)
                all_probs = np.concatenate(all_probs)
                f1 = f1_score(all_labels_fold, all_preds, average='weighted')
                all_labels_one_hot = label_binarize(all_labels_fold, classes=range(num_classes))
                try:
                    auc = roc_auc_score(all_labels_one_hot, all_probs, multi_class='ovr')
                except Exception:
                    auc = 0.0
                top5_acc = top5_correct / len(val_dataset)
                print(f'[fold {fold+1} epoch {epoch+1}] train_loss: {running_loss/len(train_loader):.3f} train_acc: {train_acc/len(train_dataset):.3f} val_loss: {val_loss/len(val_loader):.3f} val_acc: {val_accurate:.3f} f1: {f1:.3f} auc: {auc:.3f} top5_acc: {top5_acc:.3f}')
                if val_accurate > best_acc:
                    best_acc = val_accurate
                    torch.save(net.state_dict(), save_path)
                    early_stop_counter = 0  # 有提升则归零
                else:
                    early_stop_counter += 1  # 无提升则+1
                if early_stop_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1} for fold {fold+1}")
                    break

                # ====== 每个epoch结束后用test集评估 ======
                test_image_paths, test_labels, _ = get_image_paths_and_labels(os.path.join(dataset_root, "test"))
                test_dataset = CustomDataset(test_image_paths, test_labels, transform=data_transform)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)
                net.eval()
                test_loss = 0.0
                test_acc = 0.0
                all_test_labels = []
                all_test_preds = []
                all_test_probs = []
                test_top5_correct = 0
                with torch.no_grad():
                    for test_data in test_loader:
                        test_images, test_labels_ = test_data
                        test_images, test_labels_ = test_images.to(device), test_labels_.to(device)
                        outputs = net(test_images)
                        loss = loss_function(outputs, test_labels_)
                        test_loss += loss.item()
                        probs = torch.softmax(outputs, dim=1)
                        all_test_probs.append(probs.cpu().numpy())
                        all_test_labels.append(test_labels_.cpu().numpy())
                        all_test_preds.append(torch.argmax(probs, dim=1).cpu().numpy())
                        test_acc += torch.eq(torch.argmax(probs, dim=1), test_labels_).sum().item()
                        _, top5 = torch.topk(probs, 5, dim=1)
                        test_top5_correct += sum([test_labels_.cpu().numpy()[i] in top5.cpu().numpy()[i] for i in range(len(test_labels_))])
                test_accurate = test_acc / len(test_dataset)
                all_test_labels = np.concatenate(all_test_labels)
                all_test_preds = np.concatenate(all_test_preds)
                all_test_probs = np.concatenate(all_test_probs)
                test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')
                all_test_labels_one_hot = label_binarize(all_test_labels, classes=range(num_classes))
                try:
                    test_auc = roc_auc_score(all_test_labels_one_hot, all_test_probs, multi_class='ovr')
                except Exception:
                    test_auc = 0.0
                test_top5_acc = test_top5_correct / len(test_dataset)
                print(f'[fold {fold+1} epoch {epoch+1}] test_loss: {test_loss/len(test_loader):.3f} test_acc: {test_accurate:.3f} test_f1: {test_f1:.3f} test_auc: {test_auc:.3f} test_top5_acc: {test_top5_acc:.3f}')

                writer.writerow({
                    'fold': fold+1,
                    'epoch': epoch+1,
                    'train_loss': round(running_loss/len(train_loader), 3),
                    'train_acc': round(train_acc/len(train_dataset), 3),
                    'val_loss': round(val_loss/len(val_loader), 3),
                    'val_acc': round(val_accurate, 3),
                    'f1_score': round(f1, 3),
                    'auc': round(auc, 3),
                    'top5_acc': round(top5_acc, 3),
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'batch_size': batch_size,
                    'test_loss': round(test_loss/len(test_loader), 3),
                    'test_acc': round(test_accurate, 3),
                    'test_f1': round(test_f1, 3),
                    'test_auc': round(test_auc, 3),
                    'test_top5_acc': round(test_top5_acc, 3)
                })
    print('Finished K-Fold Training')

if __name__ == '__main__':
    main()