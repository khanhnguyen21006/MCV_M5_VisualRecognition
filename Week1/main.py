# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
from torchvision import transforms, datasets

import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class FinalNet(nn.Module):
    def __init__(self, output_channels, input_channels, dropout=0.5):
        super(FinalNet, self).__init__()

        self.conv_block_1 = nn.Sequential(nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=2, bias=False), nn.ReLU(), nn.BatchNorm2d(32),
                                          nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_block_2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2, bias=False), nn.ReLU(), nn.BatchNorm2d(32),
                                          nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2, bias=False), nn.ReLU(), nn.BatchNorm2d(32),
                                          nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_block_3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2, bias=False), nn.ReLU(), nn.BatchNorm2d(64),
                                          nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, bias=False), nn.ReLU(), nn.BatchNorm2d(64),
                                          nn.MaxPool2d(kernel_size=2, stride=2),
                                          nn.Dropout(dropout))
        # self.conv_block_4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2, dilation=2, bias=False), nn.ReLU(), nn.BatchNorm2d(128),
        #                                   nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, bias=False), nn.ReLU(), nn.BatchNorm2d(128),
        #                                   nn.Dropout(dropout))
        self.conv_block_5 = nn.Sequential(nn.Conv2d(64, output_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.ReLU(), nn.BatchNorm2d(output_channels),
                                          nn.AdaptiveAvgPool2d((1, 1)))

        self.fc1 = nn.Sequential(nn.Flatten(), nn.Softmax())

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        # x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.fc1(x)

        return x


def get_model(device, output_channels, input_channels, dropout=0.5):
    model = FinalNet(output_channels, input_channels, dropout)
    return model.to(device)


def get_data(train_dir, test_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=0)

    test_set = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size - 2,
                                              shuffle=True, num_workers=0)

    return train_loader, test_loader


def loss_batch(model, loss_func, xb, yb, opt=None):
    outputs = model(xb)
    loss = loss_func(outputs, yb)
    _, preds = torch.max(outputs, 1)
    acc = torch.sum(preds == yb.data).item()

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), len(xb), acc


def fit(device, epochs, model, loss_func, opt, train_dl, test_dl):
    writer = SummaryWriter('./runs/final_net_training_' + str(time.time()))
    start = time.time()

    train_loss_vec = []
    test_loss_vec = []
    train_acc_vec = []
    test_acc_vec = []

    for e in range(epochs):
        model.train()
        train_loss = 0.0
        train_corrects = 0
        for i, (xb, yb) in enumerate(train_dl):
            loss, _, acc = loss_batch(model, loss_func, xb.to(device), yb.to(device), opt)
            train_loss += loss
            train_corrects += acc
            # if i % 2000 == 1999:
            #     print(f'[epoch {e + 1:%d}, mini-batch {i + 1:%3d}]: training loss {training_loss/2000}')
            #     training_loss = 0.0
        train_loss_vec.append(train_loss / len(train_dl.dataset))
        train_acc_vec.append(train_corrects / len(train_dl.dataset))
        writer.add_scalar('training loss', train_loss_vec[-1], e)
        writer.add_scalar('training accuracy', train_acc_vec[-1], e)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_corrects = 0
            for xb, yb in test_dl:
                loss, _, acc = loss_batch(model, loss_func, xb.to(device), yb.to(device))
                val_loss += loss
                val_corrects += acc
            test_loss_vec.append(val_loss / len(test_dl.dataset))
            test_acc_vec.append(val_corrects / len(test_dl.dataset))
            writer.add_scalar('validation loss', test_loss_vec[-1], e)
            writer.add_scalar('validation accuracy', test_acc_vec[-1], e)
        print(f'[epoch {e + 1}]: train loss {train_loss/len(train_dl.dataset):10.5f} - val loss {val_loss/len(test_dl.dataset):10.5f} - train acc {train_corrects/len(train_dl.dataset)*100.0:10.2f} - val acc {val_corrects/len(test_dl.dataset)*100.0:10.2f}')
    writer.close()
    print(f'Finished Training Process, time elapsed {time.time() - start} ms')

    return train_loss_vec, test_loss_vec, train_acc_vec, test_acc_vec


def visualize(train_loss_vec, test_loss_vec, train_acc_vec, test_acc_vec, epochs, work_dir):
    # accuracy graph
    epoch_axis = list(range(epochs))
    plt.plot(epoch_axis, train_acc_vec)
    plt.plot(epoch_axis, test_acc_vec)
    plt.legend(['train', 'validation'], loc='lower right')
    plt.title('training and validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig(work_dir + os.sep + 'accuracy.jpg')

    plt.figure()

    # loss graph
    plt.plot(epoch_axis, train_loss_vec)
    plt.plot(epoch_axis, test_loss_vec)
    plt.legend(['train', 'validation'], loc='upper right')
    plt.title('training and validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(work_dir + os.sep + 'loss.jpg')


def main():
    data_dir = '~/MIT_split'
    work_dir = '~/work'
    train_dir = data_dir + os.sep + 'train'
    test_dir = data_dir + os.sep + 'test'
    # hyper parameters
    learning_rate = 1e-4
    epochs = 100
    input_channels = 8
    output_channels = 3
    batch_size = 16

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU for training, number of GPUs available: {str(torch.cuda.device_count())}')
    else:
        device = torch.device('cpu')
        print('Using CPU for training')

    train_dl, test_dl = get_data(train_dir, test_dir, batch_size)
    print('DataLoaders created.')
    model = get_model(device, input_channels, output_channels, dropout=0.5)
    model.to(device)
    print('Model defined.')
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss_vec, test_loss_vec, train_acc_vec, test_acc_vec = fit(device, epochs, model, criterion, opt, train_dl, test_dl)
    visualize(train_loss_vec, test_loss_vec, train_acc_vec, test_acc_vec, epochs, work_dir)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
