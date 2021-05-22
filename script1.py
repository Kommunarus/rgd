import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from unit import CustomImageDataset
from unit import Net

device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_epoch = 40

if __name__ == '__main__':
    training_data = CustomImageDataset(
        train=True, device=device
    )
    testing_data = CustomImageDataset(
        train=False, device=device
    )
    train_dataloader = DataLoader(training_data, batch_size=6, shuffle=True)
    testloader = DataLoader(testing_data, batch_size=4, shuffle=False)


    # model = NetD().to(device)
    model = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    maxval = 0
    bestval = ''

    for epoch in range(n_epoch):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_loss3 = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels1, labels2, labels3 = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputsOpenDoor,  outputsInDoor, outputsUpal = model(inputs)
            loss1 = criterion(outputsOpenDoor, labels1)
            loss2 = criterion(outputsInDoor, labels2)
            loss3 = criterion(outputsUpal, labels3)
            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
            running_loss3 += loss3.item()
        print('{} loss: {} {} {} {}'.format(epoch + 1, running_loss, running_loss1, running_loss2, running_loss3))
        model.eval()

        correct1 = 0
        correct2 = 0
        correct3 = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                inputs, labels1, labels2, labels3 = data
                # calculate outputs by running images through the network
                outputsOpenDoor,  outputsInDoor, outputsUpal = model(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predictedOpenDoor = torch.max(outputsOpenDoor.data, 1)
                _, predictedInDoor = torch.max(outputsInDoor.data, 1)
                _, predictedUpal = torch.max(outputsUpal.data, 1)
                total += labels1.size(0)
                correct1 += (predictedOpenDoor == labels1).sum().item()
                correct2 += (predictedInDoor == labels2).sum().item()
                correct3 += (predictedUpal == labels3).sum().item()

        if maxval <= correct1+correct2+correct3:
            PATH = './rjd_3d_best_3.pth'
            torch.save(model.state_dict(), PATH)
            bestval = 'Best: {} {} {} ({}) из {}'.format(correct1, correct2, correct3, correct1+correct2+correct3, total)
            maxval = correct1+correct2+correct3

        print('{} {} {} ({}) из {}'.format(correct1, correct2, correct3, correct1 + correct2 + correct3, total))

        # running_loss = 0.0
    print('Finished Training')

    PATH = './rjd_3d_3.pth'
    torch.save(model.state_dict(), PATH)
    print(bestval)