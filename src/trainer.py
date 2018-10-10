import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, train_loader, test_loader, net):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.net = net

        self.correct = 0
        self.total = 0
        self.class_correct = list(0. for i in range(10))
        self.class_total = list(0. for i in range(10))

        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    def train(self, epochs, lr):
        self.net.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=lr)

        for e in range(epochs):
            for i, (data, label) in enumerate(self.train_loader):
                pred = self.net(data)
                loss = criterion(pred, label)
                loss.backward()
                optimizer.zero_grad()
                optimizer.step()
            print('Epoch {0} Loss: {1}'.format(e, loss.item()))

            if e % (epochs / 3) == 0 or e == epochs - 1:
                print('[CHECK] total: {}'.format(self.total))
                self.test()

    def test(self):
        self.net.eval()
        for (data, label) in self.test_loader:
            pred = self.net(data)
            prob, pred_class = torch.max(pred.data, 1)
            self.total += label.size(0)
            self.correct += (pred_class == label).sum().item()
            c = (pred_class == label).squeeze()

            for i in range(4):
                l = label[i]
                self.class_correct[l] += c[i].item()
                self.class_total[l] += 1

        print('Accuracy: {}%'.format(100 * self.correct / self.total))
        for i in range(10):
            print('Accuracy of {} : {:.6f} %'.format(
                self.classes[i], 100 * self.class_correct[i] / self.class_total[i]))
        
