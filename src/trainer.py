import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, train_loader, test_loader, net, device):
        self.device = device
        print(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = net.to(self.device)

        self.correct = 0
        self.total = 0
        self.class_correct = list(0. for i in range(10))
        self.class_total = list(0. for i in range(10))

        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    def train(self, epochs, lr):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        sum_loss = 0.0
        for e in range(epochs):
            sum_loss = 0.0
            for i, (data, label) in enumerate(self.train_loader):
                data = data.to(self.device)
                label = label.to(self.device)
                optimizer.zero_grad()
                pred = self.model(data)
                loss = criterion(pred, label)
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
            print('Epoch {0} Loss: {1}'.format(e, sum_loss / (i+1)))

            if e % int(epochs / 3) == 0 or e == epochs - 1:
                self.test()
                self.model.train()

        torch.save(self.model.state_dict(), 'cnn_model.pth')

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for (data, label) in self.test_loader:
                data = data.to(self.device)
                label = label.to(self.device)
                pred = self.model(data)
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
        
