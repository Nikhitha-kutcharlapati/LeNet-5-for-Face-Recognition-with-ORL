from orl import ORL
from lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import visdom

viz = visdom.Visdom()

data_train = ORL('./trainlist.txt', 
                 transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor(),
                       transforms.Normalize([0.442], [0.196])]))

data_test = ORL('./testlist.txt', 
                 transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor(),
                       transforms.Normalize([0.442], [0.196])]))

data_train_loader = DataLoader(data_train, batch_size=8, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=8, num_workers=8)

net = LeNet5()
if torch.cuda.is_available():
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)
#optimizer = optim.Adam(net.parameters(), lr=2e-3)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

loss_list, iter_list = [], []
cur_batch_win = None
cur_batch_win_opts = {
    'title': 'Epoch Loss Trace',
    'xlabel': 'Batch Number',
    'ylabel': 'Loss',
    'width': 1200,
    'height': 600,
}


def train(epoch):
    global cur_batch_win
    net.train()
    for i, (images, labels) in enumerate(data_train_loader):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        
        optimizer.zero_grad()

        output = net(images)

        loss = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        iter_list.append((epoch-1)*40+i)

        if i % 1 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        # Update Visualization
        if viz.check_connection():
            cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(iter_list),
                                     win=cur_batch_win, name='current_batch_loss',
                                     update=(None if cur_batch_win is None else 'replace'),
                                     opts=cur_batch_win_opts)

        loss.backward()
        optimizer.step()


def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))


def train_and_test(epoch):
    scheduler.step()
    train(epoch)
    test()
    if epoch % 10 == 0:
        torch.save(net.state_dict(), './models/torch_lenet5_orl_'+str(epoch)+'_params.pkl')


def main():
## For train and val
#    for e in range(1, 101):
#        train_and_test(e)

## For test
    net.load_state_dict(torch.load('./weights/torch_lenet5_orl_100_params_98.75%.pkl'))
    test()

if __name__ == '__main__':
    main()
