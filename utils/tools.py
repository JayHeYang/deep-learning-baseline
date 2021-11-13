import numpy as np
import matplotlib.pyplot as plt

def calculation_accuracy(pred, label):
    pred = pred.cpu().detach().numpy() # 如果没有用GPU加速，则把.cpu()删除
    pred = np.argmax(pred, axis=1)
    # 如果没有用GPU加速，则把.cpu()删除
    right_count = np.sum(pred == label.cpu().numpy())
    acc = right_count / label.size(0)
    return acc


def training_process_visualization(data):
    train_acc = data['train_acc']
    train_loss = data['train_loss']
    test_acc = data['test_acc']

    plt.figure(1)
    plt.plot(range(len(train_loss)), train_loss)
    plt.title('trian_loss')
    plt.ylabel('trian loss')
    plt.xlabel('step')
    plt.savefig('train_loss.png')

    plt.figure(2)
    plt.plot(range(len(train_acc)), train_acc)
    plt.title('train_acc')
    plt.ylabel('train acc')
    plt.xlabel('step')
    plt.savefig('train_acc.png')

    plt.figure(3)
    plt.plot(range(len(test_acc)), test_acc)
    plt.title('test_acc')
    plt.ylabel('test acc')
    plt.xlabel('epoch')
    plt.savefig('test_acc.png')
    plt.show()


