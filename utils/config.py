from torchvision import transforms as T
class Config():
    train_roots = ['F:/DeepL/cifar-10-batches-py/data_batch_'
                        +str(i) for i in range(1, 6)]
    test_roots = ['F:/DeepL/cifar-10-batches-py/test_batch']
    net = 'Res' # 网络结构
    num_classes = 10 # 类别数
    nw = 0 # 多线程加载数据集（windows多线程加载有问题，所以改成了0）
    wd = 0.001 # 权重衰减
    m = 0.9 # SGD动量
    bs = 512 # batchsize
    epochs = 20 
    lr = 0.001
    # 自定义训练和测试数据集所用的数据增强
    train_trans = T.Compose([
        T.RandomCrop(32),
        T.RandomRotation((-10, 10)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])])
    test_trans = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])])
    
