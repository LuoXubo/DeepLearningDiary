import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

# 读取数据集
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式
# 并除以255使得所有像素的数值均在0到1之间
# trans = transforms.ToTensor()
# mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=False)
# mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=False)

# print(len(mnist_train), len(mnist_test))
# print(mnist_train[0][0].shape)
# def get_fashion_mnist_labels(labels):
#     '''返回Fashion-MNIST数据集的文本标签'''
#     text_labels = [
#         't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
#         'sneaker', 'bag', 'ankle boot'
#     ]
#     return [text_labels[int(i)] for i in labels]

'''可视化这些样本'''
# def show_images(img, num_rows, num_cols, title=None, scale=1.5):
#     '''Plot a list of images'''
#     figsize = (num_cols * scale, num_rows * scale)
#     _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
#     axes = axes.flatten()
#     for i, (ax, img) in enumerate(zip(axes,img)):
#         if torch.is_tensor(img):
#             # 图片张量
#             ax.imshow(img.numpy())
#         else:
#             # PIL图片
#             ax.imshow(img)
#         ax.axes.get_xaxis().set_visible(False)
#         ax.axes.get_yaxis().set_visible(False)
#         if title:
#             ax.set_title(title[i])
#
#     return axes

# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_images(X.reshape(18, 28, 28), 2, 9, title=get_fashion_mnist_labels(y))


# 读取小批量
# batch_size = 256
def get_dataloader_workers():
    '''使用四个进程来读取数据'''
    return 4

# train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
#                              num_workers=get_dataloader_workers())
#
# timer = d2l.Timer()
# for X,y in train_iter:
#     continue
# print(f'{timer.stop(): .2f} sec')

# 整合所有组件
def load_data_fashion_mnist(batch_size, resize=None):
    '''下载Fashion-MNIST数据集，然后将其加载到内存中'''
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True,
                                                    transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data/', train=False,
                                                   transform=trans, download=True)

    return (data.DataLoader(mnist_train,batch_size,shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test,batch_size,shuffle=False,
                            num_workers=get_dataloader_workers()))

# 通过指定resize参数来测试load_data_fashion_mnist函数图像大小调整功能
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
