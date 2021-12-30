from PIL import Image
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from glob import glob
import torch.utils
import torch.utils.data
from torch import nn
import time

def RGB2HSI(rgb_img):
  """
  这是将RGB彩色图像转化为HSI图像的函数
  :param rgm_img: RGB彩色图像
  :return: HSI图像
  """
  #保存原始图像的行列数
  row = np.shape(rgb_img)[0]
  col = np.shape(rgb_img)[1]
  #对原始图像进行复制
  hsi_img = rgb_img.copy()
  #对图像进行通道拆分
  B,G,R = cv2.split(rgb_img)
  #把通道归一化到[0,1]
  [B,G,R] = [ i/ 255.0 for i in ([B,G,R])]
  H = np.zeros((row, col))  #定义H通道
  I = (R + G + B) / 3.0    #计算I通道
  S = np.zeros((row,col))   #定义S通道
  for i in range(row):
    den = np.sqrt((R[i]-G[i])**2+(R[i]-B[i])*(G[i]-B[i]))
    thetha = np.arccos(0.5*(R[i]-B[i]+R[i]-G[i])/den)  #计算夹角
    h = np.zeros(col)        #定义临时数组
    #den>0且G>=B的元素h赋值为thetha
    h[B[i]<=G[i]] = thetha[B[i]<=G[i]]
    #den>0且G<=B的元素h赋值为thetha
    h[G[i]<B[i]] = 2*np.pi-thetha[G[i]<B[i]]
    #den<0的元素h赋值为0
    h[den == 0] = 0
    H[i] = h/(2*np.pi)   #弧度化后赋值给H通道
  #计算S通道
  for i in range(row):
    min = []
    #找出每组RGB值的最小值
    for j in range(col):
      arr = [B[i][j],G[i][j],R[i][j]]
      min.append(np.min(arr))
    min = np.array(min)
    #计算S通道
    S[i] = 1 - min*3/(R[i]+B[i]+G[i])
    #I为0的值直接赋值0
    S[i][R[i]+B[i]+G[i] == 0] = 0
  #扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
  hsi_img[:,:,0] = H*255
  hsi_img[:,:,1] = S*255
  hsi_img[:,:,2] = I*255
  return hsi_img
 
def HSI2RGB(hsi_img):
  """
  这是将HSI图像转化为RGB图像的函数
  :param hsi_img: HSI彩色图像
  :return: RGB图像
  """
  # 保存原始图像的行列数
  row = np.shape(hsi_img)[0]
  col = np.shape(hsi_img)[1]
  #对原始图像进行复制
  rgb_img = hsi_img.copy()
  #对图像进行通道拆分
  H,S,I = cv2.split(hsi_img)
  #把通道归一化到[0,1]
  [H,S] = [ i/ 255.0 for i in ([H,S])]
  [I] = [ i/ 255.0 for i in ([I])]
  R,G,B = H,S,I
  for i in range(row):
    h = H[i]*2*np.pi
    #H大于等于0小于120度时
    a1 = h >=0
    a2 = h < 2*np.pi/3
    a = a1 & a2     #第一种情况的花式索引
    tmp = np.cos(np.pi / 3 - h)
    b = I[i] * (1 - S[i])
    r = I[i]*(1+S[i]*np.cos(h)/tmp)
    g = 3*I[i]-r-b
    B[i][a] = b[a]
    R[i][a] = r[a]
    G[i][a] = g[a]
    #H大于等于120度小于240度
    a1 = h >= 2*np.pi/3
    a2 = h < 4*np.pi/3
    a = a1 & a2     #第二种情况的花式索引
    tmp = np.cos(np.pi - h)
    r = I[i] * (1 - S[i])
    g = I[i]*(1+S[i]*np.cos(h-2*np.pi/3)/tmp)
    b = 3 * I[i] - r - g
    R[i][a] = r[a]
    G[i][a] = g[a]
    B[i][a] = b[a]
    #H大于等于240度小于360度
    a1 = h >= 4 * np.pi / 3
    a2 = h < 2 * np.pi
    a = a1 & a2       #第三种情况的花式索引
    tmp = np.cos(5 * np.pi / 3 - h)
    g = I[i] * (1-S[i])
    b = I[i]*(1+S[i]*np.cos(h-4*np.pi/3)/tmp)
    r = 3 * I[i] - g - b
    B[i][a] = b[a]
    G[i][a] = g[a]
    R[i][a] = r[a]
  rgb_img[:,:,0] = B*255
  rgb_img[:,:,1] = G*255
  rgb_img[:,:,2] = R*255
  return rgb_img

def load_images(file):
    im = Image.open(file)
    return np.array(im)

def jhh(image):   #直方图均衡化
    rows = image.shape[0]
    cols = image.shape[1]
    for d in range(3):
        t = np.zeros(256)
        for i in range(rows):
            for j in range(cols):
                t[image[i,j,d]] += 1
        for i in range(1,256):
            t[i] += t[i-1]
        for i in range(rows):
            for j in range(cols):
                image[i,j,d] = int(t[image[i,j,d]]/t[255]*255)
    return image

def kz(image):   #扩展
    rows = image.shape[0]
    cols = image.shape[1]
    out = image.copy()
    for d in range(3):
        mi = 255
        ma = 0
        for i in range(rows):
            for j in range(cols):
                mi = min(mi,image[i,j,d])
                ma = max(ma,image[i,j,d])
        for i in range(rows):
            for j in range(cols):
                out[i,j,d] = int((image[i,j,d]-mi)/(ma-mi)*255);
    return out

def image_split(image):
    r = image.copy()
    r[:,:,2]=255
    r[:,:,1]=255
    g = image.copy()
    g[:,:,0]=255
    g[:,:,2]=255
    b = image.copy()
    b[:,:,1]=255
    b[:,:,0]=255
    return [r,g,b]

def jz(image1):  #均值滤波器
    dx = [1,1,1,0,0,0,-1,-1,-1]
    dy = [1,0,-1,1,0,-1,1,0,-1]
    d = [1,1,1,1,1,1,1,1,1]

    rows = image1.shape[0]
    cols = image1.shape[1]
    ite = 1
    while ite>0:
        print(ite)
        ite -= 1
        image2 = np.zeros([rows,cols],dtype = np.uint8)
        for i in range(rows):
            for j in range(cols):
                for k in range(9):
                    xx = i+dx[k]
                    yy = j+dy[k]
                    if xx >= 0 and xx <rows and yy >= 0 and yy <cols :
                        image2[i,j] += image1[xx,yy]/9
                image2[i,j] = int(image2[i,j])
        image1 = image2
    return image1

def ld(image):   #返回图片的估计亮度
    rows = image.shape[0]
    cols = image.shape[1]
    light = image[:,:,0].copy()
    for i in range(rows):
        for j in range(cols):
            light[i,j]=sum(image[i,j])/3;
    return light

def zft(image): #计算直方图 ,0-100
    out = np.zeros(256)
    rows = image.shape[0]
    cols = image.shape[1]
    tmp = 1/(rows*cols)
    for i in range(rows):
        for j in range(cols):
            out[int(sum(image[i,j])/3)] += tmp
    return out

def przft(l,q):
    for i in range(256):
        l[i] = int(l[i]*100)
    x = np.linspace(start=0,stop=255,num = 256)
    y = np.array(l)
    if q == 1:
        plt.subplot(121)
        plt.title("low")
    else:
        plt.subplot(122)
        plt.title("high")
    plt.plot(x,y)

def ans(image, l): #计算结果
    s = sum(l)
    for i in range(len(l)):
        l[i] = l[i]/s
    out = np.zeros(256)
    f = np.zeros(256)
    rows = len(image)
    cols = len(image[0])
    tmp = 1/(rows*cols)
    for i in range(rows):
        for j in range(cols):
            out[int(sum(image[i,j,:])/3)] += tmp
    i=0
    si=0
    j=0
    sj=0
    while i<256 and j<256:
        f[i] = j
        if si+out[i] < sj+l[j]:
            si += out[i]
            i+=1
        else:
            sj += l[j]
            j+=1
    while i<256:
        f[i] = 255
        i+=1
    hsi = RGB2HSI(image)
    for i in range(rows):
        for j in range(cols):
            hsi[i,j,2] = f[hsi[i,j,2]]
    image2 = HSI2RGB(hsi)
    for i in range(rows):
        for j in range(cols):
            if max(image2[i,j,:])>=255:
                if image2[i,j,0]>=255:
                    image2[i,j,0]=255
                    image2[i,j,1]=255*image[i,j,1]/image[i,j,0]
                    image2[i,j,2]=255*image[i,j,2]/image[i,j,0]
                if image2[i,j,1]>=255:
                    image2[i,j,1]=255
                    image2[i,j,0]=255*image[i,j,0]/image[i,j,1]
                    image2[i,j,2]=255*image[i,j,2]/image[i,j,1]
                if image2[i,j,2]>=255:
                    image2[i,j,2]=255
                    image2[i,j,1]=255*image[i,j,1]/image[i,j,2]
                    image2[i,j,0]=255*image[i,j,0]/image[i,j,2]
    return image2


def train(inputs,targets):
    dims = 256
    DATA_SIZE = len(inputs)
    batchsz = 24
    epochs = 100

    lr = 1e-3
    inputs = np.array(inputs, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    tic_total = time.time()

    model = nn.Sequential(
        nn.Linear(dims, dims),
        nn.ReLU(),
        nn.Linear(dims, dims),
        nn.ReLU(),
        nn.Linear(dims, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, dims),
        nn.ReLU(),
        nn.Linear(dims, dims),
        nn.ReLU(),
        nn.Linear(dims, dims),
    ).cuda()

    train_x = torch.tensor(inputs).cuda()
    train_y = torch.tensor(targets).cuda()

    train_db = torch.utils.data.TensorDataset(train_x, train_y)
    train_db = torch.utils.data.DataLoader(train_db, batch_size=batchsz, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        tic_epoch = time.time()
        loss = 0
        for step, data in enumerate(train_db):
            loss = (model(data[0]) - data[1]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        toc_epoch = time.time()
        print('time in epoch %d of %d: %f , loss = %f' % (epoch+1,epochs,toc_epoch - tic_epoch, loss))
    toc_total = time.time()
    print('time elapsed:', toc_total - tic_total)
    # 模型的保存
    torch.save(model.state_dict, 'model.pth')
    return model
    
from skimage import io
def test(eval_low_data,eval_low_data_name,model):
    # 模型的加载
    #model = nn.Module
    #model.load_state_dict(torch.load("model.pth"), strict=False)
    eval_high_data = []
    tot = len(eval_low_data_name)
    for idx in range(len(eval_low_data_name)):
        input = torch.tensor(zft(eval_low_data[idx])).cuda()
        input = input.to(torch.float32)
        output = model(input)
        img = ans(eval_low_data[idx],output.cpu())

        plt.subplot(1,2,1)
        io.imshow(eval_low_data[idx])
        plt.title("low")
        plt.subplot(1,2,2)
        io.imshow(img)
        plt.title("high")
        plt.savefig("./result/compare %d.png" % (idx+1))

        img = Image.fromarray(np.uint8(img))
        path = "./result/Figure" + str(idx) + ".jpg"
        print("Figure %d of %d Finished" % (idx+1,tot))
        img.save(path)

        

#   Input

train_low_img = []
train_high_img = []

train_low_data_names = glob('./data/train/low/*.png')
train_low_data_names.sort()
train_high_data_names = glob('./data/train/high/*.png')
train_high_data_names.sort()
assert len(train_low_data_names) == len(train_high_data_names)

print('[*] Number of training data: %d' % len(train_low_data_names))
tot = len(train_low_data_names)
train_low_data = [] #
train_high_data = [] #
for idx in range(len(train_low_data_names)):
    #print("working on %d of %d" % (idx+1,tot))
    low_im = load_images(train_low_data_names[idx])
    train_low_img.append(low_im)
    high_im = load_images(train_high_data_names[idx])
    train_high_img.append(high_im)

#    train_low_data.append(zft(low_im))
#    train_high_data.append(zft(high_im))

#np.save("train_low_data.npy",np.array(train_low_data))
#np.save("train_high_data.npy",np.array(train_high_data))
#np.save("train_low_data_names.npy",np.array(train_low_data_names))
#np.save("train_high_data_names.npy",np.array(train_high_data_names))

train_low_data = np.load("train_low_data.npy").tolist()
train_high_data = np.load("train_high_data.npy").tolist()
train_low_data_names = np.load("train_low_data_names.npy").tolist()
train_high_data_names = np.load("train_high_data_names.npy").tolist()

#przft(train_low_data[0],1)
#przft(train_high_data[0],2)

#  ================wait====================
eval_low_data = []
eval_high_data = []

eval_low_data_name = glob('./data/test/low/*.*')

for idx in range(len(eval_low_data_name)):
    eval_low_im = load_images(eval_low_data_name[idx])
    eval_low_data.append(eval_low_im)
print("Input finished!");
# Input finished


model = train(train_low_data, train_high_data)
test(eval_low_data,eval_low_data_name,model)



