import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

image_size = 64
batch_size = 64
stats = (0.5, 0.5)

dataset=datasets.MNIST(root='data',download=True,train=True,
        transform=transforms.Compose([transforms.Resize(image_size),transforms.ToTensor(),transforms.Normalize(stats[0],stats[1])]))
dataloader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)

def denorm(image_tensor):
    return image_tensor*stats[1]+stats[0]

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

device = get_default_device()
dataloader = DeviceDataLoader(dataloader, device)


# define the Discriminator
class Discriminator(nn.Module):
    def __init__(self,input_size,hidden_dim,output_size):
        super(Discriminator,self).__init__()
        self.fc1=nn.Conv2d(1,hidden_dim,4,2,1).cuda()
        # self.fc2=nn.Conv2d(hidden_dim,hidden_dim*2,4,2,1)
        self.lin=nn.Linear(hidden_dim,output_size).cuda()
        self.activation=nn.Tanh().cuda()
        self.hidden_dim=hidden_dim
    def forward(self,x):
        x=F.leaky_relu(self.fc1(x),0.4)
        # x=F.leaky_relu(self.fc2(x),0.4)
        x=x.view(-1,self.hidden_dim)
        x=self.lin(x)
        out=self.activation(x)
        return (out+torch.tensor(1))/2


# define the Generator

class Generator(nn.Module):
    def __init__(self,input_size,hidden_dim,output_size):
        super(Generator,self).__init__()
        self.fc1=nn.ConvTranspose2d(input_size,hidden_dim*8,4,1,0).cuda()
        self.fc_1=nn.BatchNorm2d(hidden_dim*8)
        self.fc2=nn.ConvTranspose2d(hidden_dim*8,hidden_dim*4,4,2,1).cuda()
        self.fc_2=nn.BatchNorm2d(hidden_dim*4)
        self.fc3=nn.ConvTranspose2d(hidden_dim*4,hidden_dim*2,4,2,1).cuda()
        self.fc_3=nn.BatchNorm2d(hidden_dim*2)
        self.fc4=nn.ConvTranspose2d(hidden_dim*2,hidden_dim,4,2,1).cuda()
        self.fc_4=nn.BatchNorm2d(hidden_dim)
        self.fc5=nn.ConvTranspose2d(hidden_dim,1,4,2,1).cuda()
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc3.weight)
        init.xavier_normal_(self.fc4.weight)
        init.xavier_normal_(self.fc5.weight)
        self.out=nn.Sigmoid().cuda()
        # self.hidden_dim=hidden_dim
        # self.linear = nn.Linear(input_size,output_size)
        self.input_size=input_size
    def forward(self,x):
        # x=F.leaky_relu(self.fc1(x),0.1)
        # x=F.leaky_relu(self.fc2(x),0.1)
        # x=F.leaky_relu(self.fc3(x),0.1)
        # x=F.leaky_relu(self.fc4(x),0.1)
        # x=F.leaky_relu(self.fc5(x),0.1)
        x=F.tanh(self.fc1(x))
        x = self.fc_1(x)
        x=F.tanh(self.fc2(x))
        x = self.fc_2(x)
        x=F.tanh(self.fc3(x))
        x = self.fc_3(x)
        x=F.tanh(self.fc4(x))
        x = self.fc_4(x)
        x=F.tanh(self.fc5(x))
        # x = self.fc_5(x)
        out = self.out(x)
        # out=(self.out(x)+torch.tensor(1))/2
        # return torch.ones_like(out)*torch.sum(self.linear.weight)
        # return self.linear(x.reshape(-1,self.input_size))
        return out 
# Discriminator hyperparams
input_size = image_size**2
d_output_size = 1
d_hidden_size = 32

# Generator hyperparams
z_size = 100
g_output_size = input_size
g_hidden_size = 32

# instantiate discriminator and generator
D = Discriminator(input_size, d_hidden_size, d_output_size)
G = Generator(z_size, g_hidden_size, g_output_size)

print(device)

# move models to GPU
D = D.to(device)
G = G.to(device)

# training hyperparams
num_epochs = 5
lr = 1e7

# Create optimizers for the discriminator and generator
# d_optimizer = torch.optim.SGD(D.parameters(), lr) 
# g_optimizer = torch.optim.Adam(G.parameters(), lr) 
g_optimizer = torch.optim.Adam(G.parameters(), lr) 


# calculate the number of batches per epoch
num_batches = len(dataloader)
print(num_batches)

# Train the discriminator and generator
for epoch in range(1):
    for batch_i, (real_images, _) in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images*2 - 1
        real_images = real_images.view(-1, 1, image_size, image_size)
        real_images = to_device(real_images,device)
        # TRAIN THE DISCRIMINATOR
        if batch_i % 20 == 0 and True:
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            # 1. Train with real images
            # Compute the discriminator losses on real images 
            D_real = D(real_images)
            d_real_loss = torch.mean((1-D_real)**2)
            
            # 2. Train with fake images
            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size , 1, 1))
            z = torch.from_numpy(z).float()
            z = z.cuda()
            fake_images = G(z)
            
            # Compute the discriminator losses on fake images        
            D_fake = D(fake_images)
            d_fake_loss = torch.mean(D_fake**2)
            
            # add up loss and perform backprop
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
        
        # =========================================
        #            TRAIN THE GENERATOR
        # =========================================
            
        for _ in range(6):
            g_optimizer.zero_grad()
            # d_optimizer.zero_grad()
            
            # 1. Train with fake images and flipped labels
            
            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size ,1 ,1))
            # print(z)
            z = torch.from_numpy(z).float()
            z = z.cuda()
            # z = torch.ones((batch_size, z_size ,1 ,1),dtype=torch.float32).cuda()
            fake_images = G(z)
            # print(fake_images)
            # Compute the discriminator losses on fake images 
            # using flipped labels!
            D_fake = D(fake_images)
            g_loss = torch.mean((1-D_fake)**2)
            # g_loss = torch.sum(fake_images)
            # print('grad',G.linear.weight.grad)
            
            # perform backprop
            g_loss.backward()
            print(G.fc1.weight.grad)
            g_optimizer.step()

        # Print some loss stats
        if batch_i % 2 == 0:
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch+1, num_epochs, d_loss.item(), g_loss.item()))
        if batch_i>1:
          break


# Show the generated images
fig, ax = plt.subplots()
fake_images = fake_images.detach().numpy()
fake_images = fake_images.reshape(-1, image_size, image_size)

#denormalize
fake_images = denorm(torch.tensor(fake_images))

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(fake_images[i], cmap='gray')
    plt.axis('off')
plt.show()

