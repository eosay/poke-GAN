import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms, utils

'''Vanilla Generative Adversarial Network'''

# disciminator network
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.ndf = 32
        self.main = nn.Sequential(
            nn.Conv2d(3, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x)

# generator network 
class G(nn.Module):
    def __init__(self, latent):
        super(G, self).__init__()
        self.ngf = 32
        self.latent = latent
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.latent, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# custom pytorch dataset
class PokeDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.tform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(os.listdir(self.root))

    def __getitem__(self, idx):
        file = os.path.dirname(__file__)
        working_dir = os.path.join(file, self.root)
        imname = str(idx).zfill(3) + '.jpg'
        impath = os.path.join(working_dir, imname)
        return self.tform(Image.open(impath))


# hyperparameters
epochs = 1000
lr = 0.0003
torch.manual_seed(1)
batch_size = 64
use_cuda = torch.cuda.is_available()
im_samples = 50
latent_size = 100

dataset = PokeDataset('./data64')
dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)

discriminator = D()
generator = G(latent_size)

# loss(o, t) = - 1/n \sum_i (t[i] log(o[i]) + (1 - t[i]) log(1 - o[i]))
loss = nn.BCELoss(size_average=True)

if use_cuda:
    print('CUDA device found and active')
    discriminator.cuda()
    generator.cuda()
    loss.cuda()

# optimizers
optimD = optim.Adam(discriminator.parameters(), lr, betas=(0.5, 0.999))
optimG = optim.Adam(generator.parameters(), lr, betas=(0.5, 0.999))

test_noise = torch.Tensor(batch_size, latent_size, 1, 1).normal_(0, 1)
if use_cuda:
    test_noise = test_noise.cuda()

test_noiseV = Variable(test_noise)

for i in range(epochs):
    for j, data in enumerate(dataloader):
        latent = torch.Tensor(data.size(0), latent_size, 1, 1)
        label = torch.Tensor(data.size(0), 1, 1, 1)

        if use_cuda:
            latent = latent.cuda()
            label = label.cuda()
            data = data.cuda()

        # train discriminator        
        # train on real
        # input an image, 0|1 if fake|real        
        optimD.zero_grad()
        real_label = Variable(label.fill_(1), requires_grad=False)
        real_im = Variable(data, requires_grad=False)

        out = discriminator(real_im)
        loss_real = loss(out, real_label)
        loss_real.backward()

        # train D on fake
        noise = Variable(latent.normal_(0, 1), requires_grad=False)
        fake_label = Variable(label.fill_(0), requires_grad=False)

        fake = generator(noise)
        out = discriminator(fake.detach())
        loss_fake = loss(out, fake_label)
        loss_fake.backward()
        optimD.step()

        # train generator
        fake_real_label = Variable(label.fill_(1), requires_grad=False)       
        optimG.zero_grad()
        out = discriminator(fake)
        loss_gen = loss(out, fake_real_label)
        loss_gen.backward()
        optimG.step()

        print('epoch [{}]/[{}]    batch {}    lossD {:.5f}    lossG {:.5f}'.format(
                i, epochs, j, (loss_real.cpu().data[0] + loss_fake.cpu().data[0]), 
                loss_gen.cpu().data[0]))

        if j % im_samples == 0:
            out = generator(test_noiseV).cpu().data
            utils.save_image(out, './fake.jpg', normalize=True)
            torch.save(discriminator, 'dis.pt')
            torch.save(generator, 'gen.pt')

torch.save(discriminator, 'dis.pt')
torch.save(generator, 'gen.pt')
