import argparse
import os
import torch
import torchvision
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from model import VAE
import PIL.Image as Image
from dataset import RacingCarDataset
from core.logger import TensorBoardLogger

num_epochs = 1001   # 300
batch_size = 72
learning_rate = 1e-4

img_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 96, 96)
    return x

def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    # sum or mean
    # KL divergence
    return BCE + KLD

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest='train', action='store_true', help='train model')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    args = parser.parse_args()

    if not os.path.exists('./VAE_training'):
        os.mkdir('./VAE_training')

    board_logger = TensorBoardLogger('./VAE_training')

    model = VAE()
    if torch.cuda.is_available():
        model.cuda()

    '''
    from torchsummary import summary
    vae = VAE().cuda()
    summary(vae, (batch_size, 96*96))
    exit()
    '''

    if args.train:
        dataset = RacingCarDataset('./RacingCarDataset', transform=img_transform) # RacingCarDataset
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) #, num_workers=4

        reconstruction_function = nn.MSELoss(size_average=False)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        t = 0
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            
            for batch_idx, data in enumerate(dataloader):

                img = data
                img = img.view(img.size(0), -1)
                img = Variable(img)
                
                if torch.cuda.is_available():
                    img = img.cuda()
                optimizer.zero_grad()

                recon_batch, z, mu, logvar = model(img)

                loss = loss_function(recon_batch, img, mu, logvar)
                loss.backward()

                train_loss += loss.item()
                optimizer.step()

                if batch_idx % 100 == 0:
                    board_logger.scalar_summary('Epoch Loss', t, loss.item() / len(img))
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t{}'.format(
                        epoch,
                        batch_idx * len(img),
                        len(dataloader.dataset), 100. * batch_idx / len(dataloader),
                        loss.item() / len(img), t))
                    t += 1

            board_logger.scalar_summary('Epoch Average tLoss', epoch, train_loss / len(dataloader.dataset))
            print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / len(dataloader.dataset)))
            if epoch % 100 == 0:
                save = to_img(recon_batch.cpu().data)
                save_image(save, './VAE_training/image_{}.png'.format(epoch))
                torch.save(model.state_dict(), './vae'+str(epoch)+'.pth')

        torch.save(model.state_dict(), './vae.pth')

    elif args.test:
        policy = torch.load('./vae.pth')
        model.load_state_dict(policy)

        for _ in range(8):
            img = Image.open('./README/ImageSource'+str(_)+'.jpg')
            # img = Image.open('./RacingCarDataset_/history143.jpg')
            img = img_transform(img)

            img = img.view(img.size(0), -1)
            # img = Variable(img)

            if torch.cuda.is_available():
                img = img.cuda()

            recon_batch, z, mu, logvar = model(img)
            feature = z.squeeze(0).cpu().detach().numpy()

            print(feature, feature.shape)
    
            save = to_img(recon_batch.cpu().data)
            save_image(save, './VAE_training/image_{}.png'.format('test'+str(_)))

        