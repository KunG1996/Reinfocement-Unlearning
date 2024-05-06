import pickle

import matplotlib.pyplot as plt  # plotting library
import numpy as np  # this module is useful to work with numerical arrays

import tensorboardX
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torchvision.transforms import transforms

from model_defination import Encoder, Decoder, ResNet18
from train_victim import SimpleNet

from ms_ssim import MS_SSIM
from CIFAR10.unlearning.RL_and_GA import save_model





def train_epoch_den(encoder, decoder, dataloader, optimizer, extractor, start_step, start_epoch, epoch):
    writer = tensorboardX.SummaryWriter()
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    loss1 = nn.MSELoss()
    # loss2 = MS_SSIM()
    loss2 = nn.MSELoss()
    loss3 = MS_SSIM()
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 400], gamma=0.2)

    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    global_step=start_step
    for epoch in range(start_epoch, start_epoch + epoch):
        for image_batch, _ in dataloader:  # with "_" we just ignore the labels (the second element of the dataloader tuple)
            # Move tensor to the proper device
            image_batch = image_batch.cuda()
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Evaluate loss
            l1 = loss1(extractor.encode(image_batch), extractor.encode(decoded_data))
            l2 = loss2(image_batch, decoded_data)
            l3 = loss3(image_batch, decoded_data)
            loss = l1 - l3  # 特征距离减小，输出距离减小

            writer.add_scalar('feature_loss', l1, global_step=global_step)
            writer.add_scalar('out_loss', l2, global_step=global_step)
            writer.add_scalar('SSIM', l3, global_step=global_step)
            writer.add_scalar('total_loss', loss, global_step=global_step)


            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Print batch loss

            global_step += 1

            if global_step % 100 == 0:
                print(global_step, 'feature_loss:', np.round(l1.detach().cpu().numpy(), 3),
                      ' out_loss:', np.round(l2.detach().cpu().numpy(),3 ),
                      ' SSIM:', np.round(l3.detach().cpu().numpy(), 3),
                      ' loss:', np.round(loss.detach().cpu().numpy(), 3),
                      )

        if epoch % 15 == 0:
            save_model(encoder, './models/AE/Encoder_ResNet' + str(epoch) + '.pth')
            save_model(decoder, './models/AE/Decoder_ResNet' + str(epoch) + '.pth')

            plot_ae_outputs(encoder, decoder, n=5)

        # scheduler.step()

    save_model(encoder, './models/AE/Encoder_ResNet_SSIM.pth')
    save_model(decoder, './models/AE/Decoder_ResNet_SSIM.pth')
    print('Finished Training')



def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data

def plot_ae_outputs(encoder,decoder,n=5):
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    plt.figure(figsize=(10,4.5))
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_dataset[i][0].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
        rec_img  = decoder(encoder(img))
      plt.imshow(np.transpose(img.cpu().squeeze().numpy(), [1, 2, 0]))
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(np.transpose(rec_img.cpu().squeeze().numpy(), [1, 2, 0]))
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()


