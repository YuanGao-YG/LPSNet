import torch
import torch.nn as nn

# from tensorboardX import SummaryWriter
import numpy as np
import cv2
import time
import os
from LPSNet_Model import *
import utils_train

# classification
import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from discriminator import discriminator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(checkpoint_dir, IsGPU):
    if IsGPU == 1:
        model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
        net = LPSNet()
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids).to(device)
        model.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch']
    else:
        print('here')
        model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar', map_location=torch.device('cpu'))
        net = LPSNet()
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids)
        model.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch']

    return model, optimizer, cur_epoch


def adjust_learning_rate(optimizer, epoch, lr_update_freq):
    if not epoch % lr_update_freq and epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
            print(param_group['lr'])
    return optimizer


def train_psnr(train_in, train_out):
    psnr = utils_train.batch_psnr(train_in, train_out, 1.)
    return psnr


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])


def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])


if __name__ == '__main__':
    checkpoint_dir = './checkpoint/'
    test_dir1 = './dataset/Test'
    result_dir = './result'

    testfiles1 = os.listdir(test_dir1)

    maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    print('> Loading dataset ...')
    model, optimizer, cur_epoch = load_checkpoint(checkpoint_dir,IsGPU=0)

    epoch = 1
    for w in range(epoch):
        for f in range(len(testfiles1)):
            img = cv2.imread(test_dir1 + '/' + testfiles1[f])


            #--------------------------------------------------------------------------------#
            #        判别器
            #---------------------------------------------------------------------------------#
            def main():
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                # device=torch.device('cpu')

                data_transform = transforms.Compose(
                    [transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

                # load image
                img_path = test_dir1 + '/' + testfiles1[f]
                assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
                img = Image.open(img_path)

                plt.imshow(img)
                # [N, C, H, W]
                img = data_transform(img)
                # expand batch dimension
                img = torch.unsqueeze(img, dim=0)

                # read class_indict
                json_path = './class_indices.json'
                assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

                json_file = open(json_path, "r")
                class_indict = json.load(json_file)

                # create model
                model0 = discriminator(num_classes=3).to(device)

                # load model weights
                weights_path = "discriminator.pth"
                assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
                model0.load_state_dict(torch.load(weights_path, map_location='cpu'))

                model0.eval()
                with torch.no_grad():
                    # predict class
                    output = torch.squeeze(model0(img.to(device))).cpu()
                    predict = torch.softmax(output, dim=0)
                    predict_cla = torch.argmax(predict).numpy()
                    print(predict_cla)
                if predict_cla == 1:#low
                    model.eval()
                    with torch.no_grad():
                        img = cv2.imread(test_dir1 + '/' + testfiles1[f])
                        h, w, c = img.shape
                        img_ccc = cv2.resize(img, (512, 512)) / 255.0
                        img_h = hwc_to_chw(img_ccc)
                        input_var = torch.from_numpy(img_h.copy()).type(torch.FloatTensor).unsqueeze(0).to(device)

                        e_out = model(input_var, Type=1) #low
                        print(input_var.shape)
                        e_out = e_out.squeeze().cpu().detach().numpy()
                        e_out = chw_to_hwc(e_out)

                        e_out = cv2.resize(e_out, (w, h))
                        cv2.imwrite(result_dir + '/' + testfiles1[f][:-4] + '_LPSNet.png',
                                    np.clip(e_out * 255, 0.0, 255.0))

                elif predict_cla == 0 or predict_cla == 2:#hazy
                    model.eval()
                    with torch.no_grad():
                        img = cv2.imread(test_dir1 + '/' + testfiles1[f])
                        h, w, c = img.shape
                        img_ccc = cv2.resize(img, (512, 512)) / 255.0
                        img_h = hwc_to_chw(img_ccc)
                        input_var = torch.from_numpy(img_h.copy()).type(torch.FloatTensor).unsqueeze(0).to(device)
                        e_out = model(input_var, Type=2) #hazy
                        print(input_var.shape)
                        e_out = e_out.squeeze().cpu().detach().numpy()
                        e_out = chw_to_hwc(e_out)

                        e_out = cv2.resize(e_out, (w, h))
                        cv2.imwrite(result_dir + '/' + testfiles1[f][:-4] + '_LPSNet.png',
                                    np.clip(e_out * 255, 0.0, 255.0))




            if __name__ == '__main__':
                main()
