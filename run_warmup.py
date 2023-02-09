import os
import argparse
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay
from src.models import create_model
from src.loss_functions.losses import AsymmetricLoss
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast

from dataset import get_COCO2014, generate_noisy_labels, COCO2014_handler

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data', metavar='DIR', help='path to dataset', default='/home/algroup/sunfeng/data/coco')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--model-name', default='tresnet_l')
parser.add_argument('--model-path', default='./models_local/tresnet_l.pth', type=str)
parser.add_argument('--num-classes', default=80)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--thre', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--noise_rate', type=float, default=0.2, 
                    help='corruption rate, should be less than 1')

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
args = parser.parse_args()
args.do_bottleneck_head = False

Stage = 'first_stage'
Save_dir = os.path.join('checkpoint', Stage, str(args.noise_rate))
if not os.path.exists(Save_dir):
    os.makedirs(Save_dir)

def main():

    # Setup model
    print('creating model...')
    # model = create_model(args).cuda()
    model = create_model(args)
    model = nn.DataParallel(model)
    model = model.to(DEVICE)

    if args.model_path:  # make sure to load pretrained ImageNet model
        state = torch.load(args.model_path, map_location='cpu')
        filtered_dict = {'module.' + k: v for k, v in state['model'].items() if
                         ('module.' + k in model.state_dict() and 'head.fc' not in k)}
       
        model.load_state_dict(filtered_dict, strict=False)
    print('done\n')

    # COCO Data loading
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((args.image_size, args.image_size)),
        CutoutPIL(cutout_factor=0.5),
        RandAugment(),
        transforms.ToTensor()])
    
    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()])
    
    instances_path_val = os.path.join(args.data, 'val_anno.json')
    instances_path_train = os.path.join(args.data, 'train_anno.json')

    data_path_val   = f'{args.data}/val2014'    # args.data
    data_path_train = f'{args.data}/train2014'  # args.data


    train_images, train_labels, test_images, test_labels = get_COCO2014(instances_path_train, instances_path_val)
    train_plabels = generate_noisy_labels(train_labels, noise_level=args.noise_rate)

    train_dataset = COCO2014_handler(train_images, train_plabels, data_path_train, transform=train_transform)
    test_dataset = COCO2014_handler(test_images, test_labels, data_path_val, transform=test_transform)

    print("len(val_dataset)): ", len(test_dataset))
    print("len(train_dataset)): ", len(train_dataset))
    
    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # Actuall Training
    train_multi_label_coco(model, train_loader, test_loader, args.lr)


def train_multi_label_coco(model, train_loader, val_loader, lr):
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = 80
    Stop_epoch = 15
    weight_decay = 1e-4
    criterion = AsymmetricLoss(gamma_neg=0, gamma_pos=0, clip=0, disable_torch_grad_focal_loss=False)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)

    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    for epoch in range(Epochs):
        if epoch > Stop_epoch:
            break
        for i, (inputData, target, _) in enumerate(train_loader):
            inputData = inputData.to(DEVICE)
            target = target.to(DEVICE)
            # target = target.max(dim=1)[0]
            with autocast():  # mixed precision
                output = model(inputData).float()  # sigmoid will be done in loss !

            loss = criterion(output, target)
            model.zero_grad()

            scaler.scale(loss).backward()
            # loss.backward()

            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            scheduler.step()

            ema.update(model)
            # store information
            if i % 100 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss.item()))
    
        model.eval()
        mAP_score, regular_flag = validate_multi(val_loader, model, ema)
        model.train()
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            if regular_flag is True:
                torch.save(model.state_dict(), os.path.join(
                    Save_dir, 'model_tresnet_l_highest.pth'))
            else:
                torch.save(ema.state_dict(), os.path.join(
                    Save_dir, 'model_tresnet_l_highest.pth'))

        
        torch.save(model.state_dict(), os.path.join(
                Save_dir, 'model_tresnet_l_{}.pth'.format(epoch)))
    
        torch.save(ema.state_dict(), os.path.join(
                Save_dir, 'ema_tresnet_l_{}.pth'.format(epoch))) 

        print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))

def validate_multi(val_loader, model, ema_model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for i, (input, target, _) in enumerate(val_loader):
        target = target
        # target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.cuda())).cpu()
                output_ema = Sig(ema_model.module(input.cuda())).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    return max(mAP_score_regular, mAP_score_ema), mAP_score_regular > mAP_score_ema


if __name__ == '__main__':
    main()
