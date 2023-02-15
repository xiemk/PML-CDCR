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


from scipy.io import loadmat, savemat
from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay
from src.models import create_model
from src.loss_functions.losses import AsymmetricLoss
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast

from tensorboardX import SummaryWriter
from dataset import get_COCO2014, generate_noisy_labels, COCO2014_handler, COCO2014_handler_two_augment

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data', metavar='DIR', help='path to dataset', default='/home/algroup/sunfeng/data/coco')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--model-name', default='tresnet_l')
parser.add_argument('--model-path', default='./models/warmup/ema_tresnet_l_15.pth', type=str)
parser.add_argument('--num-classes', default=80)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--thre', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--noise_rate', type=float, default=0.1, 
                    help='corruption rate, should be less than 1')

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
args = parser.parse_args()
args.do_bottleneck_head = False
args.model_path = os.path.join('checkpoint/first_stage/',  str(args.noise_rate), 'ema_tresnet_l_15.pth')
args.alpha = 0.6

Stage = 'second_stage+'
Save_dir = os.path.join('checkpoint', Stage, str(args.noise_rate))
if not os.path.exists(Save_dir):
    os.makedirs(Save_dir)


def main():
    # Setup model
    print('creating model...')
    model = create_model(args)
    model = nn.DataParallel(model)
    model = model.to(DEVICE)

    if args.model_path:  # make sure to load pretrained ImageNet model
        state = torch.load(args.model_path, map_location='cpu')
        filtered_dict = {k[7:]: v for k, v in state.items() if
                         (k[7:] in model.state_dict())}
        model.load_state_dict(filtered_dict, strict=False)

    print('done\n')

    # COCO Data loading
    train_transform_1 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((args.image_size, args.image_size)),
        CutoutPIL(cutout_factor=0.5),
        RandAugment(),
        transforms.ToTensor()])

    train_transform_2 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((args.image_size, args.image_size)),
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

    train_dataset = COCO2014_handler_two_augment(train_images, train_plabels, data_path_train, transform_1=train_transform_1, transform_2=train_transform_2)
    test_dataset = COCO2014_handler(test_images, test_labels, data_path_val, transform=test_transform)

    print("len(val_dataset): ", len(test_dataset))
    print("len(train_dataset): ", len(train_dataset))
    
    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # Actuall Training
    train_multi_label_coco(model, train_loader, test_loader, args.lr, train_labels, train_plabels)

def purity(weights, true_labels, partial_labels):

    weights_tmp = np.zeros_like(weights)
    weights_tmp[true_labels==1] = weights[true_labels==1]
    true_select_num = np.sum(weights_tmp)

    select_or_not = (weights == 1)
    select_or_not[partial_labels == 0] = False
    select_num = np.sum(select_or_not)  

    return true_select_num / select_num, true_select_num, select_num

def confidence_selection(confidences, targets):

    threshold = args.alpha

    weights = np.zeros_like(confidences)
    weights[targets==0] = 1

    confidences_tmp = np.zeros_like(confidences)
    confidences_tmp[targets==1] = confidences[targets==1]

    confidences_true = np.zeros_like(confidences)
    confidences_true[confidences_tmp >= 0] = confidences_tmp[confidences_tmp >= 0]

    confidences_true_sum = np.sum(confidences_true, axis=0)
    # confidences_true_num = np.sum(confidences_true!=0, axis=0)
    confidences_true_num = np.sum(targets!=0, axis=0)
    confidences_true_mean = confidences_true_sum / confidences_true_num
    confidences_mean = np.sum(confidences_true_sum) / np.sum(confidences_true_num)

    confidences_tmp -= confidences_true_mean
    threshold -= confidences_mean

    weights[confidences_tmp >= threshold] = 1

    return weights

def curriculum_disambiguation(model, train_loader, true_labels, partial_labels, epoch=0):
    print('Starting Disambiguation....')

    Sig = torch.nn.Sigmoid()

    confidences = np.zeros_like(true_labels)

    for i, (_, input, _, ind) in enumerate(train_loader):
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.cuda())).cpu()
        
        confidences[ind] = output_regular.cpu().detach()
       
    weights = confidence_selection(confidences, partial_labels)

    pure, true_num, select_num = purity(weights, true_labels, partial_labels)
    print('select purity = {:.2f}, true_select_num = {: >6}, select_num = {: >6}\n'.format(pure, true_num, select_num))

    return weights


def train_multi_label_coco(model, train_loader, val_loader, lr, true_labels, partial_labels):
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = 80
    Stop_epoch = 20
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
        
        model.eval()
        weights = curriculum_disambiguation(model, train_loader, true_labels, partial_labels, epoch=epoch)


        model.train()

        for i, (inputData, _,  target, ind) in enumerate(train_loader):
            inputData = inputData.to(DEVICE)
            target = target.to(DEVICE)
            weight = torch.from_numpy(weights[ind]).to(DEVICE)

            with autocast():  # mixed precision
                output = model(inputData).float()  # sigmoid will be done in loss !

            loss = criterion(output, target, weight)
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
            if regular_flag:
                torch.save(model.state_dict(), os.path.join(
                    Save_dir, 'model_tresnet_l_highest.pth'))
            else:
                torch.save(ema.state_dict(), os.path.join(
                    Save_dir, 'model_tresnet_l_highest.pth'))
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
