import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from dataloader import KITTILoader as DA
import utils.logger as logger
import torch.backends.cudnn as cudnn
from dataloader.RealToF4Anynet import RealToF4Anynet
from models.anynet_m import AnyNet
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='Anynet fintune on KITTI')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default=None, help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=6,
                    help='batch size for training (default: 6)')
parser.add_argument('--test_bsize', type=int, default=8,
                    help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='results/finetune_anynet',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default=None,
                    help='resume path')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--with_spn',default=False, action='store_true', help='with spn network or not')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
parser.add_argument('--start_epoch_for_spn', type=int, default=121)
parser.add_argument('--pretrained', type=str, default='results/pretrained_anynet/checkpoint.tar',
                    help='pretrained model path')
parser.add_argument('--split_file', type=str, default=None)
parser.add_argument('--evaluate', action='store_true')


args = parser.parse_args()

# if args.datatype == '2015':
#     from dataloader import KITTIloader2015 as ls
# elif args.datatype == '2012':
#     from dataloader import KITTIloader2012 as ls
# elif args.datatype == 'other':
#     from dataloader import diy_dataset as ls


def main():
    global args
    os.makedirs(args.save_path, exist_ok=True)
    log = logger.setup_logger(args.save_path + '/training.log')

    train_data_json = [
        '/workspace/MobileToFDataset/dataset/realtof_train_data.json',
        '/workspace/MobileToFDataset/dataset/realtof_val_data_3d_test.json'
    ]
    # train_data_json = '/workspace/MobileToFDataset/dataset/realtof_train_data.json'
    train_dataset = RealToF4Anynet(
        train_data_json,dsize=[240,180])
    train_dataloader = DataLoader(train_dataset,batch_size= torch.cuda.device_count() * 4,shuffle=True,num_workers=1)
    
    val_data_json =[ '/workspace/MobileToFDataset/dataset/realtof_val_data_3d_test.json']
    val_dataset = RealToF4Anynet(
        val_data_json,dsize=[240,180])
    val_dataloader = DataLoader(val_dataset,batch_size= torch.cuda.device_count() * 4,shuffle=False,num_workers=1)
   
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    model = AnyNet(args)
    model = nn.DataParallel(model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

   
    args.start_epoch = 0
  
    cudnn.benchmark = True
    start_full_time = time.time()
  

    for epoch in range(args.start_epoch, args.epochs):
        log.info('This is {}-th epoch'.format(epoch))
        adjust_learning_rate(optimizer, epoch)

        train(train_dataloader, model, optimizer, log, epoch)

        savefilename = args.save_path + '/checkpoint.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, savefilename)

        if epoch % 1 ==0:
            test(val_dataloader, model, log)

    # test(TestImgLoader, model, log)
    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


def build_input_target(data):
    norm_stero_l = data["stereo_left"]
    norm_stero_r = data["stereo_right"]
    gt_disp = data["disp_left"]
 
    return norm_stero_l,norm_stero_r,gt_disp


def train(dataloader, model, optimizer, log, epoch=0):

    stages = 3 + args.with_spn
    losses = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.train()

    for batch_idx, data in enumerate(dataloader):
        for key in data.keys():
            data[key] = data[key].to(device)

        imgL, imgR, disp_L = build_input_target(data)

        optimizer.zero_grad()
        mask = disp_L > 0
        mask.detach_()
        outputs = model(imgL, imgR)

        if args.with_spn:
            if epoch >= args.start_epoch_for_spn:
                num_out = len(outputs)
            else:
                num_out = len(outputs) - 1
        else:
            num_out = len(outputs)
        # mask = torch.squeeze(mask, 1)
        # outputs = [torch.squeeze(output, 1) for output in outputs]
        
        loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], size_average=True)
                for x in range(num_out)]
        sum(loss).backward()
        optimizer.step()

        for idx in range(num_out):
            losses[idx].update(loss[idx].item())

        if batch_idx % args.print_freq:
            info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(num_out)]
            info_str = '\t'.join(info_str)

            log.info('Epoch{} [{}/{}] {}'.format(
                epoch, batch_idx, length_loader, info_str))
        data.clear()
        torch.cuda.empty_cache()
        
        
    info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
    log.info('Average train loss = ' + info_str)


def test(dataloader, model, log):

    stages = 3 + args.with_spn
    D1s = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.eval()

    for batch_idx, data in enumerate(dataloader):
        for key in data.keys():
            data[key] = data[key].to(device)

        imgL, imgR, disp_L = build_input_target(data)
        
        with torch.no_grad():
            outputs = model(imgL, imgR)
            for x in range(stages):
                output = outputs[x]
                # output = torch.squeeze(outputs[x], 1)
                D1s[x].update(error_estimating(output, disp_L).item())

        info_str = '\t'.join(['Stage {} = {:.4f}({:.4f})'.format(x, D1s[x].val, D1s[x].avg) for x in range(stages)])

        log.info('[{}/{}] {}'.format(
            batch_idx, length_loader, info_str))
        data.clear()
        torch.cuda.empty_cache()
    info_str = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].avg) for x in range(stages)])
    log.info('Average test 3-Pixel Error = ' + info_str)


def error_estimating(disp, ground_truth, maxdisp=192):
    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)

    errmap = torch.abs(disp - gt)
    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    return err3.float() / mask.sum().float()

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = args.lr
    elif epoch <= 400:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    # 设置随机数种子，确保随机数值不变
  
    main()
