#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/main.py

import sys, os, torch, shutil, argparse, torch.nn as nn, torch.nn.functional as F
sys.path.append('models')
sys.path.append('utils')
from PC_Augmentation import random_point_dropout, random_scale_point_cloud, random_shift_point_cloud
from ModelNetDataLoader import ModelNetDataLoader
from torch.utils.tensorboard import SummaryWriter
from dgcnn_cls import DGCNN, DGCNN_NRS
from TrainLogger import TrainLogger
from tqdm import tqdm


def cal_loss(pred, gold, smoothing=True):
    """ Calculate cross entropy loss, apply label smoothing if needed. """
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps, n_class = 0.2, pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()  # mean
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


def parse_args():
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')

    parser.add_argument('--gpu', type=str, default='0', help='GPU')
    parser.add_argument('--log_dir', type=str, default='cls_vanilla', help='LOG')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Training Batch Size')
    parser.add_argument('--epoch', type=int, default=250, help='number of training epochs')
    parser.add_argument('--k', type=int, default=20, help='Num of nearest neighbors to use')
    parser.add_argument('--emb_dims', type=int, default=1024, help='Dimension of Embeddings')
    parser.add_argument('--nrs_cfg', type=str, default='dgcnn_cls', help='config for NFL modules')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--num_point', type=int, default=1024, help='num of points of each object')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--model', type=str, default='dgcnn', help='Model to use, [dgcnn, dgcnn_nrs]')
    parser.add_argument('--model_path', type=str, default='', help='Pre-Trained model path, only used in test')
    parser.add_argument('--use_sgd', action='store_true', default=False, help='Use SGD Optimiser[default: True]')
    parser.add_argument('--data_aug', action='store_true', default=False, help='Data Augmentation[default: True]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001, 0.1 if using sgd)')

    return parser.parse_args()


def main(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    ''' === Create Loggers and Backup Scripts === '''
    MyLogger = TrainLogger(args, name=args.model.upper(), subfold='cls')
    writer = SummaryWriter(os.path.join(MyLogger.experiment_dir, 'runs'))
    shutil.copy(os.path.join('models', 'dgcnn_cls.py'), MyLogger.log_dir)
    shutil.copy(os.path.abspath(__file__), MyLogger.log_dir)
    shutil.copy(args.nfl_cfg, MyLogger.log_dir)

    ''' === Load Data (excludes normals) === '''
    MyLogger.logger.info('Load dataset ...')
    DATA_PATH = 'data/modelnet40_normal_resampled/'
    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train', normal_channel=False)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=False)
    train_loader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=6, shuffle=False, num_workers=4)
    # use smaller batch size in test_loader (no effect on training), to make it applicable on a single GTX 1080 (8GB Mem)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===================")
    print("Let's use", torch.cuda.device_count(), "GPUs: %s!" % args.gpu)
    print("===================")

    ''' === Load Models, Allow Multiple GPUs === '''
    if args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    elif args.model == 'dgcnn_nrs':
        model = DGCNN_NRS(args).to(device)
    else:
        raise Exception("Specified Model is Not Implemented")
    model = nn.DataParallel(model)

    try:
        checkpoint = torch.load(MyLogger.savepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        MyLogger.update_from_checkpoints(checkpoint)
    except:
        MyLogger.logger.info('No pre-trained model, start training from scratch...')

    if args.use_sgd:
        print("Use SGD Optimiser")
        opt = torch.optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam Optimiser")
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epoch, eta_min=args.lr)

    for epoch in range(1, args.epoch+1):

        '''=== Train ==='''
        MyLogger.cls_epoch_init()
        scheduler.step()
        model.train()

        for _, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
            points, label = data  # points -> (batch_size, num_points, 3),

            if args.data_aug:
                points = random_point_dropout(points.numpy())
                points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
                points[:, :, 0:3] = random_shift_point_cloud(points[:, :, 0:3])

            points, label = torch.Tensor(points).transpose(2, 1).cuda(), label[:, 0].type(torch.int64).cuda()
            # (batch_size, 3, num_points)
            # batch_size = points.size()[0]  # the last batch is smaller than args.batch_size

            opt.zero_grad()
            logits = model(points)
            loss = cal_loss(logits, label)
            loss.backward()
            opt.step()

            MyLogger.cls_step_update(logits.max(1)[1].cpu().numpy(),
                                     label.long().cpu().numpy(),
                                     loss.cpu().detach().numpy())
        MyLogger.cls_epoch_summary(writer=writer, training=True)

        '''=== Test ==='''
        MyLogger.cls_epoch_init(training=False)
        model.eval()

        for _, data in tqdm(enumerate(test_loader, 0), total=len(test_loader), smoothing=0.9):
            points, label = data
            points, label = points.transpose(2, 1).cuda(), label[:, 0].type(torch.int64).cuda()
            logits = model(points)
            loss = cal_loss(logits, label)
            MyLogger.cls_step_update(logits.max(1)[1].cpu().numpy(),
                                     label.long().cpu().numpy(),
                                     loss.cpu().detach().numpy())
        MyLogger.cls_epoch_summary(writer=writer, training=False)

        if MyLogger.save_model:
            state = {
                'step': MyLogger.step,
                'epoch': MyLogger.best_instance_epoch,
                'instance_acc': MyLogger.best_instance_acc,
                'best_class_acc': MyLogger.best_class_acc,
                'best_class_epoch': MyLogger.best_class_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.state_dict(),
            }
            torch.save(state, MyLogger.savepath)

    MyLogger.cls_train_summary()


if __name__ == "__main__":

    ''' Parse Args for Training'''
    args = parse_args()
    args.nrs_cfg = os.path.join('nrs_cfg', args.nrs_cfg + '.yaml')

    ''' Train the Model'''
    main(args)
