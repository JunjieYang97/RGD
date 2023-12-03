import argparse
import torch
import torch.nn as nn
import numpy as np
import csv
import os 
from typing import Callable, Dict, Optional, Sequence, Set, Tuple

from robustbench.utils import load_model
from robustbench.loaders import CustomImageFolder
from Linf_attack import LinfPGDAttack
import torchvision.transforms as transforms
import torch.utils.data as data
from cifar_models import ResNet18
import torchvision.datasets as datasets

from autoattack import AutoAttack


ROOT_PATH = os.path.expanduser("~/.advertorch")
DATA_PATH = os.path.join(ROOT_PATH, "data")

def get_cifar10_test_loader(batch_size, shuffle=False):
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(DATA_PATH, train=False, download=True,
                         transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle)
    loader.name = "cifar10_test"
    return loader

def get_cifar100_test_loader(batch_size, shuffle=False):
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(DATA_PATH, train=False, download=True,
                         transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle)
    loader.name = "cifar100_test"
    return loader

PREPROCESSINGS = {
    'Res256Crop224':
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
}

def load_imagenet(
    batch_size: Optional[int] = 128,
    data_dir: str = './data',
    transforms_test: Callable = PREPROCESSINGS['Res256Crop224']
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    if batch_size > 5000:
        raise ValueError(
            'The evaluation is currently possible on at most 5000 points')

    imagenet = CustomImageFolder(data_dir + '/val', transforms_test)

    test_loader = data.DataLoader(imagenet,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4)

    return test_loader

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def get_model_name(args):
    model_name = str(args.nb_iter)+'_'+str(args.eps)+'_'+str(args.alpha)+'_'+args.update_method+'_'+args.alg+'_'+args.model_name+'_'+args.init_method+'_'+args.dataset
    if args.alg == 'AA':
        model_name = model_name + '_' + str(args.restarts)
        if args.update_method == 'rgd_aa':
            model_name = model_name + '_'+str(args.rgd_iter)
    return model_name

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def main():
    parser = argparse.ArgumentParser(description='PGD/RGD Attack')
    parser.add_argument('--seed', default=0, type=int, help='random seed for PyTorch')
    parser.add_argument('--batch_size', default=128, type=int, help='Evaluation batch size')
    parser.add_argument('--nb_iter', default=7, type=int, help='Number of Inner Iterations')
    parser.add_argument('--eps', default=0.0314, type=float, help='Epsilon Ball size, 0.0314=8/255')
    parser.add_argument('--alpha', default=0.00784, type=float, help='Inner update step size, 0.00784=2/255')
    # noclip_raw refers to rgd, rgd_aa refers to rgd + Autoattack training
    parser.add_argument('--update_method', default='sign_grad', type=str, choices=['sign_grad', 'raw_grad', 'noclip_raw', 
                            'noclip_sign', 'rgd_aa'], help='how we update')
    parser.add_argument('--alg', default='PGD', type=str, choices=['PGD', 'AA'], help='Algorithm we used, PGD or autoattack')
    parser.add_argument('--model_name', default='Wong2020Fast', help='the model imported')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--init_method', default='rand_init', choices=['rand_init', 'zero_init'], help='how we initialize perturbations')
    parser.add_argument('--log_path', default='./log.csv', type=str, help='log file path')
    parser.add_argument('--save', action='store_true', help='used when saving perturbation')

    # parameter used for autoattack algorithm
    parser.add_argument('--restarts', default=1, type=int, help='the restarts for autoattack')
    parser.add_argument('--verbose', action='store_true', help='used when showing detailed output')
    parser.add_argument('--rgd_iter', default=2, type=int, help='rgd iteration in aa')
    
    args = parser.parse_args()
    args.rand_init = True if args.init_method == 'rand_init' else False
    
    print(args)
    torch.manual_seed(args.seed)
    # Load Dataset
    ITER_TOTAL = 0
    if args.dataset == 'cifar10':    
        loader = get_cifar10_test_loader(batch_size=args.batch_size, shuffle=True)
    elif args.dataset == 'cifar100':
        loader = get_cifar100_test_loader(batch_size=args.batch_size, shuffle=True)
    elif args.dataset == 'imagenet':
        loader = load_imagenet(batch_size=args.batch_size, data_dir='./ILSVRC/Data/CLS-LOC')
        # we only consider 5000 validation sample in ImageNet
        assert 5000%args.batch_size==0
        ITER_TOTAL = 5000//args.batch_size
    
    criterion = nn.CrossEntropyLoss().cuda()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    # Load Model
    if args.model_name == 'pgd_resnet18':
        # This model is downloaded from https://github.com/MadryLab/robustness
        filename = "models/pgd_resnet18_cifar10_linf_8.pt"
        model = ResNet18()
        model = nn.parallel.DataParallel(model)
        model.load_state_dict(torch.load(filename)['net'])
    else:
        model = load_model(model_name=args.model_name, dataset=args.dataset, norm='Linf')
    model.to(device)
    model.eval()

   
    # Define algorithm
    if args.alg == 'PGD':
        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="mean"), eps=args.eps,
            nb_iter=args.nb_iter, eps_iter=args.alpha, rand_init=args.rand_init, clip_min=0.0, clip_max=1.0,
            targeted=False, update_method=args.update_method)
    elif args.alg == 'AA':
        adversary = AutoAttack(
            model, norm='Linf', eps=args.eps, attacks_to_run=['apgd-ce'], version='custom',
            device=device, n_iter=args.nb_iter, n_restarts=args.restarts, update_method=args.update_method,
            rand_init = args.rand_init, alpha=args.alpha, verbose=args.verbose, rgd_iter=args.rgd_iter)
    
    pertubations, labels = [], []
    acc_steps, acc_steps_total = None, None
    acc_steps_list = []
    for iter, data_tuple in enumerate(loader):
        cln_data, true_label = data_tuple[0], data_tuple[1]
        if args.dataset == 'imagenet' and iter == ITER_TOTAL:
            break    
        cln_data, true_label = cln_data.to(device), true_label.to(device)
        if args.alg == 'PGD':
            adv_untargeted = adversary.perturb(cln_data, true_label)
        elif args.alg == 'AA':
            adv_untargeted, acc_steps = adversary.run_standard_evaluation(cln_data, true_label, bs=args.batch_size)
            acc_steps_total = torch.cat([acc_steps.cpu(), torch.zeros((args.nb_iter+1, cln_data.shape[0]-acc_steps.shape[1]))], dim=1)
            acc_steps_list.append(acc_steps_total)
        logits = model(adv_untargeted)
        loss = criterion(logits, true_label)
        losses.update(loss.item(), cln_data.size(0))
        acc1, acc5 = accuracy(logits, true_label, topk=(1, 5))
        top1.update(acc1[0].item(), cln_data.size(0))
        if args.save:
            labels.append(true_label)
            pertubations.append(adv_untargeted-cln_data)

    if args.alg == 'AA':
        acc_all = torch.cat(acc_steps_list, dim=1)
        acc_list = torch.mean(acc_all, dim=1).float().tolist()
    model_setting = get_model_name(args)
    if args.save:
        pertu_torch = torch.cat(pertubations, dim=0)
        print('Linf distance:'+str(torch.max(torch.abs(pertu_torch))))
        label_torch = torch.cat(labels, dim=0)
        saved_path = 'save/'+model_setting+ '_' 
        np.save(saved_path+'pertu_'+str(args.seed)+'.npy', pertu_torch.cpu().numpy())
        np.save(saved_path+'labels_'+str(args.seed)+'.npy', label_torch.cpu().numpy())

    print("Sample num: "+str(losses.count))
    print("Accuracy 1: "+str(top1.avg)+"%")

    output_file=args.log_path
    with open(output_file,'a+',newline='') as f:
        writer=csv.writer(f)
        if args.alg == 'AA':
            writer.writerow([model_setting, *acc_list])
        writer.writerow([model_setting, top1.avg])

if __name__ == '__main__':
    main()

