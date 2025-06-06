import copy
import csv
import os
import warnings
from argparse import ArgumentParser

import torch
import torchvision
from torchvision.transforms import v2 
#import torchvision.transforms.v2 as v2
import tqdm
import yaml
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")

data_dir = ''

def collate_fn_test(batch): #original
    print(batch)
    images, targets = zip(*batch)
    targets = pd.DataFrame(targets).to_dict(orient="list")
    
    target = {}
    
    if "labels" not in targets:
        for key in ("cls", "box", "idx"):
            target[key] = torch.tensor([])
    else:
        for key in ("labels", "boxes"):
            target[key] = list(map(lambda t: t if isinstance(t, torch.Tensor) else torch.tensor([]), targets[key]))
            target[key] = torch.cat(target[key], dim=0)
            
        target["idx"] = torch.tensor([list(map(lambda t: t if isinstance(t, torch.Tensor) else torch.tensor([]), target["labels"]))]) # or torch.cat(list) or torch.tensor([list])
        target["cls"] = target.pop("labels")
        target["box"] = target.pop("boxes")
    
    samples = torch.stack(images, dim=0)

    return images, target

def fix_targets(targets): #original
    fixed_targets = targets
    
    # fixed_targets["idx"] = torch.tensor([list(map(lambda t: t if isinstance(t, torch.Tensor) else torch.tensor([]), targets["idx"][0]))])
    
    #fixed_targets["idx"] = torch.tensor([list(map(lambda t: torch.arange(t.size(0)) if isinstance(t, torch.Tensor) else torch.tensor([]), targets["idx"][0]))]) #if len(t.shape) > 0
    fixed_targets["idx"] = torch.arange(targets["labels"][0].size(0))
    fixed_targets["box"] = targets["box"][0]
    return fixed_targets
    

def get_sampler_split(dataset, ratio, seed = 42, shuffle = False): #new
    import numpy as np
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(ratio * dataset_size))
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
        
    _, split_indices = indices[split:], indices[:split]
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    
    return sampler
    
def compose_transforms(inference = False): #new
    if inference:
        composed_transforms = torchvision.transforms.v2.Compose(
                                [
                                    torchvision.transforms.v2.ToImage(),
                                    #PRZEKSZTAŁCENIE NA TENSOR TYPU OBRAZOWEGO (TZW. IMAGE TENSOR)
                            
                                    torchvision.transforms.v2.ConvertImageDtype(torch.float32),
                                    #ZMIANA TYPU DANYCH ELEMENTÓW TENSORA
                            
                                    torchvision.transforms.v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    #NORMALIZACJA DANYCH TENSORA
                                    torchvision.transforms.v2.CenterCrop([640, 640])
                                ]
                                )
    else:
        composed_transforms = torchvision.transforms.v2.Compose(
                                [
                                    torchvision.transforms.v2.ToImage(), #PRZEKSZTAŁCENIE NA TENSOR TYPU OBRAZOWEGO (TZW. IMAGE TENSOR)
                                    torchvision.transforms.v2.ConvertImageDtype(torch.float32), #ZMIANA TYPU DANYCH ELEMENTÓW TENSORA
                                    torchvision.transforms.v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    torchvision.transforms.v2.RandomHorizontalFlip(),
                                    torchvision.transforms.v2.RandomVerticalFlip(),
                                    torchvision.transforms.v2.RandomVerticalFlip(),
                                    torchvision.transforms.v2.CenterCrop([640, 640])
                                ]
                                )
    
    return composed_transforms
    
def get_dataset(img_path, anno_path, inference = False, wrap = False): #new
    transforms = compose_transforms(inference)
    dataset = torchvision.datasets.CocoDetection(img_path, anno_path, transforms)
    if wrap:
        dataset = torchvision.datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=("boxes", "labels"))
        #dataset = torchvision.datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=("boxes", "labels", "image_id", "bbox", "category_id", "image_id"))
        
    return dataset

def train(args, params):
    # Model
    model = nn.yolo_v11_n(len(params['names']))
    model.cuda()

    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    optimizer = torch.optim.SGD(util.set_params(model, params['weight_decay']),
                                params['min_lr'], params['momentum'], nesterov=True)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    # filenames = []
    # with open(f'{data_dir}/train2017.txt') as f:
    #     for filename in f.readlines():
    #         filename = os.path.basename(filename.rstrip())
    #         filenames.append(f'{data_dir}/images/train2017/' + filename)

    # sampler = None
    # dataset = Dataset(filenames, args.input_size, params, augment=True)

    img_path = data_dir + "/images" + "/train2017" #new
    anno_path = data_dir + "/annotations" + "/instances_train2017.json" #new
    dataset = get_dataset(img_path, anno_path, inference = False, wrap = True) #new
    shuffling = args.shuffle

    sampler = None
    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)
        loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler,
                             num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)
    else:
        if args.tsplit:
            sampler = get_sampler_split(dataset, args.tratio, shuffling)
            shuffling = False
            
        loader = data.DataLoader(dataset, args.batch_size, sampler = sampler, shuffle = shuffling,
                    num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)

    
    # loader = data.DataLoader(dataset, args.batch_size, sampler is None if args.distributed else sampler = sampler, sampler is None if arg, shuffle = shuffling,
    #                          num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)

    # Scheduler
    num_steps = len(loader)
    scheduler = util.LinearLR(args, params, num_steps)

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    best = 0
    amp_scale = torch.amp.GradScaler()
    criterion = util.ComputeLoss(model, params)

    with open('weights/step.csv', 'w') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch',
                                                     'box', 'cls', 'dfl',
                                                     'Recall', 'Precision', 'mAP@50', 'mAP'])
            logger.writeheader()

        for epoch in range(args.epochs):
            model.train()
            if args.distributed:
                sampler.set_epoch(epoch)
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            p_bar = enumerate(loader)

            if args.local_rank == 0:
                print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'box', 'cls', 'dfl'))
                p_bar = tqdm.tqdm(p_bar, total=num_steps)

            optimizer.zero_grad()
            avg_box_loss = util.AverageMeter()
            avg_cls_loss = util.AverageMeter()
            avg_dfl_loss = util.AverageMeter()
            for i, (samples, targets) in p_bar:

                step = i + num_steps * epoch
                scheduler.step(step, optimizer)

                samples = samples.cuda().float() / 255

                # Forward
                with torch.amp.autocast('cuda'):
                    outputs = model(samples)  # forward
                    #print(f'train outputs: {outputs}')
                    loss_box, loss_cls, loss_dfl = criterion(outputs, targets)

                avg_box_loss.update(loss_box.item(), samples.size(0))
                avg_cls_loss.update(loss_cls.item(), samples.size(0))
                avg_dfl_loss.update(loss_dfl.item(), samples.size(0))

                loss_box *= args.batch_size  # loss scaled by batch_size
                loss_cls *= args.batch_size  # loss scaled by batch_size
                loss_dfl *= args.batch_size  # loss scaled by batch_size
                loss_box *= args.world_size  # gradient averaged between devices in DDP mode
                loss_cls *= args.world_size  # gradient averaged between devices in DDP mode
                loss_dfl *= args.world_size  # gradient averaged between devices in DDP mode

                # Backward
                amp_scale.scale(loss_box + loss_cls + loss_dfl).backward()

                # Optimize
                if step % accumulate == 0:
                    # amp_scale.unscale_(optimizer)  # unscale gradients
                    # util.clip_gradients(model)  # clip gradients
                    amp_scale.step(optimizer)  # optimizer.step
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                torch.cuda.synchronize()

                # Log
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
                    s = ('%10s' * 2 + '%10.3g' * 3) % (f'{epoch + 1}/{args.epochs}', memory,
                                                       avg_box_loss.avg, avg_cls_loss.avg, avg_dfl_loss.avg)
                    p_bar.set_description(s)

            if args.local_rank == 0:
                # mAP
                last = test(args, params, ema.ema)

                logger.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'box': str(f'{avg_box_loss.avg:.3f}'),
                                 'cls': str(f'{avg_cls_loss.avg:.3f}'),
                                 'dfl': str(f'{avg_dfl_loss.avg:.3f}'),
                                 'mAP': str(f'{last[0]:.3f}'),
                                 'mAP@50': str(f'{last[1]:.3f}'),
                                 'Recall': str(f'{last[2]:.3f}'),
                                 'Precision': str(f'{last[3]:.3f}')})
                log.flush()

                # Update best mAP
                if last[0] > best:
                    best = last[0]

                # Save model
                save = {'epoch': epoch + 1,
                        'model': copy.deepcopy(ema.ema)}

                # Save last, best and delete
                torch.save(save, f='./weights/last.pt')
                if best == last[0]:
                    torch.save(save, f='./weights/best.pt')
                del save

    if args.local_rank == 0:
        util.strip_optimizer('./weights/best.pt')  # strip optimizers
        util.strip_optimizer('./weights/last.pt')  # strip optimizers


@torch.no_grad()
def test(args, params, model=None):
    # filenames = []
    # with open(f'{data_dir}/val2017.txt') as f:
    #     for filename in f.readlines():
    #         filename = os.path.basename(filename.rstrip())
    #         filenames.append(f'{data_dir}/images/val2017/' + filename)

    # dataset = Dataset(filenames, args.input_size, params, augment=False)

    img_path = data_dir + "/images" + "/val2017" #new
    anno_path = data_dir + "/annotations" + "/instances_val2017.json" #new
    dataset = get_dataset(img_path, anno_path, inference = True, wrap = True) #new
    
    sampler = None
    if args.vsplit:
        sampler = get_sampler_split(dataset, args.vratio)
            
    # loader = data.DataLoader(dataset, args.batch_size, sampler = sampler, shuffle = shuffling,
    #                 num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)
    
    loader = data.DataLoader(dataset, batch_size=4, sampler = sampler, shuffle=False, num_workers=4,
                             pin_memory=True, collate_fn=Dataset.collate_fn)

    plot = False
    if not model:
        plot = True
        model = torch.load(f='./weights/best.pt', map_location='cuda')
        model = model['model'].float().fuse()

    model.half()
    model.eval()

    # Configure
    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0
    m_rec = 0
    map50 = 0
    mean_ap = 0
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP'))
    for batch in p_bar:
        #samples, targets = collate_fn_test(batch)
        
        samples, targets = batch
        print(f"test samples before stacking: {samples}") #delete
        samples = torch.stack(samples, dim = 0) #?
        print(f"test samples after stacking, before cuda: {samples}") #delete
        print(f"targets at first: {targets}") #delete
        #targets = fix_targets(targets) 
        samples = samples.cuda()
        print(f"samples after cuda: {samples}") #delete
        samples = samples.half()  # uint8 to fp16/32
        print(f"samples after half: {samples}") #delete
        samples = samples / 255.  # 0 - 255 to 0.0 - 1.0
        print(f"samples after /255: {samples}") #delete
        _, _, h, w = samples.shape  # batch-size, channels, height, width
        scale = torch.tensor((w, h, w, h)).cuda()
        # Inference
        outputs = model(samples)
        print(f"all outputs here: {outputs}") #delete
        # NMS
        outputs = util.non_max_suppression(outputs)
        # Metrics
        for i, output in enumerate(outputs):
            print(f"to output: {output}")  #delete
            idx = targets['idx'] == i
            print(f"to idx: {idx}")  #delete
            cls = targets['cls'][idx]
            print(f"to cls: {cls}")  #delete
            box = targets['box'][idx]
            print(f"to box: {box}")  #delete

            cls = cls.cuda()
            box = box.cuda()

            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                continue
            # Evaluate
            if cls.shape[0]:
                target = torch.cat(tensors=(cls, util.wh2xy(box) * scale), dim=1)
                metric = util.compute_metric(output[:, :6], target, iou_v)
            # Append
            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

    # Compute metrics
    print(f'metrics tut: {metrics}')
    metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics, plot=plot, names=params["names"])
    # Print results
    print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))
    # Return results
    model.float()  # for training
    return mean_ap, map50, m_rec, m_pre


def profile(args, params):
    import thop
    shape = (1, 3, args.input_size, args.input_size)
    model = nn.yolo_v11_n(len(params['names'])).fuse()

    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(model, inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[2 * flops, num_params], format="%.3f")

    if args.local_rank == 0:
        print(f'Number of parameters: {num_params}')
        print(f'Number of FLOPs: {flops}')


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--tsplit', action='store_true')
    parser.add_argument('--vsplit', action='store_true')
    parser.add_argument('--tratio', default=0.05, type=float)
    parser.add_argument('--vratio', default=0.05, type=float)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--direc', type=str, help='Path to the input directory.')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    global data_dir
    data_dir = args.direc
    
    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    with open('utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    util.setup_seed()
    util.setup_multi_processes()

    profile(args, params)

    if args.train:
        train(args, params)
    if args.test:
        test(args, params)

    # Clean
    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
