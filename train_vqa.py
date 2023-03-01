import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from head.blip_vqa import blip_vqa
import utils
from utils import cosine_lr_schedule, save_result, calculate_metric
from dataset_prep import create_dataset, create_sampler, create_loader, vqa_collate_fn
import wandb 

def train(model, data_loader, optimizer, epoch, device):
    # train
    model.train()  
    

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50    
    
    for i,(image, question, answer, weights, n) in enumerate(data_loader):
        image, weights = image.to(device,non_blocking=True), weights.to(device,non_blocking=True)      

        loss = model(image, question, answer, train=True, n=n, weights=weights)        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    

        wandb.log({"loss": loss.item()})
        wandb.log({'lr':optimizer.param_groups[0]["lr"]})
        
    # gather the stats from all processes


@torch.no_grad()
def evaluation(model, data_loader, device, config) :
    # test
    model.eval()
            
    header = 'Generate VQA test result:'
    print_freq = 50
    
    result = []
    answers = []

    for i, (image, question, answer, weights, n) in enumerate(data_loader):        
        image = image.to(device,non_blocking=True)             

        if config['inference']=='generate':
            answers = model(image, question, train=False, inference='generate') 
            result.extend(answers)
            answers.append(answer)
    
    metric = calculate_metric(answers, result)
    wandb.log({"metric_val":metric})
    return metric


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    #### Dataset #### 
    print("Creating vqa datasets")
    datasets = create_dataset(num_retries=args.num_retries)   
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)         
    else:
        samplers = [None, None]
    
    train_loader, test_loader = create_loader(datasets,samplers,
                                              batch_size=[args.batch_size_train,args.batch_size_test],
                                              num_workers=[4,4],is_trains=[True, False], 
                                              collate_fns=[vqa_collate_fn,None]) 
    #### Model #### 
    print("Creating model")
    model = blip_vqa(pretrained=args.pretrained, filenames=args.filenames, med_config = args.med_config, vit = config['vit'], freeze_vit =args.freeze_vit)

    model = model.to(device)   
    wandb.watch(model, log_freq=100)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    best = 0
    best_epoch = 0 
       
    print("Start training")
    start_time = time.time()    
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, args.max_epoch, config['init_lr'], config['min_lr'])
                
            train_stats = train(model, train_loader, optimizer, epoch, device) 

        else:         
            break        
        
        if utils.is_main_process():     
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")                        
                    
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  

        dist.barrier()         

    vqa_result = evaluation(model_without_ddp, test_loader, device, config)        
    result_file = save_result(vqa_result, args.result_dir, 'vqa_result')  
                      
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
            

if __name__ == '__main__':
    wandb.login(key = '6f14de91cf14f3f40b53951e012cacd8c6e761b0')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='vqa.yaml') 
    parser.add_argument('--output_dir', default='output/VQA')
    parser.add_argument('--evaluate', action='store_true')      
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--num_retries', default=1, type=int)
    parser.add_argument('--med_config', default='sberbank-ai/ruBert-base', type=str)
    parser.add_argument('--max_epoch', default=5, type=int)
    parser.add_argument('--batch_size_train', default=8, type=int)
    parser.add_argument('--batch_size_test', default=8, type=int)
    parser.add_argument('--freeze_vit', default=False)

    parser.add_argument('--filenames',nargs='+')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    wandb.init(config=args)

    main(args, config)
