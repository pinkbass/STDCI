import os
import os.path as osp
import sys 
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)

from stci.datasets.builder import build_dataset 
from stci.models.builder import build_model
from stci.utils.optim_builder import  build_optimizer
from stci.utils.loss_builder import build_loss
from torch.utils.data import DataLoader
from stci.utils.mask import generate_BlockRandomST_masks
from stci.utils.config import Config
from stci.utils.logger import Logger
from stci.utils.utils import save_image, load_checkpoints, get_device_info, st_modulation_stcs
from stci.utils.eval import eval_psnr_ssim_STCI_NonBlock

import torch
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, MultiStepLR

# try:
#     import lpips
# except Exception as e:
#     lpips = None

import time
import argparse 
import json 
import einops
import datetime

#  CUDA_VISIBLE_DEVICES="1"

# 定义相关参数
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str,default="configs/STCS_swintransformer/STCI_3dswintransformerConv.py")
    parser.add_argument("--work_dir",type=str)
    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument("--distributed",type=bool,default=False)
    parser.add_argument("--resume",type=str,default=None)
    parser.add_argument("--local_rank",default=-1)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    local_rank = int(args.local_rank) 
    if args.distributed:
        args.device = torch.device("cuda",local_rank)
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if args.work_dir is None:
        args.work_dir = osp.join('work_dirs',osp.splitext(osp.basename(args.config))[0],'ver_E224L14P3D',now) 
        # args.work_dir = osp.join('work_dirs', 'ver_test')     
        # work_dirs/STCI_3dswintransformerConv/ver_ablation_GroupConv

    if args.resume is not None:
        cfg.resume = args.resume

    log_dir = osp.join(args.work_dir,"log")
    show_dir = osp.join(args.work_dir,"show")
    train_image_save_dir = osp.join(args.work_dir,"train_images")
    checkpoints_dir = osp.join(args.work_dir,"checkpoints")

    if not osp.exists(args.work_dir):
        os.makedirs(args.work_dir)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    if not osp.exists(show_dir):
        os.makedirs(show_dir)
    if not osp.exists(train_image_save_dir):
        os.makedirs(train_image_save_dir)
    if not osp.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    logger = Logger(log_dir)
    writer = SummaryWriter(log_dir = show_dir)

    rank = 0                                      # 分布式训练参数
    if args.distributed:
        local_rank = int(args.local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()                   

    dash_line = '-' * 80 + '\n'
    device_info = get_device_info()
    env_info = '\n'.join(['{}: {}'.format(k,v) for k, v in device_info.items()])
    
    device = args.device
    model = build_model(cfg.model).to(device)
    if cfg.checkpoints is not None:
            logger.info("Load pre_train model...")
            model=torch.load(cfg.checkpoints)

    if rank==0:
        logger.info('GPU info:\n' 
                + dash_line + 
                env_info + '\n' +
                dash_line)
        logger.info('cfg info:\n'
                + dash_line + 
                json.dumps(cfg, indent=4)+'\n'+
                dash_line) 
        logger.info('Model info:\n'
                + dash_line + 
                str(model)+'\n'+
                dash_line)

    mask, mask_sum, mask_position = generate_BlockRandomST_masks(cfg.train_data.mask_blocksize, cfg.train_data.mask_path,cfg.train_data.mask_shape, cfg.train_data.s_cr)
    logger.info('mask cut coordinate:\n' + str(mask_position))
    train_data = build_dataset(cfg.train_data,{"mask":mask})
    if cfg.eval.flag:
        test_data = build_dataset(cfg.test_data,{"mask":mask})
    if args.distributed:
        dist_sampler = DistributedSampler(train_data,shuffle=True)
        train_data_loader = DataLoader( dataset=train_data, 
                                        batch_size=cfg.data.samples_per_gpu,
                                        sampler=dist_sampler,
                                        num_workers = cfg.data.workers_per_gpu)
    else:
        train_data_loader = DataLoader( dataset=train_data, 
                                        batch_size=cfg.data.samples_per_gpu,
                                        shuffle=True,
                                        num_workers = cfg.data.workers_per_gpu)
        
    init_lr = cfg.Lr.init_lr
    final_lr = cfg.Lr.final_lr
    warmup_epochs = 3
    total_cosine_epochs = 200

    optimizer = build_optimizer(cfg.optimizer,{"params":model.parameters()})        

    # 设置学习率调节器    
    # scheduler = MultiStepLR(optimizer, milestones=[100,150], gamma=0.1, last_epoch=-1, verbose=False)  
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_cosine_epochs-warmup_epochs, eta_min=final_lr)
    
    criterion = build_loss(cfg.loss)
    criterion = criterion.to(args.device)

    # 其他损失权重
    # lambda_td = getattr(cfg, 'lambda_td', 0.05)
    # lambda_lpips = getattr(cfg, 'lambda_lpips', 0.05)

    # lpips_model = None
    # if lambda_lpips > 0:
    #     if lpips is None:
    #         raise ImportError("未安装 lpips 库，请先运行: pip install lpips")
    #     lpips_model = lpips.LPIPS(net='alex').to(args.device)
    #     lpips_model.eval()
    
    start_epoch = 0
    if rank==0:
        if cfg.checkpoints is not None:
            logger.info("Load pre_train model...")
            resume_dict = torch.load(cfg.checkpoints)
            if "model_state_dict" not in resume_dict.keys():
                model_state_dict = resume_dict
            else:
                model_state_dict = resume_dict["model_state_dict"]
            load_checkpoints(model,model_state_dict)
        else:            
            logger.info("No pre_train model")

        if cfg.resume is not None:
            logger.info("Load resume...")
            resume_dict = torch.load(cfg.resume)
            start_epoch = resume_dict["epoch"]
            model_state_dict = resume_dict["model_state_dict"]
            load_checkpoints(model,model_state_dict)
            optim_state_dict = resume_dict["optim_state_dict"]
            # optim_state_dict['param_groups'][0]['lr']=0.0000005     # 在训练模型加载的时候修改学习率
            optimizer.load_state_dict(optim_state_dict)

    
    if args.distributed:
        model = DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
    
    iter_num = len(train_data_loader) 
    best_psnr=0
    for epoch in range(start_epoch,cfg.runner.max_epochs):
        epoch_loss = 0
        model = model.train()
        if epoch < warmup_epochs:                     # 手动设置warmup阶段的学习率
            current_lr = init_lr * (epoch+1)/warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        start_time = time.time()
        accumulation_steps = 2
        optimizer.zero_grad()
        for iteration, data in enumerate(train_data_loader):
            gt, meas = data                                                   # gt[b,d,H,W]/meas[b,h,w]
            gt = gt.float().to(args.device)
            meas = meas.unsqueeze(1).float().to(args.device)
            batch_size = meas.shape[0]

            Phi_s = einops.repeat(mask_sum,'h w->b 1 h w',b=batch_size)
            Phi_st = einops.repeat(mask, 'cr h w->b cr h w',b=batch_size) 
            Phi_s = torch.from_numpy(Phi_s).to(args.device)                         # [b 1 64 64]
            Phi_st = torch.from_numpy(Phi_st).to(args.device)                       # [b 8 256 256]           
   

            input_x = (meas, Phi_st, Phi_s)
            model_out = model(input_x)

            if not isinstance(model_out,list):
                model_out = [model_out]
            pred_seq = model_out[-1].squeeze(1)
            loss_re = criterion(pred_seq, gt)                 # pred_seq=[b,n,h,w]
            loss_cs = criterion(st_modulation_stcs(pred_seq, Phi_st, Phi_s, s_cr=cfg.train_data.s_cr), meas)

            # # 帧间差异损失（时间梯度一致性）
            # loss_td = torch.tensor(0.0, device=args.device)
            # if lambda_td > 0:
            #     if pred_seq.shape[1] > 1:
            #         pred_dt = pred_seq[:, 1:, :, :] - pred_seq[:, :-1, :, :]
            #         gt_dt = gt[:, 1:, :, :] - gt[:, :-1, :, :]
            #         loss_td = F.l1_loss(pred_dt, gt_dt)

            # # LPIPS 感知损失
            # loss_lpips = torch.tensor(0.0, device=args.device)
            # if lambda_lpips > 0 and lpips_model is not None:
            #     b, t, h, w = pred_seq.shape
            #     pred_img = pred_seq.reshape(b * t, 1, h, w)
            #     gt_img = gt.reshape(b * t, 1, h, w)
            #     # 归一化到 [-1,1] 并扩展到 3 通道
            #     pred_img = pred_img.clamp(0, 1) * 2 - 1
            #     gt_img = gt_img.clamp(0, 1) * 2 - 1
            #     pred_img = pred_img.repeat(1, 3, 1, 1)
            #     gt_img = gt_img.repeat(1, 3, 1, 1)
            #     loss_lpips = lpips_model(pred_img, gt_img).mean()

            # loss = 0.9*loss_re + 0.01*loss_cs + lambda_td*loss_td + lambda_lpips*loss_lpips
            loss = 0.9*loss_re + 0.01*loss_cs
            # loss = loss_re
            # loss = 0.9*loss_re + 0.001*loss_cs
            # loss = torch.sqrt(criterion(model_out[-1].squeeze(1), gt))
            loss = loss/accumulation_steps
            epoch_loss += loss.item()

            loss.backward()
            # optimizer.step()
            if (iteration + 1) % accumulation_steps == 0 or (iteration + 1) == len(train_data_loader):
                optimizer.step()
                optimizer.zero_grad() # 清空梯度，释放显存

            if rank==0 and (iteration % cfg.log_config.interval) == 0:
                # lr = optimizer.state_dict()["param_groups"][0]["lr"]
                lr = optimizer.param_groups[0]['lr']
                iter_len = len(str(iter_num))
                logger.info("epoch: [{}][{:>{}}/{}], lr: {:.6f}, loss: {:.5f}, loss_re: {:.5f}, loss_cs: {:.5f}.".format(epoch,iteration,iter_len,iter_num,lr,loss.item(), loss_re.item(), loss_cs.item()))
                writer.add_scalar("Losses/loss",loss.item(),epoch*len(train_data_loader) + iteration)
                writer.add_scalar("Losses/loss_re",loss_re.item(),epoch*len(train_data_loader) + iteration)
                writer.add_scalar("Losses/loss_cs",loss_cs.item(),epoch*len(train_data_loader) + iteration)
                # writer.add_scalar("Losses/loss_td",float(loss_td.item()),epoch*len(train_data_loader) + iteration)
                # writer.add_scalar("Losses/loss_lpips",float(loss_lpips.item()),epoch*len(train_data_loader) + iteration)

        end_time = time.time()  

        # scheduler.step()
        if epoch > warmup_epochs:
            cosine_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            # current_lr = 0.0000005

        if rank==0:
            logger.info("epoch: {}, avg_loss: {:.5f}, time: {:.2f}s.\n".format(epoch,epoch_loss/(iteration+1),end_time-start_time))

        if rank==0 and (epoch % cfg.checkpoint_config.interval) == 0:
            if args.distributed:
                save_model = model.module
            else:
                save_model = model
            checkpoint_dict = {
                "epoch": epoch, 
                "model_state_dict": save_model.state_dict(), 
                "optim_state_dict": optimizer.state_dict(), 
            }
            # torch.save(checkpoint_dict,osp.join(checkpoints_dir,"epoch_"+str(epoch)+".pth")) 
            torch.save(checkpoint_dict,osp.join(checkpoints_dir,"epoch_latest.pth")) 

        if rank==0:
        # if rank==0 and cfg.eval.flag and epoch % cfg.eval.interval==0:
            if args.distributed:
                psnr_dict,ssim_dict = eval_psnr_ssim_STCI_NonBlock(model.module,test_data,mask, mask_sum,args)
            else:
                psnr_dict,ssim_dict = eval_psnr_ssim_STCI_NonBlock(model,test_data,mask, mask_sum, args)

            psnr_str = ", ".join([key+": "+"{:.4f}".format(psnr_dict[key]) for key in psnr_dict.keys()])
            ssim_str = ", ".join([key+": "+"{:.4f}".format(ssim_dict[key]) for key in ssim_dict.keys()])
            logger.info("Mean PSNR: \n{}.\n".format(psnr_str))
            logger.info("Mean SSIM: \n{}.\n".format(ssim_str))
            psnr_epoch = psnr_dict['psnr_mean']
            if psnr_epoch>best_psnr:
                torch.save(checkpoint_dict, osp.join(checkpoints_dir,"best_psnr.pth"))

if __name__ == '__main__':
    main()


