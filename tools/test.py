import os
import os.path as osp
import sys 
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)
import torch 
from torch.utils.data import DataLoader
from stci.utils.mask import generate_BlockRandomST_masks
from stci.utils.utils import save_single_image,get_device_info,load_checkpoints
from stci.utils.metrics import compare_psnr,compare_ssim
from stci.utils.config import Config
from stci.models.builder import build_model
from stci.datasets.builder import build_dataset 
from stci.utils.logger import Logger
from torch.cuda.amp import autocast
import numpy as np 
import argparse 
import time
import einops 

CUDA_VISIBLE_DEVICES="1"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str, default="configs/STCS_swintransformer/test_STCI_3dswintransformerConv.py")
    parser.add_argument("--work_dir",type=str)
    parser.add_argument("--weights",type=str)
    parser.add_argument("--device",type=str,default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device="cpu"
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    device = args.device
    config_name = osp.splitext(osp.basename(args.config))[0]
    if args.work_dir is None:
        args.work_dir = osp.join('./test_log',config_name)

    mask, mask_sum, mask_position = generate_BlockRandomST_masks(cfg.test_data.mask_blocksize, cfg.test_data.mask_path,cfg.test_data.mask_shape, cfg.test_data.s_cr)
    cr = mask.shape[0]

    test_dir = osp.join(args.work_dir,"test_images")
    log_dir = osp.join(args.work_dir,"test_log")    
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    logger = Logger(log_dir)

    dash_line = '-' * 80 + '\n'
    device_info = get_device_info()
    env_info = '\n'.join(['{}: {}'.format(k,v) for k, v in device_info.items()])
    logger.info('GPU info:\n' 
            + dash_line + 
            env_info + '\n' +
            dash_line) 
    
    test_data = build_dataset(cfg.test_data,{"mask":mask})
    data_loader = DataLoader(test_data,batch_size=1,shuffle=False)

    model = build_model(cfg.model).to(device)
    logger.info("Load pre_train model...")
    resume_dict = torch.load(cfg.checkpoints)
    if "model_state_dict" not in resume_dict.keys():
        model_state_dict = resume_dict
    else:
        model_state_dict = resume_dict["model_state_dict"]
    load_checkpoints(model,model_state_dict,strict=True)

    Phi_s = einops.repeat(mask_sum,'h w->b 1 h w',b=1)
    Phi = einops.repeat(mask, 'cr h w->b cr h w',b=1) 
    Phi_s = torch.from_numpy(Phi_s).to(args.device)                         
    Phi = torch.from_numpy(Phi).to(args.device) 
      
    psnr_dict,ssim_dict = {},{}
    psnr_list,ssim_list = [],[]
    sum_time=0.0
    time_count = 0
    cr, patch_H, patch_W = mask.shape
    s_cr = cfg.test_data.s_cr
    
    for data_iter,data in enumerate(data_loader):
        psnr,ssim = 0,0
        batch_output = []
        meas, gt = data
        gt = gt[0].numpy()
        meas = meas[0].float().to(device)
        H,W = gt.shape[-2], gt.shape[-1]
        n_h = H // patch_H                        
        n_w = W // patch_W
        patch_h = patch_H // s_cr                  
        patch_w = patch_W // s_cr
        batch_size = meas.shape[0]
        
        name = test_data.data_name_list[data_iter]
        if "_" in name:
            _name,_ = name.split("_")
        else:
            _name,_ = name.split(".")

        for ii in range(batch_size):               
            single_meas = meas[ii].unsqueeze(0).unsqueeze(0)
            x_recon = torch.zeros(cr,H,W).to(args.device)
            with torch.no_grad():
                torch.cuda.synchronize()
                start = time.time()
                for h_id in range(n_h):            
                    for w_id in range(n_w):
                        mea_block = single_meas[:1,:1,h_id*patch_h:(h_id+1)*patch_h, w_id*patch_w:(w_id+1)*patch_w].to(args.device)
                        input_x = (mea_block, Phi, Phi_s)
                        outputs_backward = model(input_x)    
                        if not isinstance(outputs_backward,list):
                            outputs_backward = [outputs_backward]
                        output = outputs_backward[-1][0]
                        x_recon[:,h_id*patch_H:(h_id+1)*patch_H, w_id*patch_W:(w_id+1)*patch_W] = output
                end = time.time()
                run_time = end - start
                if ii>0:
                    sum_time += run_time
                    time_count += 1
            x_recon = x_recon.cpu().numpy()                                   
            batch_output.append(x_recon)                                       
            for jj in range(cr):
                if x_recon.shape[0]==3:
                    per_frame_out = x_recon[:,jj]
                    per_frame_out = np.sum(per_frame_out*test_data.rgb2raw,axis=0)
                else:
                    per_frame_out = x_recon[jj,:,:1920]
                per_frame_gt = gt[ii,jj, :, :1920]
                psnr += compare_psnr(per_frame_gt*255,per_frame_out*255)
                ssim += compare_ssim(per_frame_gt*255,per_frame_out*255)
        
        meas_num = len(batch_output)              
        psnr = psnr / (meas_num* cr)
        ssim = ssim / (meas_num* cr)
        logger.info("{}, Mean PSNR: {:.4f} Mean SSIM: {:.4f}.".format(
                    _name,psnr,ssim))
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        psnr_dict[_name] = psnr
        ssim_dict[_name] = ssim
        #save image
        out = np.array(batch_output)
        for j in range(out.shape[0]):
            image_dir = osp.join(test_dir,_name)
            if not osp.exists(image_dir):
                os.makedirs(image_dir)
            save_single_image(out[j],image_dir,j,name=config_name)
        if time_count==0:
            time_count=1
        logger.info('Average Run Time:\n' 
                + dash_line + 
                "{:.4f} s.".format(sum_time/time_count) + '\n' +
                dash_line)
        
        psnr_dict["psnr_mean"] = np.mean(psnr_list)
        ssim_dict["ssim_mean"] = np.mean(ssim_list)
        
    psnr_str = ", ".join([key+": "+"{:.4f}".format(psnr_dict[key]) for key in psnr_dict.keys()])
    ssim_str = ", ".join([key+": "+"{:.4f}".format(ssim_dict[key]) for key in ssim_dict.keys()])
    logger.info("Mean PSNR: \n"+
                dash_line + 
                "{}.\n".format(psnr_str)+
                dash_line)

    logger.info("Mean SSIM: \n"+
                dash_line + 
                "{}.\n".format(ssim_str)+
                dash_line) 

if __name__=="__main__":
    main()
