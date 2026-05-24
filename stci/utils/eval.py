import os
import os.path as osp
from torch.utils.data.dataloader import DataLoader 
import torch 
from stci.utils.utils import save_image, tansT_tensor
from stci.utils.metrics import compare_psnr,compare_ssim
import numpy as np 
import einops 
import lpips

# --------------------------------------------------------------------------------------
#                           整幅测试单时空域压缩图像/STFormer
# --------------------------------------------------------------------------------------
def eval_psnr_ssim_STCI_NonBlock(model, test_data, mask, mask_sum, args):
    s_cr=4
    psnr_dict,ssim_dict = {},{}
    psnr_list,ssim_list = [],[]
    out_list,gt_list = [],[]
    data_loader = DataLoader(test_data,1,shuffle=False,num_workers=4)
    cr, patch_H, patch_W = mask.shape
    for iter,data in enumerate(data_loader):
        psnr,ssim = 0,0
        batch_output = []
        gt, meas = data                         # mea[1(b),64,64]/gt[1(b),8,256,256]
        meas = meas.float()
        # meas_grad = meas_grad.float()
        H,W = meas.shape[-2]*s_cr, meas.shape[-1]*s_cr
        n_h = H // patch_H
        n_w = W // patch_W
        patch_h = patch_H // s_cr
        patch_w = patch_W // s_cr
        batch_size = meas.shape[0]
        
        Phi = einops.repeat(mask,'cr h w->b cr h w',b=1)
        Phi_s = einops.repeat(mask_sum,'h w->b 1 h w',b=1)            
        Phi = torch.from_numpy(Phi).to(args.device)                     # [b 8 256 256]
        Phi_s = torch.from_numpy(Phi_s).to(args.device)                 # [b 1 64 64]
                
        for ii in range(batch_size):                                    # 每个循环完成一帧压缩图像的测试，循环结束完成单个场景的测试
            single_meas = meas[ii].unsqueeze(0).unsqueeze(0)            # single_meas:[1,1,64,64]
            # single_meas_grad = meas_grad[ii].unsqueeze(0).unsqueeze(0)
            x_recon = torch.zeros(cr,H,W).to(args.device)
            with torch.no_grad():
                for h_id in range(n_h):
                    for w_id in range(n_w):
                        mea_block = single_meas[:1,:1,h_id*patch_h:(h_id+1)*patch_h, w_id*patch_w:(w_id+1)*patch_w].to(args.device)
                        # mea_grad_block = single_meas_grad[:1,:1,h_id*patch_h:(h_id+1)*patch_h, w_id*patch_w:(w_id+1)*patch_w].to(args.device)
                        input_x = (mea_block, Phi, Phi_s)
                        outputs = model(input_x)
                        if not isinstance(outputs,list):
                            outputs = [outputs]
                        # output = outputs[-1][0].cpu().numpy()     # 先取list后取[cr H W]
                        output = outputs[-1][0]
                        x_recon[:,h_id*patch_H:(h_id+1)*patch_H, w_id*patch_W:(w_id+1)*patch_W] = output
            x_recon = x_recon.cpu().numpy()                         # 单帧重建图片的完整值 [cr H W]
            batch_output.append(x_recon)                            # n(batch_size)-[cr,H,W]-可以按帧索引
            for jj in range(cr):
                if x_recon.shape[0]==3:
                    per_frame_out = x_recon[:,jj]
                    per_frame_out = np.sum(per_frame_out*test_data.rgb2raw,axis=0)
                else:
                    per_frame_out = x_recon[jj]
                per_frame_gt = gt[ii,jj, :, :].numpy()
                psnr += compare_psnr(per_frame_gt*255,per_frame_out*255)
                ssim += compare_ssim(per_frame_gt*255,per_frame_out*255)
        psnr = psnr / (batch_size * cr)                 # batch_size实际为单个场景的测量帧数//至此完成一个场景的测试
        ssim = ssim / (batch_size * cr)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        out_list.append(np.array(batch_output))
        gt_list.append(gt)

    test_dir = osp.join(args.work_dir,"test_images")
    if not osp.exists(test_dir):
        os.makedirs(test_dir)
    for i in range(len(out_list)):
        _name = f"scene_{i}"
        psnr_dict[_name] = psnr_list[i]
        ssim_dict[_name] = ssim_list[i]
        # ---------八帧图片同时保存---------
        # out = out_list[i]               # out_list[n(场景的个数)[1(batch_size:此处为单个场景的测量帧数)[8 1024 1920]]]
        # gt = gt_list[i]                 # [n[1[8 1024 1920]]]
        # for j in range(out.shape[0]):
        #     image_name = osp.join(test_dir,_name+"_"+str(j)+".png")
        #     save_image(out[j],gt[j],image_name)
        # ---------依次保存单帧图片---------
        for k in range(batch_size):
            out = out_list[i][k]      # out_list[n_scene, b, cr, H,W]
            gt = gt_list[i][k]
            for j in range(out.shape[0]):
                image_name = osp.join(test_dir,_name+"_"+str(j)+".png")
                save_image(out[j],gt[j],image_name)
    psnr_dict["psnr_mean"] = np.mean(psnr_list)
    ssim_dict["ssim_mean"] = np.mean(ssim_list)
    return psnr_dict,ssim_dict

# --------------------------------------------------------------------------------------
#                     整幅测试单时空域压缩图像/STFormer/增加lpips指标测试
# --------------------------------------------------------------------------------------
def eval_psnr_ssim_STCI_NonBlock2(model, test_data, mask, mask_sum, args):
    s_cr=4
    psnr_dict,ssim_dict,lpips_val_dict = {},{},{}
    psnr_list,ssim_list,lpips_val_list = [],[],[]
    out_list,gt_list = [],[]
    data_loader = DataLoader(test_data,1,shuffle=False,num_workers=4)
    cr, patch_H, patch_W = mask.shape

    lpips_fn = lpips.LPIPS(net='vgg').to(args.device)
    lpips_fn.eval()

    for iter,data in enumerate(data_loader):
        psnr, ssim, lpips_val= 0,0,0
        batch_output = []
        gt, meas = data                         # mea[1(b),64,64]/gt[1(b),8,256,256]
        meas = meas.float()
        # meas_grad = meas_grad.float()
        H,W = meas.shape[-2]*s_cr, meas.shape[-1]*s_cr
        n_h = H // patch_H
        n_w = W // patch_W
        patch_h = patch_H // s_cr
        patch_w = patch_W // s_cr
        batch_size = meas.shape[0]
        
        Phi = einops.repeat(mask,'cr h w->b cr h w',b=1)
        Phi_s = einops.repeat(mask_sum,'h w->b 1 h w',b=1)            
        Phi = torch.from_numpy(Phi).to(args.device)                     # [b 8 256 256]
        Phi_s = torch.from_numpy(Phi_s).to(args.device)                 # [b 1 64 64]
                
        for ii in range(batch_size):                                    # 每个循环完成一帧压缩图像的测试，循环结束完成单个场景的测试
            single_meas = meas[ii].unsqueeze(0).unsqueeze(0)            # single_meas:[1,1,64,64]
            x_recon = torch.zeros(cr,H,W).to(args.device)
            with torch.no_grad():
                for h_id in range(n_h):
                    for w_id in range(n_w):
                        mea_block = single_meas[:1,:1,h_id*patch_h:(h_id+1)*patch_h, w_id*patch_w:(w_id+1)*patch_w].to(args.device)
                        input_x = (mea_block, Phi, Phi_s)
                        outputs = model(input_x)
                        if not isinstance(outputs,list):
                            outputs = [outputs]
                        output = outputs[-1][0]
                        x_recon[:,h_id*patch_H:(h_id+1)*patch_H, w_id*patch_W:(w_id+1)*patch_W] = output
            x_recon = x_recon.cpu().numpy()            # 单帧重建图片的完整值 [cr H W]
            batch_output.append(x_recon)               # n(batch_size)-[cr,H,W]-可以按帧索引
            for jj in range(cr):
                if x_recon.shape[0]==3:
                    per_frame_out = x_recon[:,jj]
                    per_frame_out = np.sum(per_frame_out*test_data.rgb2raw,axis=0)
                else:
                    per_frame_out = x_recon[jj]
                per_frame_gt = gt[ii,jj, :, :].numpy()
                psnr += compare_psnr(per_frame_gt*255,per_frame_out*255)
                ssim += compare_ssim(per_frame_gt*255,per_frame_out*255)
                # 计算LPIPS (需要将灰度图扩展为3通道, 归一化到[-1,1])
                frame_out_t = torch.from_numpy(per_frame_out).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                frame_gt_t = torch.from_numpy(per_frame_gt).float().unsqueeze(0).unsqueeze(0)    # [1,1,H,W]
                frame_out_t = frame_out_t.expand(-1, 3, -1, -1) * 2.0 - 1.0  # 归一化到[-1,1]
                frame_gt_t = frame_gt_t.expand(-1, 3, -1, -1) * 2.0 - 1.0
                with torch.no_grad():
                    lpips_val += lpips_fn(frame_gt_t.to(args.device), frame_out_t.to(args.device)).item()
        
        psnr = psnr / (batch_size * cr)                 # batch_size实际为单个场景的测量帧数//至此完成一个场景的测试
        ssim = ssim / (batch_size * cr)
        lpips_val = lpips_val / (batch_size * cr)

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_val_list.append(lpips_val)
        out_list.append(np.array(batch_output))
        gt_list.append(gt)

    test_dir = osp.join(args.work_dir,"test_images")
    if not osp.exists(test_dir):
        os.makedirs(test_dir)
    for i in range(len(out_list)):
        _name = f"scene_{i}"
        psnr_dict[_name] = psnr_list[i]
        ssim_dict[_name] = ssim_list[i]
        # ---------save image---------
        for k in range(batch_size):
            out = out_list[i][k]      # out_list[n_scene, b, cr, H,W]
            gt = gt_list[i][k]
            for j in range(out.shape[0]):
                image_name = osp.join(test_dir,_name+"_"+str(j)+".png")
                save_image(out[j],gt[j],image_name)
    psnr_dict["psnr_mean"] = np.mean(psnr_list)
    ssim_dict["ssim_mean"] = np.mean(ssim_list)
    lpips_val_dict["lpips_mean"] = np.mean(lpips_val_list)
    return psnr_dict,ssim_dict,lpips_val_dict


# --------------------------------------------------------------------------------------
#                     整幅测试单时空域压缩图像/STFormer/增加lpips指标测试（复用lpips实例）
# --------------------------------------------------------------------------------------
def eval_psnr_ssim_STCI_NonBlock3(model, test_data, mask, mask_sum, args, lpips_fn=None):
    """
    lpips_fn: 外部传入的已初始化 LPIPS 实例（复用，避免每次评估在 GPU 上重新加载 VGG）。
              若为 None 则跳过 LPIPS 计算。
    """
    s_cr = 4
    psnr_dict, ssim_dict, lpips_val_dict = {}, {}, {}
    psnr_list, ssim_list, lpips_val_list = [], [], []

    data_loader = DataLoader(test_data, 1, shuffle=False, num_workers=4)
    cr, patch_H, patch_W = mask.shape

    test_dir = osp.join(args.work_dir, "test_images")
    if not osp.exists(test_dir):
        os.makedirs(test_dir)

    # Mask 在循环外只构建一次，避免每个 scene 重复 to(device)
    Phi   = torch.from_numpy(einops.repeat(mask,     'cr h w->b cr h w', b=1)).to(args.device)
    Phi_s = torch.from_numpy(einops.repeat(mask_sum, 'h w->b 1 h w',     b=1)).to(args.device)

    for scene_idx, data in enumerate(data_loader):
        psnr, ssim, lpips_val = 0, 0, 0
        gt, meas = data
        meas = meas.float()
        H, W     = meas.shape[-2] * s_cr, meas.shape[-1] * s_cr
        n_h      = H // patch_H
        n_w      = W // patch_W
        patch_h  = patch_H // s_cr
        patch_w  = patch_W // s_cr
        batch_size = meas.shape[0]

        for ii in range(batch_size):
            single_meas = meas[ii].unsqueeze(0).unsqueeze(0)   # [1,1,h,w]
            x_recon = torch.zeros(cr, H, W).to(args.device)
            with torch.no_grad():
                for h_id in range(n_h):
                    for w_id in range(n_w):
                        mea_block = single_meas[
                            :1, :1,
                            h_id*patch_h:(h_id+1)*patch_h,
                            w_id*patch_w:(w_id+1)*patch_w
                        ].to(args.device)
                        outputs = model((mea_block, Phi, Phi_s))
                        if not isinstance(outputs, list):
                            outputs = [outputs]
                        x_recon[
                            :,
                            h_id*patch_H:(h_id+1)*patch_H,
                            w_id*patch_W:(w_id+1)*patch_W
                        ] = outputs[-1][0]

            x_recon_np = x_recon.cpu().numpy()   # [cr, H, W]

            for jj in range(cr):
                if x_recon_np.shape[0] == 3:
                    per_frame_out = np.sum(x_recon_np[:, jj] * test_data.rgb2raw, axis=0)
                else:
                    per_frame_out = x_recon_np[jj]
                per_frame_gt = gt[ii, jj, :, :].numpy()
                psnr += compare_psnr(per_frame_gt * 255, per_frame_out * 255)
                ssim += compare_ssim(per_frame_gt * 255, per_frame_out * 255)

                if lpips_fn is not None:
                    f_out = torch.from_numpy(per_frame_out).float().unsqueeze(0).unsqueeze(0)
                    f_gt  = torch.from_numpy(per_frame_gt).float().unsqueeze(0).unsqueeze(0)
                    f_out = f_out.expand(-1, 3, -1, -1) * 2.0 - 1.0
                    f_gt  = f_gt.expand(-1, 3, -1, -1) * 2.0 - 1.0
                    with torch.no_grad():
                        lpips_val += lpips_fn(f_gt.to(args.device), f_out.to(args.device)).item()

            # 推理完即时保存，不再积压到 out_list / gt_list
            for j in range(cr):
                image_name = osp.join(test_dir, f"scene_{scene_idx}_{j}.png")
                save_image(x_recon_np[j], gt[ii, j].numpy(), image_name)

        psnr      = psnr      / (batch_size * cr)
        ssim      = ssim      / (batch_size * cr)
        lpips_val = lpips_val / (batch_size * cr)

        _name = f"scene_{scene_idx}"
        psnr_dict[_name]      = psnr
        ssim_dict[_name]      = ssim
        lpips_val_dict[_name] = lpips_val
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_val_list.append(lpips_val)

    psnr_dict["psnr_mean"]      = np.mean(psnr_list)
    ssim_dict["ssim_mean"]      = np.mean(ssim_list)
    lpips_val_dict["lpips_mean"] = np.mean(lpips_val_list)
    return psnr_dict, ssim_dict, lpips_val_dict


# --------------------------------------------------------------------------------------
#                      整幅测试单时空域压缩图像/STST_OptimizeSTDMask
# 注意：不再接收外部 mask 参数。
# STST_OptimizeSTDMask 模型内部包含可学习的时空掩膜参数（PHI_s / PHI_t），
# 测试时掩膜由模型 forward 自行生成，是联合优化结果的一部分。
# --------------------------------------------------------------------------------------
def eval_STST_OptimizeMask(model, test_data, args):
    psnr_dict, ssim_dict = {}, {}
    psnr_list, ssim_list = [], []
    out_list,  gt_list   = [], []
    data_loader = DataLoader(test_data, 1, shuffle=False, num_workers=4)

    # 从模型内部属性获取分块尺寸，保持与 forward 物理含义一致
    # 不依赖外部 mask 的 shape，避免接口歧义
    cr      = model.temCR       # 时域压缩比（帧数）
    patch_H = model.BlockSize_h # 空间块高度
    patch_W = model.BlockSize_w # 空间块宽度

    for scene_idx, gt in enumerate(data_loader):
        psnr, ssim = 0, 0
        batch_output = []

        H, W       = gt.shape[-2], gt.shape[-1]   # gt: [1, cr, H, W]
        n_h        = H // patch_H
        n_w        = W // patch_W
        batch_size = gt.shape[0]

        for ii in range(batch_size):
            single_gt = gt[ii].unsqueeze(0)             # [1, cr, H, W]
            x_recon   = torch.zeros(cr, H, W).to(args.device)

            with torch.no_grad():
                for h_id in range(n_h):
                    for w_id in range(n_w):
                        gt_block = single_gt[
                            :1, :,
                            h_id*patch_H:(h_id+1)*patch_H,
                            w_id*patch_W:(w_id+1)*patch_W
                        ].to(args.device)

                        # 模型用自身学到的掩膜执行压缩与重建，无需外部 mask
                        outputs = model(gt_block)
                        if isinstance(outputs, tuple):
                            x_out = outputs[0]          # [B, 1, cr, patch_H, patch_W]
                        elif isinstance(outputs, list):
                            x_out = outputs[-1]
                        else:
                            x_out = outputs
                        output = x_out.squeeze(1)[0]    # [cr, patch_H, patch_W]
                        x_recon[
                            :,
                            h_id*patch_H:(h_id+1)*patch_H,
                            w_id*patch_W:(w_id+1)*patch_W
                        ] = output

            x_recon = x_recon.cpu().numpy()             # [cr, H, W]
            batch_output.append(x_recon)

            for jj in range(cr):
                if x_recon.shape[0] == 3:
                    per_frame_out = np.sum(x_recon[:, jj] * test_data.rgb2raw, axis=0)
                else:
                    per_frame_out = x_recon[jj]
                per_frame_gt = gt[ii, jj, :, :].numpy()
                psnr += compare_psnr(per_frame_gt * 255, per_frame_out * 255)
                ssim += compare_ssim(per_frame_gt * 255, per_frame_out * 255)

        psnr = psnr / (batch_size * cr)
        ssim = ssim / (batch_size * cr)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        out_list.append(np.array(batch_output))
        gt_list.append(gt)

    test_dir = osp.join(args.work_dir, "test_images")
    if not osp.exists(test_dir):
        os.makedirs(test_dir)
    for i in range(len(out_list)):
        _name = f"scene_{i}"
        psnr_dict[_name] = psnr_list[i]
        ssim_dict[_name] = ssim_list[i]
        for k in range(batch_size):
            out = out_list[i][k]
            gt  = gt_list[i][k]
            for j in range(out.shape[0]):
                image_name = osp.join(test_dir, _name + "_" + str(j) + ".png")
                save_image(out[j], gt[j], image_name)

    psnr_dict["psnr_mean"] = np.mean(psnr_list)
    ssim_dict["ssim_mean"] = np.mean(ssim_list)
    return psnr_dict, ssim_dict
