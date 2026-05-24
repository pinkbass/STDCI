import numpy as np
import cv2
import os

def read_yuv420_video(filename, width, height, num_frames):
    """
        读取YUV420格式的视频文件并转换为RGB图像列表。
        
        参数：
        - filename: YUV文件路径
        - width: 图像宽度
        - height: 图像高度
        - num_frames: 要读取的帧数
        
        返回：
        - RGB图像列表（每个元素为NumPy数组）
    """
    frames = []
    frame_size = width * height * 3 // 2                  # YUV420
    with open(filename, 'rb') as f:
        for _ in range(num_frames):
            yuv = f.read(frame_size)
            if not yuv:
                break
            yuv = np.frombuffer(yuv, dtype=np.uint8)      # 一维数据/前2/3为Y分量/接续的1/6为U分量/最后剩余的1/6为V分量
            y = yuv[:width * height].reshape((height, width))
            u = yuv[width * height:width * height + (width // 2) * (height // 2)].reshape((height // 2, width // 2))
            v = yuv[width * height + (width // 2) * (height // 2):].reshape((height // 2, width // 2))
            
            # 上采样U和V分量
            u_up = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
            v_up = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)
            
            # 合并YUV分量
            yuv_full = cv2.merge((y, u_up, v_up))
            
            # 转换为RGB
            rgb = cv2.cvtColor(yuv_full, cv2.COLOR_YUV2RGB)
            frames.append(rgb)
    return frames


if __name__ == "__main__":
    width, height = 1920, 1080                             # 根据实际分辨率调整
    num_frames = 80                                         # 需要读取的帧数
    yuv_filename = "/media/gpu2/dis8t2/hxw/davis_image/UVG_dataset/YachtRide_1920x1080_120fps_420_8bit_YUV.yuv"  # 替换为实际文件路径
    
    frames = read_yuv420_video(yuv_filename, width, height, num_frames)
    
    # 保存为PNG图像
    output_dir = '/media/gpu2/dis8t2/hxw/davis_image/UVG_dataset/Image_UVG/YachtRide/'
    os.makedirs(output_dir, exist_ok=True)
    for idx, frame in enumerate(frames):
        cv2.imwrite(os.path.join(output_dir, f'frame_{idx:04d}.png'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# from PIL import Image
# import numpy as np

# def yuv420_to_rgb(y, u, v, width, height):
#     """
#     将YUV420分量转换为RGB图像。
    
#     参数：
#     - y: Y分量（NumPy数组）
#     - u: U分量（NumPy数组）
#     - v: V分量（NumPy数组）
#     - width: 图像宽度
#     - height: 图像高度
    
#     返回：
#     - RGB图像（NumPy数组）
#     """
#     # 上采样U和V分量
#     u_up = u.repeat(2, axis=0).repeat(2, axis=1)
#     v_up = v.repeat(2, axis=0).repeat(2, axis=1)
    
#     # 合并YUV分量
#     yuv = np.stack((y, u_up, v_up), axis=-1).astype(np.float32)
    
#     # YUV到RGB的转换公式（BT.601标准）
#     yuv[:, :, 0] = yuv[:, :, 0] - 16
#     yuv[:, :, 1] = yuv[:, :, 1] - 128
#     yuv[:, :, 2] = yuv[:, :, 2] - 128
    
#     r = 1.164 * yuv[:, :, 0] + 1.596 * yuv[:, :, 2]
#     g = 1.164 * yuv[:, :, 0] - 0.392 * yuv[:, :, 1] - 0.813 * yuv[:, :, 2]
#     b = 1.164 * yuv[:, :, 0] + 2.017 * yuv[:, :, 1]
    
#     rgb = np.stack((r, g, b), axis=-1)
#     rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    
#     return rgb

# def read_yuv420_frame_pillow(filename, width, height):
#     """
#     读取YUV420格式的单帧数据并转换为RGB图像（使用Pillow）。
    
#     参数：
#     - filename: YUV文件路径
#     - width: 图像宽度
#     - height: 图像高度
    
#     返回：
#     - RGB图像（NumPy数组）
#     """
#     with open(filename, 'rb') as f:
#         # 读取Y分量
#         y_size = width * height
#         y = np.frombuffer(f.read(y_size), dtype=np.uint8).reshape((height, width))
        
#         # 读取U分量
#         uv_size = (width // 2) * (height // 2)
#         u = np.frombuffer(f.read(uv_size), dtype=np.uint8).reshape((height // 2, width // 2))
        
#         # 读取V分量
#         v = np.frombuffer(f.read(uv_size), dtype=np.uint8).reshape((height // 2, width // 2))
        
#         # 转换为RGB
#         rgb = yuv420_to_rgb(y, u, v, width, height)
        
#         return rgb

# # 示例使用
# if __name__ == "__main__":
#     width, height = 1920, 1080                                                                                  # 根据实际分辨率调整
#     yuv_filename = "/media/gpu2/dis8t1/hxw/davis_image/UVG_dataset/Beauty_1920x1080_120fps_420_8bit_YUV.yuv"    # 替换为实际文件路径
#     rgb_image = read_yuv420_frame_pillow(yuv_filename, width, height)
    
#     # 创建Pillow图像
#     img = Image.fromarray(rgb_image, 'RGB')
    
#     # 显示图像
#     img.show()
    
#     # 保存为PNG
#     img.save('output_frame_pillow.png')

