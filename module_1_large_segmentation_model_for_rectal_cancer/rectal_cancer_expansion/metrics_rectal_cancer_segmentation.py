# -*- coding: utf-8 -*-
# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
import torch
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse


# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

def get_bbox(mask, bbox_shift=5):
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)
    bboxes = np.array([x_min, y_min, x_max, y_max])
    return bboxes

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

def dice_coefficient(prediction, target):
    """
    计算 Dice 系数
    Args:
        prediction: 预测的二进制数组
        target: 真实的二进制数组
    Returns:
        Dice 系数
    """
    intersection = np.sum(prediction * target)
    dice = (2. * intersection) / (np.sum(prediction) + np.sum(target))
    return dice

def hd95(image1, image2, voxel_spacing=None):
    """
    计算两个二值化图像之间的 HD95（Hausdorff 距离的 95% 分位数）。

    参数：
    image1: 第一个二值化图像的 NumPy 数组。
    image2: 第二个二值化图像的 NumPy 数组，与 image1 具有相同的形状。
    voxel_spacing: 一个元组，包含每个轴上的像素间距。如果为 None，则假定各轴上的间距均为 1。

    返回值：
    HD95 距离值。
    """
    if voxel_spacing is None:
        voxel_spacing = (1, 1, 1)  # 假设各轴上的间距均为 1

    # 计算 image1 中每个点到 image2 中最近点的距离
    dist1 = np.linalg.norm(np.array(np.where(image1)).T[:, None, :] - np.array(np.where(image2)).T[None, :, :], axis=-1)

    # 计算 image2 中每个点到 image1 中最近点的距离
    dist2 = np.linalg.norm(np.array(np.where(image2)).T[:, None, :] - np.array(np.where(image1)).T[None, :, :], axis=-1)

    # 计算每个点到对应图像的最近距离
    min_dist1 = np.min(dist1, axis=1)
    min_dist2 = np.min(dist2, axis=1)

    # 计算 HD95（Hausdorff 距离的 95% 分位数）
    hd95_1 = np.percentile(min_dist1, 95)
    hd95_2 = np.percentile(min_dist2, 95)

    return max(hd95_1, hd95_2)

def asd(image1, image2, voxel_spacing=None):
    """
    计算两个二值化图像之间的 ASD（Average Surface Distance 平均表面距离）。

    参数：
    image1: 第一个二值化图像的 NumPy 数组。
    image2: 第二个二值化图像的 NumPy 数组，与 image1 具有相同的形状。
    voxel_spacing: 一个元组，包含每个轴上的像素间距。如果为 None，则假定各轴上的间距均为 1。

    返回值：
    ASD（Average Surface Distance）值。
    """
    if voxel_spacing is None:
        voxel_spacing = (1, 1, 1)  # 假设各轴上的间距均为 1

    # 计算 image1 中每个点到 image2 中最近点的距离
    dist1 = np.linalg.norm(np.array(np.where(image1)).T[:, None, :] - np.array(np.where(image2)).T[None, :, :], axis=-1)

    # 计算 image2 中每个点到 image1 中最近点的距离
    dist2 = np.linalg.norm(np.array(np.where(image2)).T[:, None, :] - np.array(np.where(image1)).T[None, :, :], axis=-1)

    # 计算平均表面距离
    asd_1 = np.mean(np.min(dist1, axis=1))
    asd_2 = np.mean(np.min(dist2, axis=1))

    return (asd_1 + asd_2) / 2

images_path = '/home/sx91/MedSAM-main/data/test/npy1/MRI_Abd/imgs'
gts_path = '/home/sx91/MedSAM-main/data/test/npy1/MRI_Abd/gts'
segs_path = '/home/sx91/MedSAM-main/data/test/npy1/MRI_Abd/segs_lr'

dice_score_list = []
hd95_list = []
asd_list = []

for filepath in sorted(os.listdir(images_path)):
  
    image_path = os.path.join(images_path, filepath)
    gt_path = os.path.join(gts_path, filepath)

    mask = np.load(gt_path)
    # args.data_path = image_path

    # %% load model and image
    parser = argparse.ArgumentParser(
        description="run inference on testing set based on MedSAM"
    )
    parser.add_argument(
        "-i",
        "--data_path",
        type=str,
        default=image_path,
        help="path to the data folder",
    )
    parser.add_argument(
        "-o",
        "--seg_path",
        type=str,
        default=segs_path,
        help="path to the segmentation folder",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument(
        "-chk",
        "--checkpoint",
        type=str,
        # default="/hy-tmp/MedSAM-main/work_dir/SAM/sam_vit_b_01ec64.pth",
        # default="/hy-tmp/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth",
        # default="/home/sx91/Transformation/MedSAM-main/work_dir/medsam_vit_b.pth",
        default="/home/sx91/MedSAM-main/work_dir/exp6/medsam_model_best_200.pth",
        # default="/home/sx91/Transformation/MedSAM-main/work_dir/medsam_model_best.pth",
        # default="/home/sx91/Transformation/MedSAM-main/work_dir/SAM/sam_vit_b_01ec64.pth",
        # default="/home/sx91/Transformation/MedSAM-main/work_dir/medsam_vit_b.pth",
        help="path to the trained model",
    )
    parser.add_argument(
        "--box",
        type=list,
        default=get_bbox(mask),
        help="bounding box of the segmentation target",
    )
    args = parser.parse_args()
    device = args.device
    print("box:", args.box)
    medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    # medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    
    # img_np = io.imread(args.data_path)
    
    # test_npzs = sorted(os.listdir(args.data_path))
    #
    # npz_idx =np.random.randint(0, len(test_npzs))
    # npz = np.load(join(args.data_path, test_npzs[npz_idx]), allow_pickle=True)
    # array_names = npz.files
    # print("Arrays in the .npz file:", array_names)
    # for array_name in array_names:
    #     array = npz[array_name]
    #     print(f"{array_name}:")
    #     print(array)
    #
    # print(npz['imgs'].shape)
    # print(npz['imgs'][0].shape)
    
    # imgs = npz['imgs']
    # gts = npz['gts']
    # img_np = npz['imgs'][0]
    # print(img_np.shape)
    
    img_np = np.load(args.data_path)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    H, W, _ = img_3c.shape
    # %% image preprocessing
    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W, 3)
    # convert the shape to (3, H, W)
    img_1024_tensor = (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )
    
    box_np = np.array([args.box])
    # transfer box_np t0 1024x1024 scale
    box_1024 = box_np / np.array([W, H, W, H]) * 1024
    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)
    
    medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
    dice_score = dice_coefficient(medsam_seg, mask)
    hd95_score = hd95(medsam_seg, mask)
    asd_score = asd(medsam_seg, mask)
    print('Path:', os.path.basename(args.data_path))
    print('Dice:', dice_score, 'HD95:', hd95_score, 'ASD:', asd_score)
    
    dice_score_list.append(dice_score)
    hd95_list.append(hd95_score)
    asd_list.append(asd_score)
    #
    # np.save(join(args.seg_path, "seg_" + os.path.basename(args.data_path)),
    #         medsam_seg)
    #
    # io.imsave(
    #     join(args.seg_path, "seg_" + os.path.basename(args.data_path)),
    #     medsam_seg,
    #     check_contrast=False,
    # )

# print(dice_score_list)
print(np.mean(np.array(dice_score_list)))
print(np.std(np.array(dice_score_list)))
print(np.median(np.array(dice_score_list)))
print(np.max(np.array(dice_score_list)))
print(np.min(np.array(dice_score_list)))
print(np.mean(np.array(hd95_list)))
print(np.std(np.array(hd95_list)))
print(np.median(np.array(hd95_list)))
print(np.max(np.array(hd95_list)))
print(np.min(np.array(hd95_list)))
print(np.mean(np.array(asd_list)))
print(np.std(np.array(asd_list)))
print(np.median(np.array(asd_list)))
print(np.max(np.array(asd_list)))
print(np.min(np.array(asd_list)))






# %% visualize results
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(img_3c)
# show_box(box_np[0], ax[0])
# ax[0].set_title("Input Image and Bounding Box")
# ax[1].imshow(img_3c)
# show_mask(medsam_seg, ax[1])
# show_box(box_np[0], ax[1])
# ax[1].set_title("MedSAM Segmentation")
# plt.show()




# 读取两个 .npy 文件
# prediction = np.load('prediction.npy')
# target = np.load('target.npy')
#
# # 计算 Dice 系数
# dice_score = dice_coefficient(prediction, target)
#
# print("Dice 系数:", dice_score)