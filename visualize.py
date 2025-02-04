import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

def reshape_normalize(x):
    '''
    Args:
        x: [B, C, H, W]

    Returns:

    '''
    B, C, H, W = x.shape
    x = x.transpose([0, 2, 3, 1]).reshape([-1, C])

    denominator = np.linalg.norm(x, axis=-1, keepdims=True)
    denominator = np.where(denominator==0, 1, denominator)
    return x / denominator

def normalize(x):
    denominator = np.linalg.norm(x, axis=-1, keepdims=True)
    denominator = np.where(denominator == 0, 1, denominator)
    return x / denominator

def single_features_to_RGB(sat_features, idx=0, img_name='test_img.png'):
    sat_feat = sat_features[idx:idx+1,:,:,:].data.cpu().numpy()
    # 1. 重塑特征图形状为 [256, 64*64]
    B, C, H, W = sat_feat.shape
    flatten = np.concatenate([sat_feat], axis=0)
    # 2. 进行 PCA 降维到 3 维
    pca = PCA(n_components=3)
    pca.fit(reshape_normalize(flatten))
    
    # 3. 归一化到 [0, 1] 范围
    sat_feat_new = ((normalize(pca.transform(reshape_normalize(sat_feat))) + 1 )/ 2).reshape(B, H, W, 3)

    sat = Image.fromarray((sat_feat_new[0] * 255).astype(np.uint8))
    # sat = sat.resize((512, 512))
    sat.save(img_name)

def sat_features_to_RGB(sat_features, grd_features, idx=0):
    sat_feat = sat_features[idx:idx+1,:,:,:].data.cpu().numpy()
    grd_feat = grd_features[idx:idx+1,:,:,:].data.cpu().numpy()
    # 1. 重塑特征图形状为 [256, 64*64]
    B, C, A, A = sat_feat.shape
    _, _, H, W = grd_feat.shape
    flatten = np.concatenate([sat_feat.reshape(B, C, -1), grd_feat.reshape(B, C, -1)], axis=0)
    # 2. 进行 PCA 降维到 3 维
    pca = PCA(n_components=3)
    pca.fit(normalize(flatten.transpose([0,2,1]).reshape(-1, C)))
    
    # 3. 归一化到 [0, 1] 范围
    sat_feat_new = ((normalize(pca.transform(reshape_normalize(sat_feat))) + 1 )/ 2).reshape(B, A, A, 3)
    fuse_feat_new = ((normalize(pca.transform(reshape_normalize(grd_feat))) + 1 )/ 2).reshape(B, H, W, 3)

    sat = Image.fromarray((sat_feat_new[0] * 255).astype(np.uint8))
    sat = sat.resize((A, A))
    sat.save('sat_feat.png')
    
    grd = Image.fromarray((fuse_feat_new[0] * 255).astype(np.uint8))
    grd = grd.resize((W, H))
    grd.save('grd_feat.png')

def features_to_RGB(sat_feat, g2s_feat_center, g2s_conf_center, g2s_feat_gt, g2s_conf_gt, loop, level, save_dir):
    """Project a list of d-dimensional feature maps to RGB colors using PCA."""
    from sklearn.decomposition import PCA

    def reshape_normalize(x):
        '''
        Args:
            x: [B, C, H, W]

        Returns:

        '''
        B, C, H, W = x.shape
        x = x.transpose([0, 2, 3, 1]).reshape([-1, C])

        denominator = np.linalg.norm(x, axis=-1, keepdims=True)
        denominator = np.where(denominator==0, 1, denominator)
        return x / denominator

    def normalize(x):
        denominator = np.linalg.norm(x, axis=-1, keepdims=True)
        denominator = np.where(denominator == 0, 1, denominator)
        return x / denominator

    sat_feat = sat_feat.data.cpu().numpy()  # [B, C, H, W]
    g2s_feat_center = g2s_feat_center.data.cpu().numpy()  # [B, C, H, W]
    g2s_feat_gt = g2s_feat_gt.data.cpu().numpy()

    B, C, A, _ = sat_feat.shape

    flatten = np.concatenate([sat_feat, g2s_feat_center, g2s_feat_gt], axis=0)

    # if level == 0:
    pca = PCA(n_components=3)
    pca.fit(reshape_normalize(flatten))

    sat_feat_new = ((normalize(pca.transform(reshape_normalize(sat_feat))) + 1 )/ 2).reshape(B, A, A, 3)

    mask_center = g2s_conf_center[:, 0, :, :, None].data.cpu().numpy()
    mask_center = mask_center / mask_center.max()
    mask = np.linalg.norm(g2s_feat_center, axis=1)[:, :, :, None] > 0
    g2s_feat_new_center = ((normalize(pca.transform(reshape_normalize(g2s_feat_center))) + 1) / 2).reshape(B, A, A, 3) * mask

    mask_gt = g2s_conf_gt[:, 0, :, :, None].data.cpu().numpy()
    mask_gt = mask_gt / mask_gt.max()
    mask = np.linalg.norm(g2s_feat_gt, axis=1)[:, :, :, None] > 0
    g2s_feat_new_gt = ((normalize(pca.transform(reshape_normalize(g2s_feat_gt))) + 1) / 2).reshape(B, A, A, 3) * mask

    for idx in range(B):
        sat = Image.fromarray((sat_feat_new[idx] * 255).astype(np.uint8))
        sat = sat.resize((512, 512))
        sat.save(os.path.join(save_dir, 'level_' + str(level) + '_sat_feat_' + str(loop * B + idx) + '.png'))

        g2s_center = Image.fromarray((g2s_feat_new_center[idx] * 255).astype(np.uint8))
        g2s_center = g2s_center.resize((512, 512))
        g2s_center.save(os.path.join(save_dir, 'level_' + str(level) + '_g2s_feat_center' + str(loop * B + idx) + '.png'))

        g2s_center = Image.fromarray((g2s_feat_new_center[idx] * mask_center[idx] * 255).astype(np.uint8))
        g2s_center = g2s_center.resize((512, 512))
        g2s_center.save(
            os.path.join(save_dir, 'level_' + str(level) + '_g2s_feat_center_conf' + str(loop * B + idx) + '.png'))

        g2s_gt = Image.fromarray((g2s_feat_new_gt[idx] * 255).astype(np.uint8))
        g2s_gt = g2s_gt.resize((512, 512))
        g2s_gt.save(os.path.join(save_dir, 'level_' + str(level) + '_g2s_feat_gt' + str(loop * B + idx) + '.png'))

        g2s_gt = Image.fromarray((g2s_feat_new_gt[idx] * mask_gt[idx] * 255).astype(np.uint8))
        g2s_gt = g2s_gt.resize((512, 512))
        g2s_gt.save(os.path.join(save_dir, 'level_' + str(level) + '_g2s_feat_gt_conf' + str(loop * B + idx) + '.png'))

    return


def pca_2d_hsv_color(pca_2d, H, W):
    """
    将 2D PCA 的结果 (H*W, 2)：
      1. 对 x, y 各自做 min-max 归一化 -> [0,1]
      2. 将 (x, y) 映射到 HSV: H=x, S=y, V=1.0
      3. 转成 RGB，最后 reshape 到 (H, W, 3)
    """
    # pca_2d: shape (H*W, 2)
    pca_2d_norm = pca_2d.copy()

    # 分别对 x, y 做 min-max
    x_min, x_max = pca_2d_norm[:,0].min(), pca_2d_norm[:,0].max()
    y_min, y_max = pca_2d_norm[:,1].min(), pca_2d_norm[:,1].max()
    pca_2d_norm[:,0] = (pca_2d_norm[:,0] - x_min) / (x_max - x_min + 1e-8)
    pca_2d_norm[:,1] = (pca_2d_norm[:,1] - y_min) / (y_max - y_min + 1e-8)

    # HSV: hue = x, saturation = y, value = 1.0
    hsv = np.zeros((pca_2d_norm.shape[0], 3))
    hsv[:, 0] = pca_2d_norm[:,0]       # Hue
    hsv[:, 1] = pca_2d_norm[:,1]       # Saturation
    hsv[:, 2] = 1.0                    # Value=1
    # hsv[:, 2] = 0.7                    # Value=1
    # 转成 RGB
    rgb = mcolors.hsv_to_rgb(hsv)  # (H*W, 3)
    rgb = rgb.reshape(H, W, 3)     # (H, W, 3)
    gamma = 1.2  # >1会让图整体变暗
    rgb = rgb ** (1 / gamma)
    return rgb

def sat_features_to_RGB_2D_PCA(sat_features, grd_features, idx=0):
    """
    1) 取第 idx 个 batch 的 sat_feat, grd_feat
    2) 用 2D PCA 降维 -> (x, y)
    3) 每张图各自 reshape 回原尺寸后，映射到 HSV->RGB
    4) 保存可视化结果
    """
    def reshape_normalize(feat):
        """
        feat: shape (B, C, H, W)
        先把它 reshape 成 (B*H*W, C) 方便 PCA 的 transform。
        """
        B, C, H, W = feat.shape
        # (B, C, H, W) -> (B, C, H*W)
        feat = feat.reshape(B, C, -1)  
        # (B, C, H*W) -> (B, H*W, C)
        feat = feat.transpose(0,2,1)
        # (B, H*W, C) -> (B*H*W, C)
        feat = feat.reshape(-1, C)
        return feat
    # 取第 idx 个的特征
    sat_feat = sat_features[idx:idx+1,:,:,:].data.cpu().numpy()
    grd_feat = grd_features[idx:idx+1,:,:,:].data.cpu().numpy()

    B, C, A, A_ = sat_feat.shape  # A == A_
    _, _, H, W = grd_feat.shape

    # (1) reshape + 合并做 PCA 拟合（确保同一映射）
    sat_flat = reshape_normalize(sat_feat)  # shape: (A*A, C)
    grd_flat = reshape_normalize(grd_feat)  # shape: (H*W, C)
    combined = np.concatenate([sat_flat, grd_flat], axis=0)  # (A*A + H*W, C)

    # (2) 先整体 normalize，再 2D PCA
    combined_norm = normalize(combined)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(combined_norm)

    # 分别 transform
    sat_2d = pca.transform(normalize(sat_flat))  # shape: (A*A, 2)
    grd_2d = pca.transform(normalize(grd_flat))  # shape: (H*W, 2)

    # (3) 映射到 HSV->RGB
    sat_rgb = pca_2d_hsv_color(sat_2d, A, A)
    grd_rgb = pca_2d_hsv_color(grd_2d, H, W)

    # (4) 转成 [0,255] 并保存图像
    sat_img = Image.fromarray((sat_rgb * 255).astype(np.uint8))
    sat_img.save('sat_feat_2dpca.png')

    grd_img = Image.fromarray((grd_rgb * 255).astype(np.uint8))
    grd_img.save('grd_feat_2dpca.png')

    print("Saved sat_feat_2dpca.png and grd_feat_2dpca.png.")


def grd_features_to_RGB_2D_PCA_concat(grd_features, b_idx=0):
    """
    仅针对 grd_features, 形状: (B, V, C, H, W).
    1. 遍历同一个 batch b_idx 下的所有 v_idx -> 得到多张单图
    2. 将它们从上到下拼接成一张图
    3. 保存最终的大图
    """
    B, V, C, H, W = grd_features.shape

    # 判断 b_idx 合法
    assert 0 <= b_idx < B, f"b_idx={b_idx} 超出范围 [0, {B-1}]"

    # 用于存储每个视角的单图 (PIL Image)
    image_list = []

    for v_idx in range(V):
        # 1. 取出第 b_idx 个 batch、第 v_idx 个视角特征
        #    如果 grd_features 在 GPU，需要先 .cpu().numpy()
        feat = grd_features[b_idx, v_idx].detach().cpu().numpy()  # shape (C, H, W)

        # 2. reshape 成 (H*W, C)，然后 normalize
        feat_reshaped = feat.reshape(C, -1).transpose(1, 0)  # (H*W, C)
        feat_norm = normalize(feat_reshaped)

        # 3. 2D PCA
        pca = PCA(n_components=2, random_state=42)
        pca_2d = pca.fit_transform(feat_norm)  # (H*W, 2)

        # 4. 映射到 HSV->RGB
        rgb = pca_2d_hsv_color(pca_2d, H, W)

        # 5. 转成 [0,255] 并生成 PIL Image
        img = Image.fromarray((rgb * 255).astype(np.uint8))
        image_list.append(img)

    # ---- 所有视角的图像都在 image_list 里了，现在拼接它们 ----

    # 确定拼接后图像的宽度为所有图的最大宽度(一般它们应该相同)
    total_width = max(im.width for im in image_list)
    # 从上到下拼接，高度相加
    total_height = sum(im.height for im in image_list)

    # 建立一个空白画布来放置它们
    concat_img = Image.new("RGB", (total_width, total_height))

    # 逐张贴上去
    y_offset = 0
    for im in image_list:
        concat_img.paste(im, (0, y_offset))
        y_offset += im.height

    # 最终保存
    out_filename = f'grd_feat_2dpca_b{b_idx}_concat.png'
    concat_img.save(out_filename)
    print(f"Saved concatenated image: {out_filename}")


def visualize_1d_pca(tensor1, tensor2, output_filename="pca_visualization.png"):
    """
    使用1D PCA降维并可视化两个[1, 32, 128, 128]的tensor的特征，并将结果保存为.png文件。
    为可视化结果添加颜色映射，并绘制在同一张图上。

    参数:
    tensor1 (torch.Tensor): 第一个输入的tensor，形状为 [1, 32, 128, 128]
    tensor2 (torch.Tensor): 第二个输入的tensor，形状为 [1, 32, 128, 128]
    output_filename (str): 输出图像的文件名，默认为 'pca_visualization.png'
    """
    # 确保输入是四维tensor
    assert tensor1.ndimension() == 4 and tensor1.shape[0] == 1, "输入tensor1必须是[1, 32, 128, 128]的四维tensor"
    assert tensor2.ndimension() == 4 and tensor2.shape[0] == 1, "输入tensor2必须是[1, 32, 128, 128]的四维tensor"

    # 将两个tensor展平为[32, 128*128]
    tensor1_flat = tensor1.view(32, -1)  # 32x(128*128)
    tensor2_flat = tensor2.view(32, -1)  # 32x(128*128)

    # 转换为numpy数组，便于PCA操作
    tensor1_flat_np = tensor1_flat.cpu().detach().numpy()
    tensor2_flat_np = tensor2_flat.cpu().detach().numpy()

    # 使用sklearn中的PCA进行1D降维
    pca = PCA(n_components=1)
    tensor1_pca = pca.fit_transform(tensor1_flat_np)
    tensor2_pca = pca.fit_transform(tensor2_flat_np)

    # 创建一个渐变色的颜色映射
    norm = plt.Normalize(vmin=min(np.min(tensor1_pca), np.min(tensor2_pca)),
                         vmax=max(np.max(tensor1_pca), np.max(tensor2_pca)))
    cmap = cm.viridis  # 你可以选择不同的colormap，如 'viridis', 'plasma', 'inferno', 'magma' 等
    
    # 使用色彩映射给PCA结果着色
    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(len(tensor1_pca)), tensor1_pca, c=tensor1_pca, cmap=cmap, norm=norm, label="Tensor 1", alpha=0.7)
    plt.scatter(np.arange(len(tensor2_pca)), tensor2_pca, c=tensor2_pca, cmap=cmap, norm=norm, label="Tensor 2", alpha=0.7)
    
    # 添加标题和标签
    plt.title("1D PCA Visualization with Colormap")
    plt.xlabel("Channels")
    plt.ylabel("PCA Component Value")

    # 添加颜色条
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), label='PCA Component Value')

    # 添加图例
    plt.legend()

    # 保存图像为.png文件
    plt.savefig(output_filename, format='png')
    print(f"图像已保存为 {output_filename}")
    plt.close()
