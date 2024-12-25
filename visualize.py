import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import os

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

def single_features_to_RGB(sat_features, img_name='test_img.png'):
    sat_feat = sat_features[:1,:,:,:].data.cpu().numpy()
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

def sat_features_to_RGB(sat_features, grd_features):
    sat_feat = sat_features[:1,:,:,:].data.cpu().numpy()
    grd_feat = grd_features[:1,:,:,:].data.cpu().numpy()
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