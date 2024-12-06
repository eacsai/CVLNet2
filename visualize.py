# import numpy as np
# from sklearn.decomposition import PCA
# from PIL import Image

# def reshape_normalize(x):
#     '''
#     Args:
#         x: [B, C, H, W]

#     Returns:

#     '''
#     B, C, H, W = x.shape
#     x = x.transpose([0, 2, 3, 1]).reshape([-1, C])

#     denominator = np.linalg.norm(x, axis=-1, keepdims=True)
#     denominator = np.where(denominator==0, 1, denominator)
#     return x / denominator

# def normalize(x):
#     denominator = np.linalg.norm(x, axis=-1, keepdims=True)
#     denominator = np.where(denominator == 0, 1, denominator)
#     return x / denominator

# def single_features_to_RGB(sat_features, img_name='test_img.png'):
#     sat_feat = sat_features[:1,:,:,:].data.cpu().numpy()
#     # 1. 重塑特征图形状为 [256, 64*64]
#     B, C, H, W = sat_feat.shape
#     flatten = np.concatenate([sat_feat], axis=0)
#     # 2. 进行 PCA 降维到 3 维
#     pca = PCA(n_components=3)
#     pca.fit(reshape_normalize(flatten))
    
#     # 3. 归一化到 [0, 1] 范围
#     sat_feat_new = ((normalize(pca.transform(reshape_normalize(sat_feat))) + 1 )/ 2).reshape(B, H, W, 3)

#     sat = Image.fromarray((sat_feat_new[0] * 255).astype(np.uint8))
#     # sat = sat.resize((512, 512))
#     sat.save(img_name)

# def sat_features_to_RGB(sat_features, grd_features):
#     sat_feat = sat_features[:1,:,:,:].data.cpu().numpy()
#     grd_feat = grd_features[:1,:,:,:].data.cpu().numpy()
#     # 1. 重塑特征图形状为 [256, 64*64]
#     B, C, A, A = sat_feat.shape
#     _, _, H, W = grd_feat.shape
#     flatten = np.concatenate([sat_feat.reshape(B, C, -1), grd_feat.reshape(B, C, -1)], axis=0)
#     # 2. 进行 PCA 降维到 3 维
#     pca = PCA(n_components=3)
#     pca.fit(normalize(flatten.transpose([0,2,1]).reshape(-1, C)))
    
#     # 3. 归一化到 [0, 1] 范围
#     sat_feat_new = ((normalize(pca.transform(reshape_normalize(sat_feat))) + 1 )/ 2).reshape(B, A, A, 3)
#     fuse_feat_new = ((normalize(pca.transform(reshape_normalize(grd_feat))) + 1 )/ 2).reshape(B, H, W, 3)

#     sat = Image.fromarray((sat_feat_new[0] * 255).astype(np.uint8))
#     sat = sat.resize((A, A))
#     sat.save('sat_feat.png')
    
#     grd = Image.fromarray((fuse_feat_new[0] * 255).astype(np.uint8))
#     grd = grd.resize((W, H))
#     grd.save('grd_feat.png')