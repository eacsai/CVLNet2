import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation
import matplotlib.cm as cm
import cv2

def _verts_to_dirs_square(pt_a, pt_b, pt_c, pt_d, gen_res):
    """
    Creates a grid of directions from origin across the square face.
    """
    p0, p1, p2, p3 = pt_a, pt_b, pt_c, pt_d

    # Convert to torch
    p0 = torch.from_numpy(p0.astype(np.float32))
    p1 = torch.from_numpy(p1.astype(np.float32))
    p2 = torch.from_numpy(p2.astype(np.float32))
    p3 = torch.from_numpy(p3.astype(np.float32))

    # Face center
    pt_center = (p0 + p1 + p2 + p3) / 4.0

    # down_vec = bottom_mid - top_mid
    bottom_mid = 0.5 * (p0 + p1)
    top_mid    = 0.5 * (p2 + p3)
    down_vec   = bottom_mid - top_mid
    if down_vec[2] > 0.0:
        down_vec = -down_vec

    # right_vec = right_edge_mid - left_edge_mid
    right_mid = 0.5 * (p0 + p3)
    left_mid  = 0.5 * (p1 + p2)
    right_vec = right_mid - left_mid

    # normalize & scale
    d_len = torch.linalg.norm(down_vec, ord=2).item()
    r_len = torch.linalg.norm(right_vec, ord=2).item()
    down_vec  = down_vec  / d_len
    right_vec = right_vec / r_len 

    # base corner
    pt_base = pt_center - down_vec - right_vec

    # double them
    down_vec  = down_vec  * 2
    right_vec = right_vec * 2

    # meshgrid
    ii, jj = torch.meshgrid(
        torch.linspace(0.5 / gen_res, 1.0 - 0.5 / gen_res, gen_res),
        torch.linspace(0.5 / gen_res, 1.0 - 0.5 / gen_res, gen_res),
        indexing='ij'
    )

    # center sample point
    to_vec = pt_base + 0.5 * down_vec + 0.5 * right_vec

    # all sample points
    dirs = (
        pt_base[None, None, :] +
        down_vec[None, None, :]  * ii[:, :, None] +
        right_vec[None, None, :] * jj[:, :, None]
    )

    dist_sample = torch.linalg.norm(dirs, dim=-1, keepdim=True)
    dist_center = torch.linalg.norm(to_vec, dim=-1, keepdim=True)
    pers_ratios = dist_sample / dist_center

    # normalize
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)

    return dirs, pers_ratios, to_vec, 0.5 * down_vec, 0.5 * right_vec

def direction_to_pano_coord(dirs):
    dirs = dirs / torch.linalg.norm(dirs, 2, -1, True)
    beta = torch.arcsin(dirs[..., 2])
    xy = dirs[..., :2] / torch.cos(beta)[..., None]
    alpha = torch.view_as_complex(xy).angle()
    return torch.stack([beta, alpha], -1)

def pano_to_img_coord(coords):
    y, x = coords[..., 0], coords[..., 1]
    return torch.stack([-y / np.pi + 0.5, -x / (2. * np.pi) + 0.5], -1)

def direction_to_img_coord(dirs):
    return pano_to_img_coord(direction_to_pano_coord(dirs))

def img_coord_to_sample_coord(coords):
    return torch.stack([coords[..., 1], coords[..., 0]], -1) * 2. - 1.

def cube():
    h = 1.0 / np.sqrt(3.0)  # 0.577350269

    # 8 vertices, each on radius=1 sphere
    #    x   y   z
    vertices = np.array([
        [ +h, +h, +h ],  # 0
        [ -h, +h, +h ],  # 1
        [ -h, +h, -h ],  # 2
        [ +h, +h, -h ],  # 3

        [ +h, -h, +h ],  # 4
        [ -h, -h, +h ],  # 5
        [ -h, -h, -h ],  # 6
        [ +h, -h, -h ],  # 7
    ], dtype=np.float32)

    faces = [
        [7, 3, 0, 4],  # front   (x=+h)
        [2, 6, 5, 1],  # back    (x=-h)
        [3, 2, 1, 0],  # left    (y=+h)
        [6, 7, 4, 5],  # right   (y=-h)
        [1, 5, 4, 0],  # top     (z=+h)
        [6, 2, 3, 7],  # bottom  (z=-h)
    ]
    return vertices, faces

# Main function
def split_panorama(panorama, gen_res=320, device='cpu'):

    vertices, faces = cube()
    # vertices =  np.array(vertices, dtype=np.float32)
    
    vertices = torch.tensor(vertices, dtype=torch.float32)
    ang = np.arctan(0)
    rot_vec = np.array([ang, ang, ang])
    rot = Rotation.from_rotvec(rot_vec)
    vertices = rot.apply(vertices)
    vertices = vertices.astype(np.float32)
    pers_imgs = []

    # Generate coords for each face
    all_dirs = []
    all_ratios = []
    to_vecs = []
    down_vecs = []
    right_vecs = []

    for face in faces:
        pt_a, pt_b, pt_c, pt_d = vertices[face[0]], vertices[face[1]], vertices[face[2]], vertices[face[3]]
        
        dirs, ratios, to_vec, down_vec, right_vec = _verts_to_dirs_square(pt_a, pt_b, pt_c, pt_d, gen_res=gen_res)
        all_dirs.append(dirs)
        all_ratios.append(ratios)
        to_vecs.append(to_vec)
        down_vecs.append(down_vec)
        right_vecs.append(right_vec)

    pers_dirs = torch.stack(all_dirs, 0).to(device)
    pers_ratios = torch.stack(all_ratios, 0).to(device)
    to_vecs = torch.stack(to_vecs, 0).to(device)
    down_vecs = torch.stack(down_vecs, 0).to(device)
    right_vecs = torch.stack(right_vecs, 0).to(device)
    

    fx = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(right_vecs, 2, -1, True) * .5
    fy = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(down_vecs, 2, -1, True) * .5
    cx = torch.ones_like(fx) * .5
    cy = torch.ones_like(fy) * .5

    camera_k = torch.tensor([[[fx[0][0],   0.0000, cx[0][0]],
                              [0.0000, fy[0][0], cy[0][0]],
                              [0.0000,   0.0000,   1.0000]]], 
                            dtype=torch.float32, requires_grad=False, device=device)

    rot_w2c = torch.stack([right_vecs / torch.linalg.norm(right_vecs, 2, -1, True),
                            down_vecs / torch.linalg.norm(down_vecs, 2, -1, True),
                            to_vecs / torch.linalg.norm(to_vecs, 2, -1, True)],
                            dim=1)
    rot_c2w = rot_w2c.transpose(1, 2)  # 或使用 rot_w2c.t()
    
    extrinsic_matrix_4x4 = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(6,1,1)  # 形状: [4, 4]
    extrinsic_matrix_4x4[:, :3, :3] = rot_c2w

    # quaternion = Rotation.from_matrix(rot_w2c.numpy()).as_quat() 
    # print(quaternion, to_vecs / torch.linalg.norm(to_vecs, 2, -1, True))
    
    n_pers = len(pers_dirs)
    img_coords = direction_to_img_coord(pers_dirs)
    sample_coords = img_coord_to_sample_coord(img_coords)

    for b in range(panorama.shape[0]):
        pers_imgs.append(F.grid_sample(panorama[b][None].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border', align_corners=True)) # [n_pers, 3, gen_res, gen_res]

    pers_imgs = torch.stack(pers_imgs, 0)
    return pers_imgs, extrinsic_matrix_4x4, camera_k

# Image I/O functions
def load_panorama(image_path):
    """
    Load a panorama image and convert it to a PyTorch tensor.
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
    return img_tensor

def save_sub_images(pers_imgs, output_dir):
    """
    Save each sub-image to the specified directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists
    for i, img_tensor in enumerate(pers_imgs):
        img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        img.save(output_dir / f"perspective_{i + 1}.png")


def tensor_to_cv2_image(tensor):
    """
    将形状为 [3, H, W] 的 PyTorch 张量转换为适用于 OpenCV 的 [H, W, 3] 图像。

    Args:
        tensor (torch.Tensor): 输入张量，形状为 [3, H, W]。

    Returns:
        np.ndarray: 转换后的图像，形状为 [H, W, 3]，数据类型为 uint8。
    """
    # 确保输入是一个 PyTorch 张量
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("输入必须是一个 PyTorch 张量。")

    # 检查张量形状是否为 [3, H, W]
    if tensor.dim() != 3 or tensor.size(0) != 3:
        raise ValueError("输入张量必须具有形状 [3, H, W]。")

    # 1. 将张量移动到 CPU 并断开与计算图的连接
    tensor = tensor.cpu().detach()

    # 2. 转换为 NumPy 数组
    img = tensor.numpy()

    # 3. 调整维度顺序从 [C, H, W] 到 [H, W, C]
    img = np.transpose(img, (1, 2, 0))

    # 4. 如果图像是 RGB 格式，转换为 BGR 格式
    #    这一步是为了与 OpenCV 的颜色顺序一致
    img = img[..., ::-1]

    # 5. 调整数据类型和范围
    if img.dtype != np.uint8:
        # 判断张量的最大值，以决定是否需要乘以 255
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    return img

def showDepth(depth, raw_image):
    cmap = cm.Spectral
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.cpu().detach().numpy()
    depth = depth.astype(np.uint8)
    
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    
    output_path = './depth.png'

    split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
    combined_result = cv2.hconcat([raw_image, split_region, depth])
    cv2.imwrite(output_path, combined_result)

if __name__ == "__main__":
    from pathlib import Path

    # Define input and output paths
    panorama_path = "image.png"  # Replace with the path to your panorama image
    output_dir = Path("output_perspectives_CUBE")

    # Load the panorama image
    panorama = load_panorama(panorama_path)
    print(f"Loaded panorama of shape: {panorama.shape}")

    # Split the panorama into 20 sub-images
    pers_imgs = split_panorama(panorama)
    print(f"Generated {pers_imgs.shape[0]} sub-images of shape {pers_imgs.shape[1:]}")

    # Save the sub-images
    save_sub_images(pers_imgs, output_dir)
    print(f"Saved sub-images to {output_dir}")
