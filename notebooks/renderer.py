import torch
from tqdm import tqdm

### Project ##################################################################

def P(f_x, f_y, h, w, n, f):
    P = torch.tensor([
        [2.*f_x/w, 0., 0., 0.],
        [0., 2.*f_y/h, 0., 0.],
        [0., 0., (f+n)/(f-n), -2*f*n/(f-n)],
        [0., 0., 1., 0.],
    ])

    return P

def J(f_x, f_y, t_x, t_y, t_z):
    N = len(t_x)
    J = torch.zeros((N,2,3))
    J[:,0,0] = f_x/t_z
    J[:,1,1] = f_y/t_z
    J[:,0,2] = -f_x*t_x/t_z**2
    J[:,1,2] = -f_y*t_y/t_z**2

    return J

def quat_to_rot(quaternion):
    N = quaternion.shape[0]
    x, y, z, w = quaternion[:,0],quaternion[:,1],quaternion[:,2],quaternion[:,3],

    R = torch.empty((N,3,3))
    
    R[:,0,0] = 1-2*(y**2+z**2)
    R[:,0,1] = 2*(x*y-w*z)
    R[:,0,2] = 2*(x*z+w*y)

    R[:,1,0] = 2*(x*y+w*z)
    R[:,1,1] = 1-2*(x**2-z**2)
    R[:,1,2] = 2*(y*z-w*x)

    R[:,2,0] = 2*(x*z+w*y)
    R[:,2,1] = 2*(y*z+w*x)
    R[:,2,2] = 1-2*(x**2+y**2)

    return R

def project_gaussians(
        means3d, 
        scales, 
        # glob_scale, 
        quats, 
        colors, # for sorting only
        viewmat, 
        fx, 
        fy, 
        cx, 
        cy, 
        img_height, 
        img_width, 
        # block_width, 
        clip_thresh=0.01
    ):
    N = means3d.shape[0]

    # Project Means
    t = viewmat @ torch.cat((means3d.T, torch.ones(1, N)), dim=0) # (4, 4) x (4, N) = (4, N)
    t_ = P(fx, fy, img_height, img_width, clip_thresh, 10) @ t # (4, 4) x (4, N) = (4, N)

    xys = torch.vstack((
        (img_width*t_[0]/t_[3])/2+cx, # old: (img_width*t_[0]/t_[3]+1.)/2+cx,
        (img_height*t_[1]/t_[3])/2+cy, # (img_height*t_[1]/t_[3]+1.)/2+cy
    )).T
    depths = t_[2]

    # Scale + Rot. to Cov.
    R = quat_to_rot(quats); S = torch.cat([torch.diag(s)[None] for s in scales])
    RS =  R @ S
    Sigma = RS @ RS.permute(0,2,1)

    # Project Cov
    J_ = J(fx, fy, t[0], t[1], t[2])
    R_cw = viewmat[:3,:3]
    covs = J_ @ R_cw @ Sigma @ R_cw.T @ J_.permute((0,2,1))

    _, ind = torch.sort(depths)
    xys, covs, colors = xys[ind], covs[ind], colors[ind]

    return xys, covs, depths 

    # return xys, depths, radii, conics, compensation, num_tiles_hit, cov3d

### Tile #######################################################################
def get_eigenvalues(cov):
    a = cov[:,0,0]; b = cov[:,0,1]; d = cov[:,1,1]

    A = torch.sqrt(a*a - 2*a*d + 4*b**2 + d*d)
    B = a + d

    return 0.5 * (A + B), 0.5 * (-A + B)

def get_radii(cov):
    eigs = get_eigenvalues(cov)
    l1, l2 = eigs[0], eigs[1]
    s = 2
    return s * torch.sqrt(l1), s * torch.sqrt(l2) # TODO: change to 3 * sigma

def get_box(mu, cov):
    N = len(mu)

    r1, r2 = get_radii(cov)

    B = torch.empty((N, 4, 2))

    B[:,0,0] = mu[:,0] - r1; B[:,0,1] = mu[:,1] + r2
    B[:,1,0] = mu[:,0] + r1; B[:,1,1] = mu[:,1] + r2
    B[:,3,0] = mu[:,0] - r1; B[:,2,1] = mu[:,1] - r2
    B[:,2,0] = mu[:,0] + r1; B[:,3,1] = mu[:,1] - r2
    
    return B

def get_orientation(cov):
    a = cov[:,0,0]; b = cov[:,0,1]; c = cov[:,1,1]
    eigs = get_eigenvalues(cov)
    l1 = eigs[0]

    theta = torch.zeros_like(a)
    theta[(b == 0) & (a >= c)] = torch.pi/2
    theta[b != 0] = torch.atan2(l1 - a, b)

    return theta

def get_rotation(cov):
    theta = get_orientation(cov)

    cos = torch.cos(theta)
    sin = torch.sin(theta)

    R = torch.empty((len(cos),2,2))
    R[:,0,0] = cos
    R[:,0,1] = -sin
    R[:,1,0] = sin
    R[:,1,1] = cos

    return R

def get_bounding_boxes(xys, covs):
    rot = get_rotation(covs)

    box = get_box(xys, covs)
    box_mean = box.mean(dim=1, keepdim=True)

    rot_box = (rot @ (box - box_mean).permute((0,2,1))).permute((0,2,1)) + box_mean

    return rot_box

def tile_gaussians(xys, covs, tile_size, img_height, img_width):
    assert img_height % tile_size == 0
    assert img_width % tile_size == 0
    
    # Compute Bounding-Boxes
    bbs = get_bounding_boxes(xys, covs)
    tile_map = [[[] for tw in range(img_width//tile_size)] for th in range(img_height//tile_size)]

    minmax = torch.cat(((bbs.amin(dim=1) / tile_size).floor()[:,:,None], (bbs.amax(dim=1) / tile_size).ceil()[:,:,None]), dim=2)
    minmax[:,0].clamp_(0, img_width//tile_size - 1)
    minmax[:,1].clamp_(0, img_height//tile_size - 1)

    for i, g in enumerate(minmax.to(torch.long)):
        x_min, x_max = g[0]
        y_min, y_max = g[1]

        for x in range(x_min, x_max+1):
            for y in range(y_min, y_max+1):
                tile_map[y][x].append(i)

    return tile_map

### Rasterize ##################################################################

def inv_2d(A):
    A_inv = torch.tensor([
        [A[1,1], -A[0,1]],
        [-A[1,0], A[0,0]],
    ])
    A_inv *= 1/(A[0,0]*A[1,1]-A[0,1]*A[1,0])

    return A_inv

def g(x, m, S):
    ''' x: (h*w, 2) matrix
        m: (2, 1) mean
        S: (2, 2) cov matrix
    '''
    
    x = x.T.view(-1, 1, 2)
    m = m.view(1, 1, 2)

    S_inv = inv_2d(S)
    x_m = x - m

    return torch.exp(-(1/2)*x_m @ S_inv @ x_m.permute(0,2,1))

def rasterize_gaussians(
        xys, 
        depths, 
        covs, 
        conics, 
        num_tiles_hit, 
        colors, 
        opacity, 
        img_height, 
        img_width, 
        block_width, 
        background=None, 
        return_alpha=False
    ):
    x, y = torch.meshgrid(torch.linspace(0,img_width,img_width),torch.linspace(0,img_height,img_height), indexing='xy')
    x = x.reshape(1,-1); y = y.reshape(1, -1)

    # Sort mu_ by depth
    _, ind = torch.sort(depths)
    xys_, covs_, colors_ = xys[ind], covs[ind], colors[ind]

    out_img = torch.zeros(img_height, img_width, 3)
    pixels_xy = torch.cat((x.reshape(1,-1),y.reshape(1,-1)), dim=0)
    cum_alphas = torch.ones(1, img_height, img_width)
    for m, S, c, o in tqdm(zip(xys_, covs_, colors_, opacity), total=len(xys_)):
        alpha = g(pixels_xy, m, S).view(1, img_height, img_width) * o
        out_img = out_img + (alpha * c.view(3,1,1) * cum_alphas).permute((1,2,0)).flip(dims=(0,))
        cum_alphas = cum_alphas * (1 - alpha)

    return out_img

def rasterize_tile(
        xys,
        covs,
        colors,
        opacity,
        x_coord, 
        y_coord,
        tile_size,
        out_img,
        tile_map
    ):
    x_min, x_max = x_coord*tile_size, (x_coord+1)*tile_size
    y_min, y_max = y_coord*tile_size, (y_coord+1)*tile_size
    
    x, y = torch.meshgrid(torch.arange(x_min, x_max), torch.arange(y_min, y_max))
    x = x.reshape(1,-1); y = y.reshape(1, -1)

    pixels_xy = torch.cat((x.reshape(1,-1),y.reshape(1,-1)), dim=0)

    cum_alphas = torch.ones(1, tile_size, tile_size)
    # for m, S, c, o in zip(xys, covs, colors, opacity):
    for tile in tile_map[y_coord][x_coord]:
        m, S, c, o = xys[tile], covs[tile], colors[tile], opacity[tile]
        alpha = g(pixels_xy, m, S).view(1, tile_size, tile_size) * o
        out_img[y_min:y_max, x_min:x_max] = out_img[y_min:y_max, x_min:x_max] + (alpha * c.view(3,1,1) * cum_alphas).permute((2,1,0))
        cum_alphas = cum_alphas * (1 - alpha)

    # return out_img

def main():
    N = 1_000

    mu = (torch.rand((N,3)) - 0.5) * 4.
    scale = torch.rand((N,3)) * 0.2
    quat = torch.rand((N, 4))
    col = torch.rand((N, 3))
    opc = torch.rand((N,))

    # Output Image Width and Height
    W = 1290 
    H =  720

    fov_x = math.pi / 2.0 # Angle of the camera frustum 90°
    focal = 0.5 * float(W) / math.tan(0.5 * fov_x) # Distance to Image Plane

    viewmat = torch.eye(4)
    viewmat[:3,3] = torch.tensor([0,0,-4])

    (
        mu_,
        cov_,
        z
    ) = project_gaussians(
        means3d=mu,
        scales=scale,
        quats=quat,
        viewmat=viewmat,
        fx=focal,
        fy=focal,
        cx=W/2,
        cy=H/2,
        img_height=H,
        img_width=W
    )

    out_img = rasterize_gaussians(
        xys=mu_,
        depths=z,
        covs=cov_,
        conics=None,
        num_tiles_hit=None,
        colors=col,
        opacity=opc,
        img_height=H,
        img_width=W,
        block_width=None,
        background=None,
        return_alpha=None
    )

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.matshow(out_img)

    ellipse_ndim(mu_, cov_, ax2, edgecolor='red')
    for m in mu_:
        ax2.scatter(m[0], m[1])
    # ax2.scatter(mu_[1,0], mu_[1,1])
    ax2.set_aspect('equal', adjustable='box')

    ax2.set_xlim([0,W])
    ax2.set_ylim([0,H])

    plt.show()

if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt
    import math
    from utils import ellipse_ndim
    from tqdm import tqdm

    main()