import sys
import math

import torch
import torch.utils
import torchvision
from tqdm import tqdm

import time

import matplotlib.pyplot as plt

class DiffGaussRenderer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.N = 10_000
        self.W, self.H = (256, 256)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        fov_x = math.pi / 2.0 # Angle of the camera frustum 90°
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x) # Distance to Image Plane

        self.viewmat = torch.eye(4, device=self.device)
        self.viewmat[:3,3] = torch.tensor([0,0,-4])

        self.mu = (torch.rand((self.N,3), device=self.device) - 0.5) * 8.
        self.mu[:,2] = torch.rand((self.N))*0.001
        self.scales = torch.rand((self.N,3), device=self.device) * 0.1
        self.quats = torch.rand((self.N, 4), device=self.device)
        self.cols = torch.rand((self.N, 3), device=self.device)
        self.opcs = torch.rand((self.N,), device=self.device)

        self.params = [self.mu, self.scales, self.quats, self.cols, self.opcs]
        for param in self.params:
            param.requires_grad = True

        self.tile_size = 16

        self.img = torch.zeros(self.H, self.W, 3)

    def render(self):
        start = time.time()

        (
            self.mu_,
            self.cov_,
            self.z
        ) = self._project_gaussians(
            fx=self.focal,
            fy=self.focal,
            cx=self.W/2,
            cy=self.H/2,
        )

        project = time.time()

        self.tile_map = self._tile_gaussians(
            self.mu_, self.cov_
        )

        tile = time.time()

        # xs = torch.arange(len(self.tile_map[0]))
        # ys = torch.arange(len(self.tile_map))
        # x, y = torch.meshgrid(xs, ys)
        # x = x.reshape(1,-1); y = y.reshape(1, -1)

        # coords = torch.cat((x.reshape(1,-1),y.reshape(1,-1)), dim=0)

        # torch.vmap(self._rasterize_tile_faster, in_dims=1)(
        #     coords
        # )

        self.img = torch.zeros_like(self.img)
        for y in tqdm(range(len(self.tile_map))):
            for x in range(len(self.tile_map[0])):
                if self.tile_map[y][x]:
                    self._rasterize_tile_fast(
                        xys=self.mu_, covs=self.cov_,
                        x_coord=x, y_coord=y,
                    )

        render = time.time()

        print(f'Project: {project - start}, Tile: {tile - project}, Render: {render - tile}')

        return self.img[:]

    def plot_img(self):
        plt.matshow(self.img)
        plt.show()

    def forward(self, x):
        pass

    ### Project ##################################################################
    def _P(self, f_x, f_y, h, w, n, f):
        P = torch.tensor([
            [2.*f_x/w, 0., 0., 0.],
            [0., 2.*f_y/h, 0., 0.],
            [0., 0., (f+n)/(f-n), -2*f*n/(f-n)],
            [0., 0., 1., 0.],
        ], device=self.device)

        return P

    def _J(self, f_x, f_y, t_x, t_y, t_z):
        N = len(t_x)
        J = torch.zeros((N,2,3), device=self.device)
        J[:,0,0] = f_x/t_z
        J[:,1,1] = f_y/t_z
        J[:,0,2] = -f_x*t_x/t_z**2
        J[:,1,2] = -f_y*t_y/t_z**2

        return J

    def _quat_to_rot(self, quaternion):
        N = quaternion.shape[0]
        x, y, z, w = quaternion[:,0].clone() ,quaternion[:,1].clone() ,quaternion[:,2].clone() ,quaternion[:,3].clone(),

        R = torch.empty((N,3,3),device=self.device)
        
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

    def _project_gaussians(
        self,
        fx, 
        fy, 
        cx, 
        cy, 
        clip_thresh=0.01,
    ):
        # Project Means
        t = self.viewmat @ torch.cat((self.mu.T, torch.ones(1, self.N, device=self.device)), dim=0)
        t_ = self._P(fx, fy, self.H, self.W, clip_thresh, 10) @ t

        xys = torch.vstack((
            (self.W * t_[0] / t_[3]) / 2 + cx, 
            (self.H * t_[1] / t_[3]) / 2 + cy,
        )).T
        depths = t_[2]

        # Scale + Rot. to Cov.
        R = self._quat_to_rot(self.quats); S = torch.cat([torch.diag(s)[None] for s in self.scales])
        RS =  R @ S
        Sigma = RS @ RS.permute(0,2,1)

        # Project Cov
        J_ = self._J(fx, fy, t[0], t[1], t[2])
        R_cw = self.viewmat[:3,:3]
        covs = J_ @ R_cw @ Sigma @ R_cw.T @ J_.permute((0,2,1))

        # _, ind = torch.sort(depths)
        # xys, covs, self.cols = xys[ind], covs[ind], self.cols[ind]

        return xys, covs, depths 
    
    ### Tile #######################################################################
    def _get_eigenvalues(self, cov):
        a = cov[:,0,0]; b = cov[:,0,1]; d = cov[:,1,1]

        A = torch.sqrt(a*a - 2*a*d + 4*b**2 + d*d)
        B = a + d

        return 0.5 * (A + B), 0.5 * (-A + B)

    def _get_radii(self, cov):
        eigs = self._get_eigenvalues(cov)
        l1, l2 = eigs[0], eigs[1]
        s = 3
        return s * torch.sqrt(l1), s * torch.sqrt(l2)

    def _get_box(self, mu, cov):
        N = len(mu)

        r1, r2 = self._get_radii(cov)

        B = torch.empty((N, 4, 2))

        B[:,0,0] = mu[:,0] - r1; B[:,0,1] = mu[:,1] + r2
        B[:,1,0] = mu[:,0] + r1; B[:,1,1] = mu[:,1] + r2
        B[:,3,0] = mu[:,0] - r1; B[:,2,1] = mu[:,1] - r2
        B[:,2,0] = mu[:,0] + r1; B[:,3,1] = mu[:,1] - r2
        
        return B

    def _get_orientation(self, cov):
        a = cov[:,0,0]; b = cov[:,0,1]; c = cov[:,1,1]
        eigs = self._get_eigenvalues(cov)
        l1 = eigs[0]

        theta = torch.zeros_like(a)
        theta[(b == 0) & (a >= c)] = torch.pi/2
        theta[b != 0] = torch.atan2(l1 - a, b)

        return theta

    def _get_rotation(self, cov):
        theta = self._get_orientation(cov)

        cos = torch.cos(theta)
        sin = torch.sin(theta)

        R = torch.empty((len(cos),2,2))
        R[:,0,0] = cos
        R[:,0,1] = -sin
        R[:,1,0] = sin
        R[:,1,1] = cos

        return R

    def _get_bounding_boxes(self, xys, covs):
        rot = self._get_rotation(covs)

        box = self._get_box(xys, covs)
        box_mean = box.mean(dim=1, keepdim=True)

        rot_box = (rot @ (box - box_mean).permute((0,2,1))).permute((0,2,1)) + box_mean

        return rot_box

    def _tile_gaussians(self, xys, covs):
        assert self.H % self.tile_size == 0
        assert self.W % self.tile_size == 0
        
        # Compute Bounding-Boxes
        bbs = self._get_bounding_boxes(xys, covs)
        bbs[bbs.isnan()] = -1.

        tile_map = [[[] for tw in range(self.W//self.tile_size)] for th in range(self.H//self.tile_size)]

        minmax = torch.cat(((bbs.amin(dim=1) / self.tile_size).floor()[:,:,None], (bbs.amax(dim=1) / self.tile_size).ceil()[:,:,None]), dim=2)
        minmax[:,0].clamp_(0, self.W//self.tile_size - 1)
        minmax[:,1].clamp_(0, self.H//self.tile_size - 1)

        for i, g in enumerate(minmax.to(torch.long)):
            x_min, x_max = g[0]
            y_min, y_max = g[1]

            for x in range(x_min, x_max+1):
                for y in range(y_min, y_max+1):
                    tile_map[y][x].append(i)

        return tile_map
    
    ### Rasterize ##################################################################
    def _inv_2d(self, A: torch.Tensor):
        A_inv = A.new_empty(A.shape)
        A_inv[0,0] = A[1,1]
        A_inv[0,1] = -A[0,1]
        A_inv[1,0] = -A[1,0]
        A_inv[1,1] = A[0,0]

        A_inv *= 1/(A[0,0]*A[1,1]-A[0,1]*A[1,0])

        return A_inv

    def _g_fast(self, x, m, S):
        ''' x: (h*w, 2) matrix
            m: (2, 1) mean
            S: (2, 2) cov matrix
        '''
        x = x.T.view(-1, 1, 2)
        m = m.view(1, 1, 2)

        S_inv = self._inv_2d(S)

        x_m = x - m + 1e-5

        return torch.exp(-(1/2)*x_m @ (S_inv + 1e-5) @ x_m.permute(0,2,1) + 1e-5)
    
    def _rasterize_tile_fast(
        self,
        xys,
        covs,
        x_coord, 
        y_coord,
    ):
        x_min, x_max = x_coord*self.tile_size, (x_coord+1)*self.tile_size
        y_min, y_max = y_coord*self.tile_size, (y_coord+1)*self.tile_size
        
        x, y = torch.meshgrid(torch.arange(x_min, x_max, device=self.device), torch.arange(y_min, y_max, device=self.device))
        x = x.reshape(1,-1); y = y.reshape(1, -1)

        pixels_xy = torch.cat((x.reshape(1,-1),y.reshape(1,-1)), dim=0)

        tile = self.tile_map[y_coord][x_coord]

        xys = xys.clamp(min=0, max=float(self.H))

        C = self.cols[None, tile].clamp(min=0.0001, max=1.)
        O = self.opcs[None, tile].clamp(min=0.0001, max=1.)

        P = torch.vmap(self._g_fast, in_dims=0)(
            pixels_xy[None].expand((len(tile), pixels_xy.shape[0], pixels_xy.shape[1])), 
            xys[tile], covs[tile]).permute((1,0,2,3)).squeeze(dim=(2,3))
        P = P.clip(max=0.999, min=0.001)

        A = O * P
        A[A<0.001] = 0.
        A = A.clip(max=0.99)

        D = (1 - A)
        D = torch.cat((torch.ones(A.shape[0], 1, device=xys.device), D[:,:-1]), dim=1).contiguous()
        D = D
        D = D.cumprod(dim=1)

        # RGB = ((C + 1e-6) * (A[:,:,None] + 1e-6) * (D[:,:,None]+ 1e-6)) # numerical stability

        RGB = C * A[:,:,None]
        RGB = RGB * D[:,:,None]

        RGB = RGB.sum(dim=1).clamp(min=0.0001, max=1.)

        self.img[y_min:y_max, x_min:x_max] = RGB.view(self.tile_size, self.tile_size, 3).permute(1,0,2)

    def _rasterize_tile_faster(
        self,
        coord: torch.Tensor
    ):
        x_coord, y_coord = coord

        x_min, x_max = x_coord*self.tile_size, (x_coord+1)*self.tile_size
        y_min, y_max = y_coord*self.tile_size, (y_coord+1)*self.tile_size
        
        # xs = torch.arange(x_min, x_max, device=self.device)
        # ys = torch.arange(y_min, y_max, device=self.device)
        xs = torch.linspace(x_min, x_max-1, self.tile_size)
        ys = torch.linspace(y_min, y_max-1, self.tile_size)

        x, y = torch.meshgrid(xs, ys)
        
        x = x.reshape(1,-1); y = y.reshape(1, -1)

        pixels_xy = torch.cat((x.reshape(1,-1),y.reshape(1,-1)), dim=0)

        tile = self.tile_map[y_coord][x_coord]

        C = self.cols[None, tile]
        O = self.opcs[None, tile]

        P = torch.vmap(self._g_fast, in_dims=0)(
            pixels_xy[None].expand((len(tile), pixels_xy.shape[0], pixels_xy.shape[1])), 
            self.mu_[tile], self.cov_[tile]).permute((1,0,2,3)).squeeze(dim=(2,3))

        A = O * P
        D = (1 - A)
        D = torch.cat((torch.ones(A.shape[0], 1, device=self.device), D[:,:-1]), dim=1).contiguous()
        D = D.cumprod(dim=1)

        RGB = (C*A[:,:,None]*D[:,:,None])
        RGB = RGB.sum(dim=1)

        self.img[y_min:y_max, x_min:x_max] = RGB.view(self.tile_size, self.tile_size, 3).permute(1,0,2)

def main():
    import matplotlib.pyplot as plt

    renderer = DiffGaussRenderer()
    
    # gt_image = torch.ones((renderer.H, renderer.W, 3)) * 1.0
    # # make top left and bottom right red, blue
    # gt_image[: renderer.H // 2, : renderer.W // 2, :] = torch.tensor([1.0, 0.0, 0.0])
    # gt_image[renderer.H // 2 :, renderer.W // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])

    gt_image = torchvision.io.read_image('./mikey_cropped.jpg').permute(1,2,0) / 255

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(params=renderer.params, lr=1e-2)

    torch.autograd.set_detect_anomaly(True)

    pred = renderer.render()

    plt.ion()
    figure, ax = plt.subplots()
    im1 = ax.matshow(pred.detach().cpu())
    # plt.show()

    for iter in tqdm(range(100)):
        optimizer.zero_grad()

        pred = renderer.render()

        # plt.matshow(pred.detach().cpu())
        # plt.show()
        im1.set_data(pred.detach().cpu())
        figure.canvas.draw()
        figure.canvas.flush_events()
        time.sleep(0.1)

        loss = criterion(pred, gt_image)
        
        loss.backward()

        torch.nn.utils.clip_grad_value_(renderer.params, clip_value=1.0)
        optimizer.step()

        print(f'Iter: {iter}, Loss: {loss.item()}, Grad. Norms: {[p.abs().max().item() for p in renderer.params]}')

    plt.matshow(pred.detach().cpu())
    plt.show()

if __name__ == '__main__':
    main()