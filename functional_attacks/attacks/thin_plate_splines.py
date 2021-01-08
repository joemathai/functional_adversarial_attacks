import torch


class ThinPlateSplines(torch.nn.Module):
    """
    An implmentation of TPS based perturbation
    TPS reference: http://user.engineering.uiowa.edu/~aip/papers/bookstein-89.pdf
    """

    @staticmethod
    def all_pair_square_l2_norm(A, B):
        """
        Get pair-wise l2 norm
        https://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf
        :param A: shape batch_size, samples_a, dim
        :param B: shape batch_size, samples_b, dim
        :return: square of the l2 norm between pairwise points in A and B shape: batch_size, samples_a, samples_b
        """
        sqrA = torch.sum(torch.pow(A, 2), dim=2, keepdim=True).expand(A.shape[0], A.shape[1], B.shape[1])
        sqrB = torch.sum(torch.pow(B, 2), dim=2, keepdim=True).expand(B.shape[0], B.shape[1], A.shape[1]).permute(0, 2, 1)
        return torch.clamp(sqrA - 2 * torch.bmm(A, B.permute(0, 2, 1)) + sqrB, min=0)

    def __init__(self, batch_shape, src_pts=None, grid_scale_factor=7):
        """
        formulation of TPS
        x' = a0 + a1x +a2y + Σ f_i * U(||(xi,yi)-(x,y)||)
        y' = b0 + b1x +b2y + Σ g_i * U(||(xi,yi)-(x,y)||)
        where U(r) = r**2 ln(r**2)

        :param batch_shape: shape of the mini-batch N,C,H,W
        :param src_pts: shape batch_size x num_points x 2 (source control points) range [-1, 1]
        :param num_random_pts: if src_pts is None then pick random num_random_pts from the grid as src pts
        """
        super().__init__()
        batch_size, c, h, w = batch_shape
        self.batch_shape = batch_shape
        self.grid = torch.nn.functional.affine_grid(theta=torch.eye(2, 3).repeat(batch_size, 1, 1),
                                                    size=(batch_size, c, h, w),
                                                    align_corners=False)  # N, H, W, 2

        # if src_pts are not provided sample a uniform grid of points for perturbation
        if src_pts is None:
            src_pts = torch.nn.functional.affine_grid(theta=torch.eye(2, 3).repeat(batch_size, 1, 1),
                                                      size=(
                                                      batch_size, c, h // grid_scale_factor, w // grid_scale_factor),
                                                      align_corners=False).view(batch_size, -1, 2)  # N, H*W, 2

        _, num_src_pts, _ = src_pts.shape
        self.register_buffer("src_pts", src_pts, persistent=False)

        # matrix formed by src control points and radial basis functions
        P = torch.cat([torch.ones(batch_size, num_src_pts, 1, requires_grad=False), src_pts], dim=2)  # N, src_pts, 3
        R2 = ThinPlateSplines.all_pair_square_l2_norm(src_pts, src_pts)
        K = R2 * torch.log(R2 + 1e-10)  # N x src_pts x src_pts
        self.register_buffer("L", torch.cat([torch.cat([K, P], dim=2),
                                             torch.cat([P.permute(0, 2, 1),
                                                        torch.zeros(P.shape[0], 3, 3, requires_grad=False)], dim=2)],
                                            dim=1), persistent=False)

        # parameters to be updated
        # delta to be added to the source control points to get destination control points
        self.register_buffer("identity_params", torch.zeros(src_pts.shape, dtype=torch.float32, requires_grad=False),
                             persistent=False)
        self.xform_params = torch.nn.Parameter(torch.zeros(src_pts.shape, dtype=torch.float32, requires_grad=True) + 1e-8)

        # grid points and U(grid_points)
        self.register_buffer('grid_pts', torch.cat([torch.ones((batch_size, h * w, 1), device=self.grid.device), self.grid.view(batch_size, h * w, 2)], dim=2))
        grid_pts_r2 = ThinPlateSplines.all_pair_square_l2_norm(self.grid.view(batch_size, h * w, 2), self.src_pts)
        self.register_buffer('grid_pts_u', grid_pts_r2 * torch.log(grid_pts_r2 + 1e-10), persistent=False)  # N, grid_pts, src_pts

    def forward(self, imgs):
        n, c, h, w = imgs.shape
        dst_pts = self.src_pts + self.xform_params
        B_x = torch.cat([dst_pts[:, :, 0].unsqueeze(dim=2), torch.zeros((dst_pts.shape[0], 3, 1), device=imgs.device)], dim=1)
        B_y = torch.cat([dst_pts[:, :, 1].unsqueeze(dim=2), torch.zeros((dst_pts.shape[0], 3, 1), device=imgs.device)], dim=1)
        Y_xform, _ = torch.solve(B_y, self.L)
        X_xform, _ = torch.solve(B_x, self.L)
        F, A = X_xform[:, :-3, 0], X_xform[:, -3:, 0]
        G, B = Y_xform[:, :-3, 0], Y_xform[:, -3:, 0]
        # x' =  a0 + a1x +a2y + Σ F_i * U(||(xi,yi)-(x,y)||)
        grid_x = torch.sum(self.grid_pts * A.unsqueeze(dim=1).expand(A.shape[0], self.grid_pts.shape[1], A.shape[1]), dim=2) \
                 + torch.sum(F.unsqueeze(dim=1).expand(F.shape[0], self.grid_pts.shape[1], F.shape[1]) * self.grid_pts_u, dim=2)
        # y' =  b0 + b1x +b2y + Σ G_i * U(||(xi,yi)-(x,y)||)
        grid_y = torch.sum(self.grid_pts * B.unsqueeze(dim=1).expand(B.shape[0], self.grid_pts.shape[1], B.shape[1]), dim=2) \
                 + torch.sum(G.unsqueeze(dim=1).expand(G.shape[0], self.grid_pts.shape[1], G.shape[1]) * self.grid_pts_u, dim=2)
        modified_grid = torch.cat([grid_x.unsqueeze(dim=2), grid_y.unsqueeze(dim=2)], dim=2).view(n, h, w, 2)
        modified_img = torch.nn.functional.grid_sample(input=imgs, grid=modified_grid, mode='bilinear', align_corners=False)
        return modified_img
