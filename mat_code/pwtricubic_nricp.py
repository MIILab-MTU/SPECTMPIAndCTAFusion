# pwtricubic_nricp.py
# Piecewise Tricubic (Cubic B-Spline FFD) Non-Rigid ICP for 3D point clouds (Python/Numpy version)
# Author: ChatGPT
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass

def bspline_basis(u):
    u2 = u*u; u3 = u2*u
    Bm1 = (1 - 3*u + 3*u2 - u3) / 6.0
    B0  = (4 - 6*u2 + 3*u3) / 6.0
    B1  = (1 + 3*u + 3*u2 - 3*u3) / 6.0
    B2  = u3 / 6.0
    return np.stack([Bm1, B0, B1, B2], axis=-1)

def clamp_grid_index(i, n): return np.clip(i, 0, n-1)

@dataclass
class PWTricubicNRICP:
    grid_size: tuple = (8,8,8)
    levels: int = 2
    iters_per_level: int = 10
    lam: float = 1e-2
    margin: float = 0.05
    nn_k: int = 1
    robust_tau: float = 0.0
    seed: int = 0
    verbose: bool = True

    def _build_grid(self, pts, gs):
        mins = pts.min(0); maxs = pts.max(0)
        diag = np.linalg.norm(maxs - mins)
        mins -= self.margin*diag; maxs += self.margin*diag
        nx,ny,nz = gs
        gx = np.linspace(mins[0], maxs[0], nx)
        gy = np.linspace(mins[1], maxs[1], ny)
        gz = np.linspace(mins[2], maxs[2], nz)
        return (gx,gy,gz), mins, maxs

    def _point_local_coords(self, pts, grid_axes):
        gx,gy,gz = grid_axes
        sx = (gx[-1]-gx[0])/(len(gx)-1)
        sy = (gy[-1]-gy[0])/(len(gy)-1)
        sz = (gz[-1]-gz[0])/(len(gz)-1)
        fx = (pts[:,0]-gx[0])/sx; fy = (pts[:,1]-gy[0])/sy; fz = (pts[:,2]-gz[0])/sz
        ix = np.floor(fx).astype(int); iy = np.floor(fy).astype(int); iz = np.floor(fz).astype(int)
        ux = fx - ix; uy = fy - iy; uz = fz - iz
        return (ix,iy,iz), (ux,uy,uz), (sx,sy,sz)

    def _gather_weights_indices(self, idxs, us, grid_shape):
        ix,iy,iz = idxs; ux,uy,uz = us; nx,ny,nz = grid_shape
        wx = bspline_basis(ux); wy = bspline_basis(uy); wz = bspline_basis(uz)
        off = np.array([-1,0,1,2])
        N = ix.shape[0]
        W = np.empty((N,64),float); GI = np.empty((N,64),int); cnt=0
        for a in range(4):
            ia = clamp_grid_index(ix+off[a], nx); wa = wx[:,a]
            for b in range(4):
                jb = clamp_grid_index(iy+off[b], ny); wb = wy[:,b]
                for c in range(4):
                    kc = clamp_grid_index(iz+off[c], nz); wc = wz[:,c]
                    w = wa*wb*wc
                    gind = ia*(ny*nz) + jb*nz + kc
                    W[:,cnt]=w; GI[:,cnt]=gind; cnt+=1
        return W,GI

    
    def _deform(self, pts, disp_grid, grid_axes):
        (ix,iy,iz), (ux,uy,uz), _ = self._point_local_coords(pts, grid_axes)
        nx,ny,nz = disp_grid.shape[:3]
        W,GI = self._gather_weights_indices((ix,iy,iz),(ux,uy,uz),(nx,ny,nz))
        D = disp_grid.reshape((-1,3))
        # Gather 64-neighbor control displacements for each point and weight-sum
        # D[GI] -> (N,64,3); W -> (N,64)
        disp = (W[...,None] * D[GI]).sum(axis=1)
        return pts + disp, disp, (W,GI)

    def _build_laplacian(self, nx,ny,nz):
        N = nx*ny*nz; idx = np.arange(N).reshape(nx,ny,nz)
        rows=[]; cols=[]; data=[]
        def add(i,j,w): rows.append(i); cols.append(j); data.append(w)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    i = idx[x,y,z]; add(i,i,6.0)
                    for dx,dy,dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                        xx,yy,zz = x+dx,y+dy,z+dz
                        if 0<=xx<nx and 0<=yy<ny and 0<=zz<nz:
                            j = idx[xx,yy,zz]; add(i,j,-1.0)
        from scipy.sparse import coo_matrix
        return coo_matrix((data,(rows,cols)), shape=(N,N)).tocsr()

    def _huber_weights(self, r, tau):
        if tau <= 0: return np.ones_like(r)
        a = np.abs(r); w = np.ones_like(a); m = a>tau; w[m] = tau/a[m]; return w

    def fit(self, source_pts, target_pts, voxel=0.0):
        rng = np.random.default_rng(self.seed)
        src = np.asarray(source_pts,float); tar = np.asarray(target_pts,float)
        if voxel>0 and len(src)>0:
            keep = min(len(src), int(5e4))
            idx = rng.choice(len(src), size=keep, replace=False); src_d = src[idx]
        else:
            src_d = src

        gx,gy,gz = self.grid_size
        sched=[]; 
        for l in reversed(range(self.levels)): sched.append((max(3,gx>>l),max(3,gy>>l),max(3,gz>>l)))
        kdt = cKDTree(tar)
        disp_grid=None; grid_axes=None

        for li,gs in enumerate(sched):
            grid_axes,_,_ = self._build_grid(src_d, gs)
            nx,ny,nz = gs
            if disp_grid is None:
                disp_grid = np.zeros((nx,ny,nz,3),float)
            else:
                ox,oy,oz,_ = disp_grid.shape
                disp_grid = disp_grid.repeat(max(1,nx//ox),0).repeat(max(1,ny//oy),1).repeat(max(1,nz//oz),2)
                disp_grid = disp_grid[:nx,:ny,:nz,:]

            L = self._build_laplacian(nx,ny,nz); I = eye(nx*ny*nz,format='csr')
            for it in range(self.iters_per_level):
                deformed, disp, (W,GI) = self._deform(src_d, disp_grid, grid_axes)
                dists, idx = kdt.query(deformed, k=1, n_jobs=-1)
                nn = tar[idx]; r = (deformed - nn)
                w = self._huber_weights(np.linalg.norm(r,axis=1), self.robust_tau)
                Wd = diags(w,0,shape=(len(w),len(w)))
                Np = W.shape[0]; Ng = nx*ny*nz
                rows = np.repeat(np.arange(Np)[:,None], 64, axis=1).ravel()
                cols = GI.ravel(); data = W.ravel()
                A = coo_matrix((data,(rows,cols)), shape=(Np,Ng)).tocsr()
                R = L.T @ L
                b = nn - src_d
                AtW = A.T @ Wd
                lhs = AtW @ A + self.lam * R + 1e-6*I
                rhs_x = AtW @ b[:,0]; rhs_y = AtW @ b[:,1]; rhs_z = AtW @ b[:,2]
                dx = spsolve(lhs, rhs_x); dy = spsolve(lhs, rhs_y); dz = spsolve(lhs, rhs_z)
                d_grid = np.stack([dx,dy,dz],1).reshape(nx,ny,nz,3)
                disp_grid = 0.5*disp_grid + 0.5*d_grid
                if self.verbose and (it%2==0 or it==self.iters_per_level-1):
                    rmse = np.sqrt((w*(r*r).sum(1)).sum()/max(1e-9,w.sum()))
                    print(f"[Level {li+1}/{len(sched)}] iter {it+1}/{self.iters_per_level} rmse={rmse:.6f}")
        self.grid_axes_ = grid_axes; self.disp_grid_ = disp_grid
        return {"deformed": self.transform(src)["points"], "disp_grid": disp_grid, "grid_axes": grid_axes}

    def transform(self, points):
        pts = np.asarray(points,float)
        deformed, disp, _ = self._deform(pts, self.disp_grid_, self.grid_axes_)
        return {"points": deformed, "disp": disp}
