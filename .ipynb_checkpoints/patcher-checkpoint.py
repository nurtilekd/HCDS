patch_size = 64
patches_per_image = 16
use_event_sampling = False
event_thresh = None

class PatchDataset(Dataset):
    def __init__(self, X, Y, indices, patch=64, ppi=16,
                 use_event=False, event_thresh=None, p_event=0.5, max_tries=10):
        self.X = X
        self.Y = Y
        self.indices = np.asarray(indices)
        self.patch = patch
        self.ppi = ppi
        self.use_event = use_event
        self.event_thresh = event_thresh
        self.p_event = p_event
        self.max_tries = max_tries

        _, _, self.H, self.W = X.shape
        if self.H < patch or self.W < patch:
            raise ValueError(f"Patch size {patch} too large for input dimensions {self.H}x{self.W}.")

        if self.use_event and (self.event_thresh is None):
            raise ValueError(f"When (use_event=True), (event_thresh) must be set.")

    def __len__(self):
        return len(self.indices) * self.ppi

    def __getitem__(self, i):
        idx = self.indices[i // self.ppi]
        x = self.X[idx]
        y = self.Y[idx]

        top = np.random.randint(0, self.H - self.patch + 1)
        left = np.random.randint(0, self.W - self.patch + 1)
        
        if self.use_event and (np.random.rand() < self.p_event):
            for _ in range(self.max_tries):
                t = np.random.randint(0, self.H - self.patch + 1)
                l = np.random.randint(0, self.W - self.patch + 1)
                y_try = y[:, t:t+self.patch, l:l+self.patch]
                if np.nanmax(y_try) >= self.event_thresh:
                    top, left = t, l
                    break

        y_patch = y[:, top:top+self.patch, left:left+self.patch]
        x_patch = x[:, top:top+self.patch, left:left+self.patch]
        return torch.from_numpy(x_patch), torch.from_numpy(y_patch)