import numpy as np
from scipy import sparse
from tqdm import tqdm

# Imports for CPU parallelization
from joblib import Parallel, delayed

from RecSysFramework.Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from RecSysFramework.Recommenders.Recommender_utils import check_matrix

# Attempt to import PyTorch for GPU support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# <<<-------------------- CPU VERSION -------------------->>>
def _solve_item_factor_cpu(item_id, user_factors, urm_csc, conf_csc, reg, n_factors):
    """
    Solves for a single item factor vector using direct least squares on the CPU.
    """
    start_pos, end_pos = urm_csc.indptr[item_id], urm_csc.indptr[item_id + 1]
    interacting_users = urm_csc.indices[start_pos:end_pos]
    
    if len(interacting_users) == 0:
        return np.zeros(n_factors)

    confidence = conf_csc.data[start_pos:end_pos]
    Y = user_factors[interacting_users]
    
    lhs = Y.T @ (Y * confidence.reshape(-1, 1)) + np.eye(n_factors) * reg
    rhs = Y.T @ confidence
    
    return np.linalg.solve(lhs, rhs)


# <<<-------------------- GPU VERSION (PyTorch) -------------------->>>
def _solve_item_factor_pytorch(user_factors_device, interacting_users_idx, confidence, reg, n_factors, device):
    """
    Solves for a single item factor vector using PyTorch on a specified device.
    """
    if len(interacting_users_idx) == 0:
        return np.zeros(n_factors, dtype=np.float32)

    # Move necessary data for this item to the target device (e.g., GPU)
    confidence_device = torch.tensor(confidence, device=device, dtype=torch.float32)
    interacting_users_device = torch.from_numpy(interacting_users_idx).to(device)

    # Select corresponding user factor vectors directly on the device
    Y = user_factors_device[interacting_users_device]

    # Perform calculations on the device
    lhs = Y.T @ (Y * confidence_device.view(-1, 1)) + torch.eye(n_factors, device=device, dtype=torch.float32) * reg
    rhs = Y.T @ confidence_device

    # Solve the linear system
    new_item_factor_device = torch.linalg.solve(lhs, rhs)

    # Transfer result back to CPU and convert to NumPy array
    return new_item_factor_device.cpu().numpy()


# <<<-------------------- UNIFIED CLASS -------------------->>>
class ItemFactorLearner(BaseMatrixFactorizationRecommender):
    """
    A recommender that learns ONLY the item factors given fixed user factors.
    It can use either a parallelized CPU implementation or a GPU-accelerated
    one (PyTorch backend) via a flag in the `fit` method.
    """

    RECOMMENDER_NAME = "ItemFactorLearner"

    def __init__(self, URM_train):
        super(ItemFactorLearner, self).__init__(URM_train)
        self.URM_train_csc = check_matrix(self.URM_train, "csc")

    def fit(self, user_factors, alpha=20.0, reg=1e-2, use_gpu=False, n_jobs=-1):
        """
        Learns the item factors for the given fixed user factors.

        Args:
            user_factors (np.ndarray): Pre-computed user factors matrix.
            alpha (float): Confidence scaling factor.
            reg (float): Regularization constant.
            use_gpu (bool): If True, attempts to use the GPU. Falls back to CPU if not available.
            n_jobs (int): Number of CPU cores for parallel computation (if use_gpu=False).
        """
        self.USER_factors = user_factors.copy()
        self.num_factors = user_factors.shape[1]

        conf_matrix = self.URM_train_csc.copy()
        conf_matrix.data = 1.0 + alpha * conf_matrix.data
        
        # --- GPU Execution Path (PyTorch) ---
        gpu_is_available = TORCH_AVAILABLE and torch.cuda.is_available()
        if use_gpu:
            if not gpu_is_available:
                print("Warning: 'use_gpu=True' but PyTorch or CUDA is not available. Falling back to CPU.")
                return self.fit(user_factors, alpha, reg, use_gpu=False, n_jobs=n_jobs)
            
            try:
                device = torch.device("cuda")
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                
                user_factors_device = torch.tensor(self.USER_factors, device=device, dtype=torch.float32)
                item_factor_list = []
                
                print(f"Learning item factors for {self.n_items} items on the GPU...")
                for item_id in tqdm(range(self.n_items), desc="Solving for item factors on GPU"):
                    start, end = self.URM_train_csc.indptr[item_id], self.URM_train_csc.indptr[item_id + 1]
                    users = self.URM_train_csc.indices[start:end]
                    conf = conf_matrix.data[start:end]
                    
                    item_factor = _solve_item_factor_pytorch(user_factors_device, users, conf, reg, self.num_factors, device)
                    item_factor_list.append(item_factor)
                
                self.ITEM_factors = np.array(item_factor_list, dtype=np.float32)
                print("Item factors learned successfully using the GPU.")

            except Exception as e:
                print(f"An error occurred during GPU execution: {e}")
                print("Falling back to CPU implementation.")
                return self.fit(user_factors, alpha, reg, use_gpu=False, n_jobs=n_jobs)

        # --- CPU Execution Path ---
        else:
            print(f"Learning item factors for {self.n_items} items using {n_jobs if n_jobs!=-1 else 'all'} CPU cores...")
            
            item_factor_list = Parallel(n_jobs=n_jobs)(
                delayed(_solve_item_factor_cpu)(
                    item_id=item_id,
                    user_factors=self.USER_factors,
                    urm_csc=self.URM_train_csc,
                    conf_csc=conf_matrix,
                    reg=reg,
                    n_factors=self.num_factors
                ) for item_id in tqdm(range(self.n_items), desc="Solving for item factors on CPU")
            )
            
            self.ITEM_factors = np.array(item_factor_list)
            print("Item factors learned successfully using the CPU.")
