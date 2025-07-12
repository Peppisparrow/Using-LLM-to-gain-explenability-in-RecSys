import numpy as np
from scipy import sparse
from tqdm import tqdm
# Imports from your original code for compatibility
from RecSysFramework.Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from RecSysFramework.Recommenders.Recommender_utils import check_matrix
from implicit.als import AlternatingLeastSquares
# Imports for the implicit library backend
try:
    import implicit.gpu
    IMPLICIT_GPU_AVAILABLE = implicit.gpu.HAS_CUDA
except ImportError:
    IMPLICIT_GPU_AVAILABLE = False

class ImplicitUserFactorLearner(BaseMatrixFactorizationRecommender):
    """
    A recommender that learns ONLY the user factors given fixed item factors,
    using the optimized GPU backend from the 'implicit' library.
    """
    RECOMMENDER_NAME = "ImplicitUserFactorLearner"
    
    def __init__(self, URM_train):
        super(ImplicitUserFactorLearner, self).__init__(URM_train)
        
        if not IMPLICIT_GPU_AVAILABLE:
            raise ImportError("Implicit GPU backend is not available. Please ensure 'implicit' is installed with CUDA support.")
        
        # The implicit solver for user factors requires the user-item matrix (Cui)
        self.URM_train_Cui = check_matrix(self.URM_train, "csr")
        self.USER_factors = None
        self.ITEM_factors = None
    
    def fit(self, item_factors, alpha=20.0, reg=1e-2, cg_steps=3):
        """
        Learns the user factors for the given fixed item factors using the implicit GPU backend.
        
        Args:
            item_factors (np.ndarray): Pre-computed item factors matrix.
            alpha (float): Confidence scaling factor.
            reg (float): Regularization constant.
            cg_steps (int): Number of Conjugate Gradient steps for the solver. 3 is a common default.
        """
        self.ITEM_factors = item_factors.astype(np.float32)
        n_items, n_factors = self.ITEM_factors.shape
        n_users = self.URM_train_Cui.shape[0]
        
        print(f"Using implicit GPU backend to learn user factors for {n_users} users.")
        print(f"Item Factors: {n_items} items, {n_factors} factors.")
        
        # 1. Create the confidence matrix Cui = 1 + alpha * Rui
        # We work with the user-item matrix as required by the solver for user factors.
        conf_matrix_Cui = self.URM_train_Cui.copy()
        conf_matrix_Cui.data = 1.0 + alpha * conf_matrix_Cui.data
        
        # 2. Instantiate the GPU solver from the implicit library
        solver = implicit.gpu.LeastSquaresSolver()
        
        # 3. Move data to the GPU using implicit's wrappers
        item_factors_gpu = implicit.gpu.Matrix(self.ITEM_factors)
        Cui_gpu = implicit.gpu.CSRMatrix(conf_matrix_Cui)
        
        # 4. Pre-calculate the YtY term (Y.T @ Y + reg * I) on the GPU
        # This is a major optimization and is handled efficiently by the library.
        YtY = implicit.gpu.Matrix.zeros(n_factors, n_factors)
        solver.calculate_yty(item_factors_gpu, YtY, reg)
        
        # 5. Allocate memory for the output user factors on the GPU
        user_factors_gpu = implicit.gpu.Matrix.zeros(n_users, n_factors)  # type: ignore
        
        # 6. Run the solver for all users at once
        # This solves for X in the equation: X = (Yt C Y + reg*I)^-1 @ Yt C p
        # where C is the confidence diagonal matrix and p is the preference vector (implicitly 1s)
        tqdm.write("Solving for user factors on GPU...")
        solver.least_squares(Cui_gpu, user_factors_gpu, YtY, item_factors_gpu, cg_steps=cg_steps)
        tqdm.write("...done.")
        
        # 7. Copy the resulting user factors from the GPU back to the CPU (as a numpy array)
        self.USER_factors = user_factors_gpu.to_numpy()
        
        print("User factors learned successfully using the implicit GPU backend.")