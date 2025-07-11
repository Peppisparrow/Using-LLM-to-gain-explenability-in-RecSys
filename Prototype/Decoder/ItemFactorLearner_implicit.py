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


class ImplicitItemFactorLearner(BaseMatrixFactorizationRecommender):
    """
    A recommender that learns ONLY the item factors given fixed user factors,
    using the optimized GPU backend from the 'implicit' library.
    """

    RECOMMENDER_NAME = "ImplicitItemFactorLearner"

    def __init__(self, URM_train):
        super(ImplicitItemFactorLearner, self).__init__(URM_train)

        if not IMPLICIT_GPU_AVAILABLE:
            raise ImportError("Implicit GPU backend is not available. Please ensure 'implicit' is installed with CUDA support.")

        # The implicit solver for item factors requires the item-user matrix (Ciu)
        self.URM_train_Ciu = check_matrix(self.URM_train, "csr").T.tocsr()
        self.USER_factors = None
        self.ITEM_factors = None
        

    def fit(self, user_factors, alpha=20.0, reg=1e-2, cg_steps=3):
        """
        Learns the item factors for the given fixed user factors using the implicit GPU backend.

        Args:
            user_factors (np.ndarray): Pre-computed user factors matrix.
            alpha (float): Confidence scaling factor.
            reg (float): Regularization constant.
            cg_steps (int): Number of Conjugate Gradient steps for the solver. 3 is a common default.
        """
        self.USER_factors = user_factors.astype(np.float32)
        n_users, n_factors = self.USER_factors.shape
        n_items = self.URM_train_Ciu.shape[0]

        print(f"Using implicit GPU backend to learn item factors for {n_items} items.")
        print(f"User Factors: {n_users} users, {n_factors} factors.")

        # 1. Create the confidence matrix Ciu = 1 + alpha * Riu
        # We work with the transposed matrix as required by the solver for item factors.
        conf_matrix_Ciu = self.URM_train_Ciu.copy()
        conf_matrix_Ciu.data = 1.0 + alpha * conf_matrix_Ciu.data

        # 2. Instantiate the GPU solver from the implicit library
        solver = implicit.gpu.LeastSquaresSolver()

        # 3. Move data to the GPU using implicit's wrappers
        user_factors_gpu = implicit.gpu.Matrix(self.USER_factors)
        Ciu_gpu = implicit.gpu.CSRMatrix(conf_matrix_Ciu)

        # 4. Pre-calculate the XtX term (X.T @ X + reg * I) on the GPU
        # This is a major optimization and is handled efficiently by the library.
        XtX = implicit.gpu.Matrix.zeros(n_factors, n_factors)
        solver.calculate_yty(user_factors_gpu, XtX, reg)

        # 5. Allocate memory for the output item factors on the GPU
        item_factors_gpu = implicit.gpu.Matrix.zeros(n_items, n_factors) # type: ignore

        # 6. Run the solver for all items at once
        # This solves for Y in the equation: Y = (Xt C X + reg*I)^-1 @ Xt C p
        # where C is the confidence diagonal matrix and p is the preference vector (implicitly 1s)
        tqdm.write("Solving for item factors on GPU...")
        solver.least_squares(Ciu_gpu, item_factors_gpu, XtX, user_factors_gpu, cg_steps=cg_steps)
        tqdm.write("...done.")

        # 7. Copy the resulting item factors from the GPU back to the CPU (as a numpy array)
        self.ITEM_factors = item_factors_gpu.to_numpy()
        
        print("Item factors learned successfully using the implicit GPU backend.")