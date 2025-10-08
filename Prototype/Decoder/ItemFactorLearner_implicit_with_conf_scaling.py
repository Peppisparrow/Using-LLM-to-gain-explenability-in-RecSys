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

    RECOMMENDER_NAME = "ImplicitItemFactorLearner_CS"

    def __init__(self, URM_train):
        super(ImplicitItemFactorLearner, self).__init__(URM_train)

        if not IMPLICIT_GPU_AVAILABLE:
            raise ImportError("Implicit GPU backend is not available. Please ensure 'implicit' is installed with CUDA support.")

        # The implicit solver for item factors requires the item-user matrix (Ciu)
        self.URM_train_Ciu = check_matrix(self.URM_train, "csr").T.tocsr()
        self.USER_factors = None
        self.ITEM_factors = None
        
    def _linear_scaling_confidence(self, alpha=1.0):

        C = check_matrix(self.URM_train_Ciu, format="csr", dtype = np.float32)
        C.data = 1.0 + alpha*C.data

        return C

    def _log_scaling_confidence(self, alpha=1.0, epsilon=1e-6):

        C = check_matrix(self.URM_train_Ciu, format="csr", dtype = np.float32)
        C.data = 1.0 + alpha * np.log(1.0 + C.data / epsilon)

        return C
    
    def _confidence_scaling(self, alpha=1.0, confidence_scaling='linear'):
        """
        Scales the confidence of the interactions in the URM_train matrix.
        
        Parameters:
        - alpha: Scaling factor for the confidence.
        - confidence_scaling: Type of scaling to apply ('linear' or 'log').
        - **confidence_args: Additional arguments for confidence scaling.
        
        Returns:
        - C: Scaled confidence matrix.
        """
        if confidence_scaling == 'linear':
            return self._linear_scaling_confidence(alpha)
        elif confidence_scaling == 'log':
            return self._log_scaling_confidence(alpha, epsilon=1)
        else:
            raise ValueError("Invalid confidence scaling method. Use 'linear' or 'log'.")
        

    def fit(self, user_factors, reg=1e-2, cg_steps=3, **confidence_args):
        """
        Learns the item factors for the given fixed user factors using the implicit GPU backend.

        Args:
            user_factors (np.ndarray): Pre-computed user factors matrix.
            reg (float): Regularization constant.
            cg_steps (int): Number of Conjugate Gradient steps for the solver. 3 is a common default.
            confidence_args (dict): Arguments for confidence scaling, e.g., {'alpha': 40, 'confidence_scaling': 'log', 'linear'}.
        """
        self.USER_factors = user_factors.astype(np.float32)
        n_users, n_factors = self.USER_factors.shape
        n_items = self.URM_train_Ciu.shape[0]

        print(f"Using implicit GPU backend to learn item factors for {n_items} items.")
        print(f"User Factors: {n_users} users, {n_factors} factors.")

        # 1. Create the confidence matrix Ciu = 1 + alpha * Riu
        # We work with the transposed matrix as required by the solver for item factors.
        conf_matrix_Ciu = self._confidence_scaling(**confidence_args)

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
        
    def _compute_item_score(self, user_id_array, items_to_compute = None):

        item_scores = self.USER_factors[user_id_array] @ self.ITEM_factors.T
        return item_scores