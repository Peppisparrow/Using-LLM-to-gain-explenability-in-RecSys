import numpy as np
from scipy import sparse
from joblib import Parallel, delayed
from tqdm import tqdm

from RecSysFramework.Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from RecSysFramework.Recommenders.Recommender_utils import check_matrix


def _solve_item_factor(item_id, user_factors, urm_csc, conf_csc, reg):
    """
    Solves for a single item factor vector using direct least squares.
    This function is designed to be called in parallel.
    """
    # Get users who interacted with this item and their confidence scores
    start_pos, end_pos = urm_csc.indptr[item_id], urm_csc.indptr[item_id + 1]
    interacting_users = urm_csc.indices[start_pos:end_pos]
    
    # If no users interacted with the item, return a zero vector
    if len(interacting_users) == 0:
        return np.zeros(user_factors.shape[1])

    confidence = conf_csc.data[start_pos:end_pos]
    
    # Get the corresponding user factor vectors
    Y = user_factors[interacting_users]

    # Pre-calculate YtY for just this item's users
    # YtY = Y.T @ Y
    
    # Calculate Yt(Cu-I)Y + YtY -> Yt Cu Y
    # This is the left-hand side of the least squares equation
    # A = Y.T @ (Y * (confidence - 1).reshape(-1, 1))
    # lhs = YtY + A + np.eye(Y.shape[1]) * reg
    lhs = Y.T @ (Y * confidence.reshape(-1, 1)) + np.eye(Y.shape[1]) * reg


    # Calculate the right-hand side of the equation: Yt Cu P_i
    # Here, P_i is a vector of 1s, so this simplifies to Yt @ confidence
    rhs = Y.T @ confidence

    # Solve the linear system and return the new item factor
    return np.linalg.solve(lhs, rhs)


class ItemFactorLearner(BaseMatrixFactorizationRecommender):
    """
    A recommender that learns ONLY the item factors given a fixed, pre-computed
    set of user factors.

    The optimal item factors are computed directly via least squares in a single,
    parallelized pass, making the process very fast.
    """

    RECOMMENDER_NAME = "ItemFactorLearner"

    def __init__(self, URM_train):
        super(ItemFactorLearner, self).__init__(URM_train)
        # Ensure the URM is in CSC format for efficient column (item) slicing
        self.URM_train_csc = check_matrix(self.URM_train, "csc")


    def fit(self, user_factors, alpha=20.0, reg=1e-2, n_jobs=-1):
        """
        Learns the item factors for the given fixed user factors.

        Args:
            user_factors (np.ndarray): The pre-computed matrix of user factors (n_users x n_factors).
            alpha (float): The confidence scaling factor.
            reg (float): The regularization constant (a.k.a lambda).
            n_jobs (int): The number of CPU cores to use for parallel computation. -1 uses all available cores.
        """
        # Store user factors and determine the number of factors from their shape
        self.USER_factors = user_factors
        self.num_factors = user_factors.shape[1]

        # 1. Build the confidence matrix: C = 1 + alpha * R
        conf_matrix = self.URM_train_csc.copy()
        conf_matrix.data = 1.0 + alpha * conf_matrix.data
        
        # 2. Solve for each item factor in parallel
        print(f"Learning item factors for {self.n_items} items using {n_jobs if n_jobs!=-1 else 'all'} cores...")
        
        item_factor_list = Parallel(n_jobs=n_jobs)(
            delayed(_solve_item_factor)(
                item_id=item_id,
                user_factors=self.USER_factors,
                urm_csc=self.URM_train_csc,
                conf_csc=conf_matrix,
                reg=reg
            ) for item_id in tqdm(range(self.n_items), desc="Solving for item factors")
        )
        
        # 3. Combine the results into the final item factors matrix
        self.ITEM_factors = np.array(item_factor_list)
        
        print("Item factors learned successfully.")