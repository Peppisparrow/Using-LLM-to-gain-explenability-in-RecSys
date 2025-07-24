'''
@author Nicola Cesare
It is a wrapper of the implicit recommender (https://github.com/benfred/implicit).
@author: Luca Pagano, Giuseppe Vitello: Added some modifications to make it work.
'''


import implicit
from implicit.als import AlternatingLeastSquares
import numpy as np

from RecSysFramework.Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from RecSysFramework.Recommenders.Recommender_utils import check_matrix

class ImplicitALSRecommender(BaseMatrixFactorizationRecommender):
    """ImplicitALSRecommender recommender"""
    
    def __init__(self, URM_train, verbose=True):
        super().__init__(URM_train, verbose=verbose)

        # Initialize the recommender
        self.model = None

        # User and item factors
        self.USER_factors = None
        self.ITEM_factors = None

    RECOMMENDER_NAME = "ImplicitALSRecommender"
    
    def _linear_scaling_confidence(self, alpha=1.0):

        C = check_matrix(self.URM_train, format="csr", dtype = np.float32)
        C.data = 1.0 + alpha*C.data

        return C

    def _log_scaling_confidence(self, alpha=1.0, epsilon=1e-6):

        C = check_matrix(self.URM_train, format="csr", dtype = np.float32)
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

    def fit(self,
            factors=100,
            regularization=0.01,
            use_native=True, use_cg=True, use_gpu=False,
            iterations=15,
            calculate_training_loss=False, num_threads=0,
            item_factors=None, user_factors=None,
            **confidence_args
            ):
        
        print("Implicit cuda support: ", implicit.gpu.HAS_CUDA)
        
        if use_gpu and not implicit.gpu.HAS_CUDA:
            raise ValueError("ImplicitALSRecommender: GPU support is requested but implicit.gpu.HAS_CUDA is False. "
                             "Please ensure that you have a compatible GPU and the necessary libraries installed.")
        
        self.model = AlternatingLeastSquares(factors=factors, regularization=regularization,
                                                        use_native=use_native, use_cg=use_cg, use_gpu=use_gpu,
                                                        iterations=iterations,
                                                        calculate_training_loss=calculate_training_loss,
                                                        num_threads=num_threads,
                                                        random_state=42
                                                        )
        # if item_factors is not None:
        #     self.model.item_factors = implicit.gpu.Matrix(item_factors)
        # if user_factors is not None:
        #     self.model.user_factors = implicit.gpu.Matrix(user_factors)
        
        self.model = self.model.to_cpu() if use_gpu else self.model
        
        self.model.item_factors = item_factors if item_factors is not None else self.model.item_factors
        self.model.user_factors = user_factors if user_factors is not None else self.model.user_factors
        
        self.model = self.model.to_gpu() if use_gpu else self.model

        C = self._confidence_scaling(**confidence_args)
        
        self.model.fit(C, show_progress=self.verbose)

        if use_gpu:
            # Convert the user and item factors to numpy arrays if using GPU
            self.USER_factors = self.model.user_factors.to_numpy()
            self.ITEM_factors = self.model.item_factors.to_numpy()
        else:
            self.USER_factors = self.model.user_factors
            self.ITEM_factors = self.model.item_factors
        
    def _compute_item_score(self, user_id_array, items_to_compute = None):

        item_scores = self.USER_factors[user_id_array] @ self.ITEM_factors.T
        return item_scores
