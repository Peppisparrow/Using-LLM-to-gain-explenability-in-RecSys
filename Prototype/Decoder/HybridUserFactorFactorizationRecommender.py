'''
@author: Luca Pagano, Giuseppe Vitello
'''

import numpy as np

from RecSysFramework.Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from RecSysFramework.Recommenders.MatrixFactorization.ImplicitALSEmbeddingsInitialization import ImplicitALSRecommender
from Prototype.Decoder.ItemFactorLearner_implicit import ImplicitItemFactorLearner

class HybridUserFactorFactorizationRecommender(BaseMatrixFactorizationRecommender):
    """
    A hybrid recommender that combines user factors learned from two different recommenders.
    The final user factors are a weighted combination of the two sets of user factors.
    """
    RECOMMENDER_NAME = "HybridUserFactorFactorizationRecommender"
    
    def __init__(self, URM_train):
        super(HybridUserFactorFactorizationRecommender, self).__init__(URM_train)
        self._original_USER_factors = None
        self._learned_USER_factors = None
        self._ITEM_factors_init = None
        self._ITEM_factors_fixed = None

        self._fitted_flag = False
        self._blending_factor = None

        self._init_model = ImplicitALSRecommender(URM_train)
        self._fixed_model = ImplicitItemFactorLearner(URM_train)

    def _normalize_factor(self, factors: np.ndarray):
        """
        Divide factor by its norm

        @param factors: np.ndarray
            The factors to normalize.
        """
        return factors / np.linalg.norm(factors, axis=1, keepdims=True)

    def set_blending_factor(self, blending_factor):
        """
        Set the blending factor for combining user factors.

        :param blending_factor: Weight for the first set of user factors. (0 <= blending_factor <= 1)
        if blending_factor = 1, only the learnt user factors are used.
        if blending_factor = 0, only the original user factors are used.
        0 < blending_factor < 1, a weighted combination of both user factors is used
        :type blending_factor: float
        :raises ValueError: If blending_factor is not between 0 and 1.
        """
        if not self._fitted_flag:
            raise ValueError("Model must be fitted before setting the blending factor.")
        
        if not (0 <= blending_factor <= 1):
            raise ValueError("Blending factor must be between 0 and 1.")
        
        self._blending_factor = blending_factor
        self.USER_factors = (self._blending_factor * self._learned_USER_factors +
                             (1 - self._blending_factor) * self._original_USER_factors)
        
        self.ITEM_factors = (self._blending_factor * self._ITEM_factors_init +
                             (1 - self._blending_factor) * self._ITEM_factors_fixed)
        
        if self.verbose:
            self._print(f"Blending factor set to {self._blending_factor}.")


    def fit(self,
            factors=100,
            regularization=0.01,
            use_native=True, use_cg=True, use_gpu=False,
            iterations=15,
            calculate_training_loss=False, num_threads=0,
            user_factors=None,
            blending_factor=0.5,
            **confidence_args):
        """
        Fit the hybrid recommender by training both underlying models and combining their user factors.
        """
        
        if user_factors is None:
            raise ValueError("User factors must be provided for fitting the hybrid model.")
        
        self._init_model.fit(factors=factors,
                             regularization=regularization,
                             use_native=use_native, use_cg=use_cg, use_gpu=use_gpu,
                             iterations=iterations,
                             calculate_training_loss=calculate_training_loss,
                             num_threads=num_threads,
                             user_factors=user_factors,
                             **confidence_args)
        
        self._fixed_model.fit(user_factors=user_factors,
                              alpha=confidence_args.get('alpha', 40),
                              reg=regularization
                              )
        
        if self.verbose:
            self._print(f"Combining user factors from both models...")
        
        self._original_USER_factors = self._normalize_factor(user_factors)
        self._learned_USER_factors = self._normalize_factor(self._init_model.USER_factors)
        
        self._ITEM_factors_init = self._normalize_factor(self._init_model.ITEM_factors)
        self._ITEM_factors_fixed = self._normalize_factor(self._fixed_model.ITEM_factors)
        
        self._fitted_flag = True
        self.set_blending_factor(blending_factor)
