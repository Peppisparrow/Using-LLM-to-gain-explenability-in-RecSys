'''
@author: Luca Pagano, Giuseppe Vitello
'''

import numpy as np

from RecSysFramework.Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from RecSysFramework.Recommenders.MatrixFactorization.ImplicitALSEmbeddingsInitialization import ImplicitALSRecommender
from Prototype.Decoder.ItemFactorLearner_implicit import ImplicitItemFactorLearner

class BlendedALSModelsUserRecommender(BaseMatrixFactorizationRecommender):
    """
    A hybrid recommender that combines user factors learned from two different recommenders.
    The final user factors are a weighted combination of the two sets of user factors.
    """
    RECOMMENDER_NAME = "BlendedALSModelsUserRecommender"

    def __init__(self, URM_train, verbose=True):
        super(BlendedALSModelsUserRecommender, self).__init__(URM_train, verbose=verbose)
        self._fitted_flag = False
        self._blending_factor = None
        self._initial_user_factors = None
        self._learned_user_factors = None
        self._model_params_cache = {}  # Cache for fixed_model parameters
        self._model = ImplicitItemFactorLearner(self.URM_train)

    def _normalize_factors(self, factors: np.ndarray) -> np.ndarray:
        """
        Divide factors by their L2 norm along the rows.
        """
        norms = np.linalg.norm(factors, axis=1, keepdims=True)
        return factors / norms

    def _get_blended_user_factors(self) -> np.ndarray:
        """
        Calculates and returns the blended user factors.
        """
        if not self._fitted_flag:
            raise ValueError("Model must be fitted before blending factors.")
        if self._blending_factor is None:
            raise ValueError("Blending factor must be set.")

        initial_normalized = self._normalize_factors(self._initial_user_factors)
        learned_normalized = self._normalize_factors(self._learned_user_factors)

        return (self._blending_factor * learned_normalized +
                (1 - self._blending_factor) * initial_normalized)

    def set_blending_factor(self, blending_factor: float, **fit_params):
        """
        Sets the blending factor and re-fits the final model.
        
        If fit_params are provided, they override the cached parameters.
        """
        if not self._fitted_flag:
            raise ValueError("Model must be fitted before setting the blending factor.")
        if not (0.0 <= blending_factor <= 1.0):
            raise ValueError("Blending factor must be between 0.0 and 1.0.")
        
        self._blending_factor = blending_factor
        blended_user_factors = self._get_blended_user_factors()
        
        # Use provided fit_params or fall back to cached parameters
        params_to_use = self._model_params_cache.copy()
        params_to_use.update(fit_params)

        self._model.fit(user_factors=blended_user_factors, **params_to_use)

        self.USER_factors = blended_user_factors
        self.ITEM_factors = self._model.ITEM_factors

        if self.verbose:
            self._print(f"Blending factor set to {self._blending_factor}. Final model re-fitted.")

    def fit(self, user_factors: np.ndarray,
            blending_factor: float = 0.5,
            init_model_params: dict = {},
            fixed_model_params: dict = {},
            use_gpu = True
            ):
        """
        Fits the two underlying models and then blends the user factors.
        """
        self._print("Starting fitting of BlendedALSModelsUserRecommender")

        if user_factors is None or not isinstance(user_factors, np.ndarray):
            raise ValueError("User factors must be a valid NumPy array.")
        
        ials_model = ImplicitALSRecommender(self.URM_train, verbose=self.verbose)

        # Fit models
        ials_model.fit(factors=user_factors.shape[1],
                       user_factors=user_factors,
                       use_gpu=use_gpu,
                       **init_model_params)

        self._initial_user_factors = user_factors 
        self._learned_user_factors = ials_model.USER_factors

        # Cache the fixed_model_params for later use
        self._model_params_cache = fixed_model_params.copy()

        self._fitted_flag = True
        
        self.set_blending_factor(blending_factor)
        
        self._print("Fitting completed.")

    def update_user_row(self, user_id, new_user_profile):
        """
        Updates the initial user factors for a single user and re-blends.
        """
        if not self._fitted_flag:
            raise ValueError("Model must be fitted before updating user factors.")
        if not (0 <= user_id < self.n_users):
            raise ValueError("Invalid user ID.")
        if new_user_profile.shape[0] != self._initial_user_factors.shape[1]:
            raise ValueError("New profile must have the same number of features as factors.")

        self._initial_user_factors[user_id, :] = new_user_profile
        
        self.set_blending_factor(self._blending_factor)