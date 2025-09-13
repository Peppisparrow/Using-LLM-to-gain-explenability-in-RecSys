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
        
        if item_factors is not None:
            item_factors = np.ascontiguousarray(item_factors, dtype=np.float32)
            self.model.item_factors = item_factors
        if user_factors is not None:
            user_factors = np.ascontiguousarray(user_factors, dtype=np.float32)
            self.model.user_factors = user_factors
        
        self.model = self.model.to_gpu() if use_gpu else self.model
        # Check if it is on gpu
        if use_gpu and not isinstance(self.model, implicit.gpu.als.AlternatingLeastSquares):
            raise ValueError("ImplicitALSRecommender: The model is not on GPU even though use_gpu is set to True. "
                             "Please check your implicit library installation.")

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


# ------------------------------------ Hybrid Initialization -------------------------------------------------

class ImplicitLinearCombinationALSRecommender(ImplicitALSRecommender):
    """
    ImplicitLinearCombinationALSRecommender
    
    This recommender learns new user and item factors using the implicit ALS algorithm,
    and then combines them linearly with pre-existing initial factors.
    
    The final factors are a weighted average:
    F_final = weight * F_initial + (1 - weight) * F_learned
    
    The model caches the initial and learned factors after the first fit, allowing
    the blending weight to be changed at runtime without retraining.
    """
    RECOMMENDER_NAME = "ImplicitLinearCombinationALSRecommender"

    def __init__(self, URM_train, verbose=True):
        super().__init__(URM_train, verbose=verbose)
        self._initial_user_factors = None
        self._initial_item_factors = None
        self._learned_user_factors = None
        self._learned_item_factors = None
        self.blending_factor = 0.5

    def fit(self,
            blending_factor=0.5,
            factors=100,
            regularization=0.01,
            use_native=True, use_cg=True, use_gpu=False,
            iterations=15,
            calculate_training_loss=False, num_threads=0,
            item_factors=None, user_factors=None,
            **confidence_args):
        """
        Fits the model by training new factors and caching them. The final factors are then
        computed via a linear combination with any provided initial factors.

        :param blending_factor: Weighting for the initial factors (0.0 to 1.0).
                      1.0 gives full weight to initial factors.
                      0.0 gives full weight to learned factors.
        :param factors: The number of latent factors to compute.
        :param regularization: The regularization factor to use.
        :param use_native: Whether to use the native C++ extension.
        :param use_cg: Whether to use Conjugate Gradient solver.
        :param use_gpu: Whether to use GPU for computation.
        :param iterations: The number of ALS iterations to run.
        :param calculate_training_loss: Whether to calculate training loss at each iteration.
        :param num_threads: The number of threads to use for computation.
        :param item_factors: (Optional) Initial item factors to cache and combine with.
        :param user_factors: (Optional) Initial user factors to cache and combine with.
        :param confidence_args: Arguments for the confidence scaling function.
        """
        
        if blending_factor < 0.0 or blending_factor > 1.0:
            raise ValueError("Blending factor must be between 0.0 and 1.0")

        if self.verbose:
            self._print(f"Fitting ImplicitLinearCombinationALSRecommender with blending_factor={blending_factor}")

        if self.verbose:
            self._print("Caching initial factors.")
            
        self._initial_user_factors = user_factors.copy() if user_factors is not None else None
        self._initial_item_factors = item_factors.copy() if item_factors is not None else None

        self._print("Starting training to learn new representations...")
        super().fit(factors=factors,
                    regularization=regularization,
                    use_native=use_native, use_cg=use_cg, use_gpu=use_gpu,
                    iterations=iterations,
                    calculate_training_loss=calculate_training_loss,
                    num_threads=num_threads,
                    item_factors=item_factors,
                    user_factors=user_factors,
                    **confidence_args)
        
        if self.verbose:
            self._print("Training complete. Caching learned representations.")
        # After super().fit(), self.USER_factors and self.ITEM_factors hold the learned representations.
        self._learned_user_factors = self.USER_factors.copy()
        self._learned_item_factors = self.ITEM_factors.copy()
        
        # Set the initial blend of factors
        self.set_blending_factor_weight(blending_factor)

        if self.verbose:
            self._print("Fit completed.")

    def set_blending_factor_weight(self, blending_factor=0.5):
        """
        Sets the blending weight between initial and learned factors and re-computes
        the final user/item factors. Requires fit() to have been called first.

        :param blending_factor: Weight for the initial factors (0.0 to 1.0).
                      1.0 gives full weight to initial factors.
                      0.0 gives full weight to learned factors.
        """
        if self._learned_user_factors is None or self._learned_item_factors is None:
            raise RuntimeError("You must call fit() before setting the factors weight.")

        if not (0.0 <= blending_factor <= 1.0):
            raise ValueError(f"blending_factor must be between 0.0 and 1.0, but was {blending_factor}.")

        self.blending_factor = blending_factor
        if self.verbose:
            self._print(f"Setting new blending_factor={self.blending_factor} and re-blending factors.")

        # Linearly combine with initial factors if they were provided
        if self._initial_user_factors is not None:
            if self._initial_user_factors.shape != self._learned_user_factors.shape:
                raise ValueError(f"Shape of initial_user_factors {self._initial_user_factors.shape} "
                                 f"does not match learned factors shape {self._learned_user_factors.shape}.")
            
            self.USER_factors = self.blending_factor * self._initial_user_factors + (1 - self.blending_factor) * self._learned_user_factors
        else:
            self._print("No initial user factors provided; using learned user factors only.")
            self.USER_factors = self._learned_user_factors

        if self._initial_item_factors is not None:
            if self._initial_item_factors.shape != self._learned_item_factors.shape:
                 raise ValueError(f"Shape of initial_item_factors {self._initial_item_factors.shape} "
                                  f"does not match learned factors shape {self._learned_item_factors.shape}.")
            
            self.ITEM_factors = self.blending_factor * self._initial_item_factors + (1 - self.blending_factor) * self._learned_item_factors
        else:
            self._print("No initial item factors provided; using learned item factors only.")
            self.ITEM_factors = self._learned_item_factors
            
# --------------------------------------- Factor Constraint Initialization ----------------------------------------------------

class ImplicitALSRecommenderWithConstraints(ImplicitALSRecommender):
    """ImplicitALSRecommender recommender with factor deviation constraint"""

    def __init__(self, URM_train, verbose=True):
        super().__init__(URM_train, verbose=verbose)
        
        self._initial_user_factors = None
        self._initial_item_factors = None

    RECOMMENDER_NAME = "ImplicitALSRecommenderWithConstraints"
    
    
    def _check_constraint_violation(self, delta):
        """
        Check if current factors violate the delta constraint.
        
        Parameters:
        - delta: Maximum allowed deviation from initial factors
        
        Returns:
        - bool: True if constraint is violated, False otherwise
        """
        if delta is None or delta <= 0:
            return False
            
        violation = False
        
        if self._initial_user_factors is not None:
            user_diff = np.abs(self.USER_factors - self._initial_user_factors)
            if np.any(user_diff > delta):
                violation = True
                if self.verbose:
                    max_deviation = np.max(user_diff)
                    self._print(f"User factors constraint violated: max deviation = {max_deviation:.4f} > delta = {delta}")
                    
        if self._initial_item_factors is not None:
            item_diff = np.abs(self.ITEM_factors - self._initial_item_factors)
            if np.any(item_diff > delta):
                violation = True
                if self.verbose:
                    max_deviation = np.max(item_diff)
                    self._print(f"Item factors constraint violated: max deviation = {max_deviation:.4f} > delta = {delta}")
                    
        return violation

    def _apply_factor_constraints(self, delta):
        """
        Apply constraints to keep learned factors within delta distance of initial factors.
        
        Parameters:
        - delta: Maximum allowed deviation from initial factors
        """
        if delta is None or delta <= 0:
            return
            
        if self._initial_user_factors is not None:
            # Compute the difference between current and initial factors
            user_diff = self.USER_factors - self._initial_user_factors
            
            # Clip the difference to be within [-delta, delta]
            user_diff_clipped = np.clip(user_diff, -delta, delta)
            
            # Update factors: initial + clipped_difference
            self.USER_factors = self._initial_user_factors + user_diff_clipped
            
        if self._initial_item_factors is not None:
            # Compute the difference between current and initial factors
            item_diff = self.ITEM_factors - self._initial_item_factors
            
            # Clip the difference to be within [-delta, delta]
            item_diff_clipped = np.clip(item_diff, -delta, delta)
            
            # Update factors: initial + clipped_difference
            self.ITEM_factors = self._initial_item_factors + item_diff_clipped

    def fit(self,
            factors=100,
            regularization=0.01,
            use_native=True, use_cg=True, use_gpu=False,
            iterations=15,
            calculate_training_loss=False, num_threads=0,
            item_factors=None, user_factors=None,
            delta=None,
            constraint_mode='early_stop',  # or 'clip'
            **confidence_args
            ):
        """
        Fit the ImplicitALS model with optional factor deviation constraints.
        
        Parameters:
        - delta: Maximum allowed deviation from initial factors. If None, no constraint is applied.
                If delta > 0, learned factors will be constrained to be within delta distance
                of their initial values.
        - Other parameters: Same as original ImplicitALS parameters
        """

        if (delta is None or delta <= 0) and item_factors is None and user_factors is None:
            self._print("WARNING: No initial factors or delta constraint provided; running standard ImplicitALS fit.")
            super().fit(factors=factors,
                        regularization=regularization,
                        use_native=use_native, use_cg=True, use_gpu=use_gpu,
                        iterations=iterations,
                        calculate_training_loss=calculate_training_loss,
                        num_threads=num_threads,
                        item_factors=item_factors,
                        user_factors=user_factors,
                        **confidence_args)

        print("Implicit cuda support: ", implicit.gpu.HAS_CUDA)

        if use_gpu and not implicit.gpu.HAS_CUDA:
            raise ValueError("ImplicitALSRecommender: GPU support is requested but implicit.gpu.HAS_CUDA is False. "
                           "Please ensure that you have a compatible GPU and the necessary libraries installed.")

        # Set initial factors if provided
        if item_factors is not None:
            item_factors = np.ascontiguousarray(item_factors, dtype=np.float32)
            self.model.item_factors = item_factors
            self._initial_item_factors = item_factors.copy()
                
        if user_factors is not None:
            user_factors = np.ascontiguousarray(user_factors, dtype=np.float32)
            self.model.user_factors = user_factors
            self._initial_user_factors = user_factors.copy()

        self.model = self.model.to_gpu() if use_gpu else self.model

        C = self._confidence_scaling(**confidence_args)

        # Fit with manual iteration control to apply constraints
        if self.verbose:
            self._print(f"Fitting with delta={delta} with initialization constraints.")
            self._print(f"\nUser Factors: {self._initial_user_factors is not None}")
            self._print(f"\nItem Factors: {self._initial_item_factors is not None}")

        if constraint_mode == 'early_stop':
            self._fit_with_early_stop_constraint(delta, factors, regularization, use_native, use_cg, use_gpu,
                                                iterations, calculate_training_loss, num_threads, C)
        elif constraint_mode == 'clip':
            self._fit_with_clipping_constraint(delta, factors, regularization, use_native, use_cg, use_gpu,
                                              iterations, calculate_training_loss, num_threads, C)
        else:
            raise ValueError("Invalid constraint_mode. Use 'early_stop' or 'clip'.")
                
    def _fit_with_early_stop_constraint(self, delta, factors, regularization, use_native, use_cg, use_gpu,
                                       iterations, calculate_training_loss, num_threads, C):
        """
        Fit with early stopping constraint mode - stops when constraint is first violated.
        
        Parameters:
        - delta: Maximum allowed deviation from initial factors
        - Other parameters: ALS training parameters
        - C: Confidence matrix
        """
        if self.verbose:
            self._print(f"Using early stopping constraint mode with delta={delta}")
        
        for iteration in range(iterations):
            if self.verbose:
                self._print(f"Iteration {iteration + 1}/{iterations}")
            
            # Perform one iteration of ALS
            temp_model = AlternatingLeastSquares(factors=factors, regularization=regularization,
                                               use_native=use_native, use_cg=use_cg, use_gpu=use_gpu,
                                               iterations=1,
                                               calculate_training_loss=calculate_training_loss,
                                               num_threads=num_threads,
                                               random_state=42
                                               )
            
            # Transfer current factors to new model
            if use_gpu:
                temp_model = temp_model.to_gpu()
                if hasattr(self.model, 'user_factors'):
                    temp_model.user_factors = self.model.user_factors
                if hasattr(self.model, 'item_factors'):
                    temp_model.item_factors = self.model.item_factors
            else:
                if hasattr(self.model, 'user_factors'):
                    temp_model.user_factors = self.model.user_factors.copy()
                if hasattr(self.model, 'item_factors'):
                    temp_model.item_factors = self.model.item_factors.copy()
            
            # Fit one iteration
            temp_model.fit(C, show_progress=False)
            
            # Extract factors
            if use_gpu:
                self.USER_factors = temp_model.user_factors.to_numpy()
                self.ITEM_factors = temp_model.item_factors.to_numpy()
            else:
                self.USER_factors = temp_model.user_factors
                self.ITEM_factors = temp_model.item_factors
            
            # Check if constraint is violated
            if self._check_constraint_violation(delta):
                if self.verbose:
                    self._print(f"Constraint violated at iteration {iteration + 1}. Stopping and clipping.")
                # Apply clipping and stop
                self._apply_factor_constraints(delta)
                break
            
            # Update model with current factors for next iteration
            self.model = temp_model

    def _fit_with_clipping_constraint(self, delta, factors, regularization, use_native, use_cg, use_gpu,
                                        iterations, calculate_training_loss, num_threads, C):
        """
        Fit with clipping constraint mode - applies clipping after each iteration.
        
        Parameters:
        - delta: Maximum allowed deviation from initial factors
        - Other parameters: ALS training parameters
        - C: Confidence matrix
        """
        if self.verbose:
            self._print(f"Using clipping constraint mode with delta={delta}")
        
        for iteration in range(iterations):
            if self.verbose:
                self._print(f"Iteration {iteration + 1}/{iterations}")
            
            # Perform one iteration of ALS
            temp_model = AlternatingLeastSquares(factors=factors, regularization=regularization,
                                            use_native=use_native, use_cg=use_cg, use_gpu=use_gpu,
                                            iterations=1,
                                            calculate_training_loss=calculate_training_loss,
                                            num_threads=num_threads,
                                            random_state=42
                                            )
            
            # Transfer current factors to new model
            if use_gpu:
                temp_model = temp_model.to_gpu()
                if hasattr(self.model, 'user_factors'):
                    temp_model.user_factors = self.model.user_factors
                if hasattr(self.model, 'item_factors'):
                    temp_model.item_factors = self.model.item_factors
            else:
                if hasattr(self.model, 'user_factors'):
                    temp_model.user_factors = self.model.user_factors.copy()
                if hasattr(self.model, 'item_factors'):
                    temp_model.item_factors = self.model.item_factors.copy()
            
            # Fit one iteration
            temp_model.fit(C, show_progress=False)
            
            # Extract factors
            if use_gpu:
                self.USER_factors = temp_model.user_factors.to_numpy()
                self.ITEM_factors = temp_model.item_factors.to_numpy()
            else:
                self.USER_factors = temp_model.user_factors
                self.ITEM_factors = temp_model.item_factors
            
            # Apply clipping to enforce constraints
            self._apply_factor_constraints(delta)
            
            # Update model with current factors for next iteration
        self.model = temp_model