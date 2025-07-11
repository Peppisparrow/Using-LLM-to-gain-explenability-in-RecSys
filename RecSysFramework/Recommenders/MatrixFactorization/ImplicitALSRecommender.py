'''
@author Nicola Cesare
It is a wrapper of the implicit recommender (https://github.com/benfred/implicit).
@author: Luca Pagano, Giuseppe Vitello: Added some modifications to make it work.
'''


import implicit
from RecSysFramework.Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender

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

    def fit(self,
            factors=100,
            regularization=0.01,
            use_native=True, use_cg=True, use_gpu=False,
            iterations=15,
            calculate_training_loss=False, num_threads=0,
            alpha=1.0,
            ):
        
        print("Implicit cuda support: ", implicit.gpu.HAS_CUDA)
        
        if use_gpu and not implicit.gpu.HAS_CUDA:
            raise ValueError("ImplicitALSRecommender: GPU support is requested but implicit.gpu.HAS_CUDA is False. "
                             "Please ensure that you have a compatible GPU and the necessary libraries installed.")
        
        self.model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=regularization,
                                                        use_native=use_native, use_cg=use_cg, use_gpu=use_gpu,
                                                        iterations=iterations,
                                                        calculate_training_loss=calculate_training_loss,
                                                        num_threads=num_threads,
                                                        alpha=alpha,
                                                        random_state=42
                                                        )

        self.model.fit(self.URM_train, show_progress=self.verbose)

        if use_gpu:
            # Convert the user and item factors to numpy arrays if using GPU
            self.USER_factors = self.model.user_factors.to_numpy()
            self.ITEM_factors = self.model.item_factors.to_numpy()
        else:
            self.USER_factors = self.model.user_factors
            self.ITEM_factors = self.model.item_factors