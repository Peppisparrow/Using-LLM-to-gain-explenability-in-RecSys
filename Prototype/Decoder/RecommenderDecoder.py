from RecSysFramework.Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from RecSysFramework.Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from RecSysFramework.Recommenders.Recommender_utils import check_matrix
import numpy as np
from tqdm import tqdm


class RecommenderDecoder(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):

    RECOMMENDER_NAME = "RecommenderDecoder"

    AVAILABLE_CONFIDENCE_SCALING = ["linear", "log"]


    def fit(self,
            user_factors,
            epochs = 300,
            num_factors = 20,
            confidence_scaling = "linear",
            alpha = 1.0,
            epsilon = 1.0,
            reg = 1e-3,
            init_mean=0.0,
            init_std=0.1,
            **earlystopping_kwargs):
        """

        :param epochs:
        :param num_factors:
        :param confidence_scaling: supported scaling modes for the observed values: 'linear' or 'log'
        :param alpha: Confidence weight, confidence c = 1 + alpha*r where r is the observed "rating".
        :param reg: Regularization constant.
        :param epsilon: epsilon used in log scaling only
        :param init_mean: mean used to initialize the latent factors
        :param init_std: standard deviation used to initialize the latent factors
        :return:
        """

        if confidence_scaling not in self.AVAILABLE_CONFIDENCE_SCALING:
           raise ValueError("Value for 'confidence_scaling' not recognized. Acceptable values are {}, provided was '{}'".format(self.AVAILABLE_CONFIDENCE_SCALING, confidence_scaling))


        self.num_factors = num_factors
        self.alpha = alpha
        self.epsilon = epsilon
        self.reg = reg

        self.USER_factors = user_factors
        self.ITEM_factors = self._init_factors(self.n_items)


        self._build_confidence_matrix(confidence_scaling)


        warm_user_mask = np.ediff1d(self.URM_train.indptr) > 0
        warm_item_mask = np.ediff1d(self.URM_train.tocsc().indptr) > 0

        self.warm_users = np.arange(0, self.n_users, dtype=np.int32)[warm_user_mask]
        self.warm_items = np.arange(0, self.n_items, dtype=np.int32)[warm_item_mask]

        self.regularization_diagonal = np.diag(self.reg * np.ones(self.num_factors))

        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)


        self.USER_factors = self.USER_factors_best
        self.ITEM_factors = self.ITEM_factors_best




    def _build_confidence_matrix(self, confidence_scaling):

        if confidence_scaling == 'linear':
            self.C = self._linear_scaling_confidence()
        else:
            self.C = self._log_scaling_confidence()

        self.C_csc= check_matrix(self.C.copy(), format="csc", dtype = np.float32)


    def _linear_scaling_confidence(self):

        C = check_matrix(self.URM_train, format="csr", dtype = np.float32)
        C.data = 1.0 + self.alpha*C.data

        return C

    def _log_scaling_confidence(self):

        C = check_matrix(self.URM_train, format="csr", dtype = np.float32)
        C.data = 1.0 + self.alpha * np.log(1.0 + C.data / self.epsilon)

        return C


    def _prepare_model_for_validation(self):
        pass


    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()


    def _run_epoch(self, num_epoch):
        print("Run epoch {}".format(num_epoch))
        UU = self.USER_factors.T.dot(self.USER_factors)
        print("UU shape: {}".format(UU.shape))
        for _, item_id in tqdm(enumerate(self.warm_items), total=len(self.warm_items), desc="Updating item factors"):

            start_pos = self.C_csc.indptr[item_id]
            end_pos = self.C_csc.indptr[item_id + 1]

            item_profile = self.C_csc.indices[start_pos:end_pos]
            item_confidence = self.C_csc.data[start_pos:end_pos]

            self.ITEM_factors[item_id, :] = self._update_row(item_profile, item_confidence, self.USER_factors, UU)



    def _update_row(self, interaction_profile, interaction_confidence, Y, YtY):
        """
        Update latent factors for a single user or item.

        Y = |n_interactions|x|n_factors|
        YtY =   |n_factors|x|n_factors|
        """

        # Latent factors ony of item/users for which an interaction exists in the interaction profile
        Y_interactions = Y[interaction_profile, :]

        # Following the notation of the original paper we report the update rule for the Item factors (User factors are identical):
        # Y are the item factors |n_items|x|n_factors|
        # Cu is a diagonal matrix |n_interactions|x|n_interactions| with the user confidence for the observed items
        # p(u) is a boolean vectors indexing only observed items. Here it will disappear as we already extract only the observed latent factors
        #       however, it will have an impact in the dimensions of the matrix, since it transforms Cu from a diagonal matrix to a row vector of 1 row and |n_interactions| columns
        # (Yt*Cu*Y + reg*I)^-1 * Yt*Cu*profile
        # which can be decomposed as
        # (YtY + Yt*(Cu-I)*Y + reg*I)^-1 * Yt*Cu*p(u)

        # A = (|n_interactions|x|n_factors|) dot (|n_interactions|x|n_interactions| ) dot (|n_interactions|x|n_factors| )
        #   = |n_factors|x|n_factors|
        # A_slow = Y_interactions.T.dot(np.diag(interaction_confidence - 1)).dot(Y_interactions)

        # if v = diag(|n_interactions|) and k = |n_interactions|x|n_factors|
        # computing np.diag(v).dot(k) will be SLOW
        # we use an equivalent formulation (v * k.T).T which is much faster
        A = Y_interactions.T.dot(((interaction_confidence - 1) * Y_interactions.T).T)

        B = YtY + A + self.regularization_diagonal

        return np.dot(np.linalg.inv(B), Y_interactions.T.dot(interaction_confidence))


    def _init_factors(self, num_factors, assign_values=True):

        if assign_values:
            return self.num_factors**-0.5*np.random.random_sample((num_factors, self.num_factors))

        else:
            return np.empty((num_factors, self.num_factors))



