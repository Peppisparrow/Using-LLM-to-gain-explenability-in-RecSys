"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

#cython: boundscheck=False
#cython: wraparound=True
#cython: initializedcheck=False
#cython: language_level=3
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

"""
Determine the operative system. The interface of numpy returns a different type for argsort under windows and linux

http://docs.cython.org/en/latest/src/userguide/language_basics.html#conditional-compilation
"""
IF UNAME_SYSNAME == "linux":
    DEF LONG_t = "long"
ELIF  UNAME_SYSNAME == "Windows":
    DEF LONG_t = "long long"
ELSE:
    DEF LONG_t = "long long"



import time, sys
import cython
import numpy as np
cimport numpy as np

from cpython.array cimport array, clone

from libc.math cimport sqrt




import scipy.sparse as sps
from RecSysFramework.Recommenders.Recommender_utils import check_matrix
from RecSysFramework.Utils.seconds_to_biggest_unit import seconds_to_biggest_unit

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
cdef class Compute_Similarity_Cython:

    cdef int topK, use_dense
    cdef long n_columns, n_rows

    cdef double[:] this_item_weights
    cdef int[:] this_item_weights_mask, this_item_weights_id
    cdef int this_item_weights_counter

    cdef int[:] user_to_item_row_ptr, user_to_item_cols
    cdef int[:] item_to_user_rows, item_to_user_col_ptr
    cdef double[:] user_to_item_data, item_to_user_data
    cdef double[:] sum_of_squared, sum_of_squared_to_1_minus_alpha, sum_of_squared_to_alpha
    cdef int shrink, normalize, adjusted_cosine, pearson_correlation, tanimoto_coefficient, asymmetric_cosine, dice_coefficient, tversky_coefficient
    cdef float asymmetric_alpha, tversky_alpha, tversky_beta

    cdef int use_row_weights
    cdef double[:] row_weights

    cdef double[:,:] W_dense

    def __init__(self, dataMatrix, topK = 100, shrink=0, normalize = True,
                 asymmetric_alpha = 0.5, tversky_alpha = 1.0, tversky_beta = 1.0,
                 similarity = "cosine", row_weights = None, use_dense = False):
        """
        Computes the cosine similarity on the columns of dataMatrix
        If it is computed on URM=|users|x|items|, pass the URM as is.
        If it is computed on ICM=|items|x|features|, pass the ICM transposed.
        :param dataMatrix:
        :param topK:
        :param use_dense:           If True returns dense numpy similarity
        :param shrink:
        :param normalize:           If True divide the dot product by the product of the norms
        :param row_weights:         Multiply the values in each row by a specified value. Array
        :param asymmetric_alpha     Coefficient alpha for the asymmetric cosine
        :param similarity:  "cosine"        computes Cosine similarity
                            "adjusted"      computes Adjusted Cosine, removing the average of the users
                            "asymmetric"    computes Asymmetric Cosine
                            "pearson"       computes Pearson Correlation, removing the average of the items
                            "jaccard"       computes Jaccard similarity for binary interactions using Tanimoto
                            "dice"          computes Dice similarity for binary interactions
                            "tversky"       computes Tversky similarity for binary interactions
                            "tanimoto"      computes Tanimoto coefficient for binary interactions

        """
        """
        Asymmetric Cosine as described in: 
        Aiolli, F. (2013, October). Efficient top-n recommendation for very large scale binary rated datasets. In Proceedings of the 7th ACM conference on Recommender systems (pp. 273-280). ACM.
        
        """

        super(Compute_Similarity_Cython, self).__init__()

        self.n_columns = dataMatrix.shape[1]
        self.n_rows = dataMatrix.shape[0]
        self.shrink = shrink
        self.normalize = normalize
        self.asymmetric_alpha = asymmetric_alpha
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

        self.adjusted_cosine = False
        self.asymmetric_cosine = False
        self.pearson_correlation = False
        self.tanimoto_coefficient = False
        self.dice_coefficient = False
        self.tversky_coefficient = False

        if similarity == "adjusted":
            self.adjusted_cosine = True
        elif similarity == "asymmetric":
            self.asymmetric_cosine = True
        elif similarity == "pearson":
            self.pearson_correlation = True
        elif similarity == "jaccard" or similarity == "tanimoto":
            self.tanimoto_coefficient = True
            # Tanimoto has a specific kind of normalization
            self.normalize = False

        elif similarity == "dice":
            self.dice_coefficient = True
            self.normalize = False

        elif similarity == "tversky":
            self.tversky_coefficient = True
            self.normalize = False

        elif similarity == "cosine":
            pass
        else:
            raise ValueError("Cosine_Similarity: value for parameter 'mode' not recognized."
                             " Allowed values are: 'cosine', 'pearson', 'adjusted', 'asymmetric', 'jaccard', 'tanimoto',"
                             "dice, tversky."
                             " Passed value was '{}'".format(similarity))

        assert topK >=0 and topK <= self.n_columns, "TopK must be between 0 and the number of columns"
        self.topK = min(topK, self.n_columns)
        self.use_dense = use_dense
        self.this_item_weights = np.zeros(self.n_columns, dtype=np.float64)
        self.this_item_weights_id = np.zeros(self.n_columns, dtype=np.int32)
        self.this_item_weights_mask = np.zeros(self.n_columns, dtype=np.int32)
        self.this_item_weights_counter = 0

        # Copy data to avoid altering the original object
        dataMatrix = dataMatrix.copy()


        if self.adjusted_cosine:
            dataMatrix = self.applyAdjustedCosine(dataMatrix)
        elif self.pearson_correlation:
            dataMatrix = self.applyPearsonCorrelation(dataMatrix)
        elif self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient:
            dataMatrix = self.useOnlyBooleanInteractions(dataMatrix)


        # Compute sum of squared values to be used in normalization
        self.sum_of_squared = np.array(dataMatrix.power(2).sum(axis=0), dtype=np.float64).ravel()

        # Tanimoto does not require the square root to be applied
        if not (self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient):
            self.sum_of_squared = np.sqrt(self.sum_of_squared)

        if self.asymmetric_cosine:
            # The power of 1-alpha may be negative so add small value to ensure values are non-zeros
            sum_of_squared_np = np.array(self.sum_of_squared) + 1e-6
            self.sum_of_squared_to_alpha = np.power(sum_of_squared_np, 2 * self.asymmetric_alpha)
            self.sum_of_squared_to_1_minus_alpha = np.power(sum_of_squared_np, 2 * (1 - self.asymmetric_alpha))

        # Apply weight after sum_of_squared has been computed but before the matrix is
        # split in its inner data structures
        self.use_row_weights = False

        if row_weights is not None:

            if dataMatrix.shape[0] != len(row_weights):
                raise ValueError("Cosine_Similarity: provided row_weights and dataMatrix have different number of rows."
                                 "Row_weights has {} rows, dataMatrix has {}.".format(len(row_weights), dataMatrix.shape[0]))


            self.use_row_weights = True
            self.row_weights = np.array(row_weights, dtype=np.float64)


        dataMatrix = check_matrix(dataMatrix, 'csr')

        self.user_to_item_row_ptr = dataMatrix.indptr
        self.user_to_item_cols = dataMatrix.indices
        self.user_to_item_data = np.array(dataMatrix.data, dtype=np.float64)

        dataMatrix = check_matrix(dataMatrix, 'csc')
        self.item_to_user_rows = dataMatrix.indices
        self.item_to_user_col_ptr = dataMatrix.indptr
        self.item_to_user_data = np.array(dataMatrix.data, dtype=np.float64)


        if self.use_dense:
            self.W_dense = np.zeros((self.n_columns,self.n_columns))



    cdef useOnlyBooleanInteractions(self, dataMatrix):
        """
        Set to 1 all data points
        :return:
        """

        cdef long index

        for index in range(len(dataMatrix.data)):
            dataMatrix.data[index] = 1

        return dataMatrix



    cdef applyPearsonCorrelation(self, dataMatrix):
        """
        Remove from every data point the average for the corresponding column
        :return:
        """

        cdef double[:] sumPerCol
        cdef int[:] interactionsPerCol
        cdef long colIndex, innerIndex, start_pos, end_pos
        cdef double colAverage

        dataMatrix = check_matrix(dataMatrix, 'csc')

        sumPerCol = np.array(dataMatrix.sum(axis=0), dtype=np.float64).ravel()
        interactionsPerCol = np.diff(dataMatrix.indptr)

        #Remove for every row the corresponding average
        for colIndex in range(self.n_columns):

            if interactionsPerCol[colIndex]>0:

                colAverage = sumPerCol[colIndex] / interactionsPerCol[colIndex]

                start_pos = dataMatrix.indptr[colIndex]
                end_pos = dataMatrix.indptr[colIndex+1]

                innerIndex = start_pos
                while innerIndex < end_pos:
                    dataMatrix.data[innerIndex] -= colAverage
                    innerIndex+=1

        return dataMatrix


    cdef applyAdjustedCosine(self, dataMatrix):
        """
        Remove from every data point the average for the corresponding row
        :return:
        """

        cdef double[:] sumPerRow
        cdef int[:] interactionsPerRow
        cdef long rowIndex, innerIndex, start_pos, end_pos
        cdef double rowAverage

        dataMatrix = check_matrix(dataMatrix, 'csr')

        sumPerRow = np.array(dataMatrix.sum(axis=1), dtype=np.float64).ravel()
        interactionsPerRow = np.diff(dataMatrix.indptr)

        #Remove for every row the corresponding average
        for rowIndex in range(self.n_rows):

            if interactionsPerRow[rowIndex]>0:

                rowAverage = sumPerRow[rowIndex] / interactionsPerRow[rowIndex]

                start_pos = dataMatrix.indptr[rowIndex]
                end_pos = dataMatrix.indptr[rowIndex+1]

                innerIndex = start_pos
                while innerIndex < end_pos:
                    dataMatrix.data[innerIndex] -= rowAverage
                    innerIndex+=1

        return dataMatrix


    cdef int[:] getUsersThatRatedItem(self, long item_id):
        return self.item_to_user_rows[self.item_to_user_col_ptr[item_id]:self.item_to_user_col_ptr[item_id+1]]

    cdef int[:] getItemsRatedByUser(self, long user_id):
        return self.user_to_item_cols[self.user_to_item_row_ptr[user_id]:self.user_to_item_row_ptr[user_id+1]]


    cdef computeItemSimilarities(self, long item_id_input):
        """
        For every item the cosine similarity against other items depends on whether they have users in common. The more
        common users the higher the similarity.
        
        The basic implementation is:
        - Select the first item
        - Loop through all other items
        -- Given the two items, get the users they have in common
        -- Update the similarity for all common users
        
        That is VERY slow due to the common user part, in which a long data structure is looped multiple times.
        
        A better way is to use the data structure in a different way skipping the search part, getting directly the
        information we need.
        
        The implementation here used is:
        - Select the first item
        - Initialize a zero valued array for the similarities
        - Get the users who rated the first item
        - Loop through the users
        -- Given a user, get the items he rated (second item)
        -- Update the similarity of the items he rated
        
        
        """

        # Create template used to initialize an array with zeros
        # Much faster than np.zeros(self.n_columns)
        #cdef array[double] template_zero = array('d')
        #cdef array[double] result = clone(template_zero, self.n_columns, zero=True)


        cdef long user_index, user_id, item_index, item_id, item_id_second

        cdef int[:] users_that_rated_item = self.getUsersThatRatedItem(item_id_input)
        cdef int[:] items_rated_by_user

        cdef double rating_item_input, rating_item_second, row_weight

        # Clean previous item
        for item_index in range(self.this_item_weights_counter):
            item_id = self.this_item_weights_id[item_index]
            self.this_item_weights_mask[item_id] = False
            self.this_item_weights[item_id] = 0.0

        self.this_item_weights_counter = 0


        # Get users that rated the items
        for user_index in range(len(users_that_rated_item)):

            user_id = users_that_rated_item[user_index]
            rating_item_input = self.item_to_user_data[self.item_to_user_col_ptr[item_id_input]+user_index]

            if self.use_row_weights:
                row_weight = self.row_weights[user_id]
            else:
                row_weight = 1.0

            # Get all items rated by that user
            items_rated_by_user = self.getItemsRatedByUser(user_id)

            for item_index in range(len(items_rated_by_user)):

                item_id_second = items_rated_by_user[item_index]

                # Do not compute the similarity on the diagonal
                if item_id_second != item_id_input:
                    # Increment similairty
                    rating_item_second = self.user_to_item_data[self.user_to_item_row_ptr[user_id]+item_index]

                    self.this_item_weights[item_id_second] += rating_item_input*rating_item_second*row_weight

                    # Update global data structure
                    if not self.this_item_weights_mask[item_id_second]:
                        self.this_item_weights_mask[item_id_second] = True
                        self.this_item_weights_id[self.this_item_weights_counter] = item_id_second
                        self.this_item_weights_counter += 1


    def compute_similarity(self, start_col=None, end_col=None):
        """
        Compute the similarity for the given dataset
        :param self:
        :param start_col: column to begin with
        :param end_col: column to stop before, end_col is excluded
        :return:
        """

        cdef int print_block_size = 500

        cdef int item_index, inner_item_index, item_id, local_topK
        cdef long long topK_item_index

        cdef long long[:] top_k_idx

        # Declare numpy data type to use vetor indexing and simplify the topK selection code
        cdef np.ndarray[LONG_t, ndim=1] relevant_items_partition
        cdef np.ndarray[np.float32_t, ndim=1] this_item_weights_np = np.zeros(self.n_columns, dtype=np.float32)

        cdef long processed_items = 0

        # Data structure to incrementally build sparse matrix
        # Preinitialize max possible length
        cdef unsigned long long max_cells = <long long> self.n_columns*self.topK
        cdef double[:] values = np.zeros((max_cells))
        cdef int[:] rows = np.zeros((max_cells,), dtype=np.int32)
        cdef int[:] cols = np.zeros((max_cells,), dtype=np.int32)
        cdef long sparse_data_pointer = 0

        cdef int start_col_local = 0, end_col_local = self.n_columns

        if start_col is not None and start_col>0 and start_col<self.n_columns:
            start_col_local = start_col

        if end_col is not None and end_col>start_col_local and end_col<self.n_columns:
            end_col_local = end_col

        start_time = time.time()
        last_print_time = start_time

        item_index = start_col_local

        # Compute all similarities for each item
        while item_index < end_col_local:

            processed_items += 1

            # Computed similarities go in self.this_item_weights
            self.computeItemSimilarities(item_index)

            # Apply normalization and shrinkage, ensure denominator != 0
            if self.normalize:
                for inner_item_index in range(self.n_columns):

                    if self.asymmetric_cosine:
                        self.this_item_weights[inner_item_index] /= self.sum_of_squared_to_alpha[item_index] * self.sum_of_squared_to_1_minus_alpha[inner_item_index]\
                                                             + self.shrink + 1e-6

                    else:
                        self.this_item_weights[inner_item_index] /= self.sum_of_squared[item_index] * self.sum_of_squared[inner_item_index]\
                                                             + self.shrink + 1e-6

            # Apply the specific denominator for Tanimoto
            elif self.tanimoto_coefficient:
                for inner_item_index in range(self.n_columns):
                    self.this_item_weights[inner_item_index] /= self.sum_of_squared[item_index] + self.sum_of_squared[inner_item_index] -\
                                                         self.this_item_weights[inner_item_index] + self.shrink + 1e-6

            elif self.dice_coefficient:
                for inner_item_index in range(self.n_columns):
                    self.this_item_weights[inner_item_index] /= self.sum_of_squared[item_index] + self.sum_of_squared[inner_item_index] +\
                                                         self.shrink + 1e-6

            elif self.tversky_coefficient:
                for inner_item_index in range(self.n_columns):
                    self.this_item_weights[inner_item_index] /= self.this_item_weights[inner_item_index] + \
                                                              (self.sum_of_squared[item_index]-self.this_item_weights[inner_item_index])*self.tversky_alpha + \
                                                              (self.sum_of_squared[inner_item_index]-self.this_item_weights[inner_item_index])*self.tversky_beta +\
                                                              self.shrink + 1e-6

            elif self.shrink != 0:
                for inner_item_index in range(self.n_columns):
                    self.this_item_weights[inner_item_index] /= self.shrink


            if self.use_dense:
                for inner_item_index in range(self.n_columns):
                    self.W_dense[inner_item_index,item_index] = self.this_item_weights[inner_item_index]

            else:
                # Sort indices and select topK
                # Using numpy implies some overhead, unfortunately the plain C qsort function is even slower
                for inner_item_index in range(self.n_columns):
                    this_item_weights_np[inner_item_index] = 0.0

                # Add weights in the same ordering as the self.this_item_weights_id data structure
                for inner_item_index in range(self.this_item_weights_counter):
                    item_id = self.this_item_weights_id[inner_item_index]
                    this_item_weights_np[inner_item_index] = - self.this_item_weights[item_id]

                # Sort indices and select topK, partition the data to extract the set of relevant items
                relevant_items_partition = np.argpartition(this_item_weights_np, self.topK - 1, axis=0)[0:self.topK]

                # Incrementally build sparse matrix, do not add zeros
                for inner_item_index in range(len(relevant_items_partition)):

                    topK_item_index = relevant_items_partition[inner_item_index]

                    item_id = self.this_item_weights_id[topK_item_index]

                    if self.this_item_weights[item_id] != 0.0:

                        values[sparse_data_pointer] = self.this_item_weights[item_id]
                        rows[sparse_data_pointer] = item_id
                        cols[sparse_data_pointer] = item_index

                        sparse_data_pointer += 1

            item_index += 1

            if processed_items % print_block_size==0 or processed_items==end_col_local:

                current_time = time.time()

                # Set block size to the number of items necessary in order to print every 300 seconds
                if current_time - start_time != 0:
                    items_per_sec = processed_items/(current_time - start_time)
                else:
                    items_per_sec = 1

                print_block_size = int(items_per_sec*300)

                if current_time - last_print_time > 300  or processed_items==end_col_local:
                    new_time_value, new_time_unit = seconds_to_biggest_unit(time.time() - start_time)

                    print("Similarity column {} ({:4.1f}%), {:.2f} column/sec. Elapsed time {:.2f} {}".format(
                        processed_items, processed_items*1.0/(end_col_local-start_col_local)*100, items_per_sec, new_time_value, new_time_unit))

                    last_print_time = current_time

                    sys.stdout.flush()
                    sys.stderr.flush()

        # End while on columns

        if self.use_dense:
            return np.array(self.W_dense)

        else:
            values = np.array(values[0:sparse_data_pointer])
            rows = np.array(rows[0:sparse_data_pointer])
            cols = np.array(cols[0:sparse_data_pointer])

            W_sparse = sps.csr_matrix((values, (rows, cols)),
                                    shape=(self.n_columns, self.n_columns),
                                    dtype=np.float32)

            return W_sparse
