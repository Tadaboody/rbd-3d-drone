#################################################################
#     Corset for Weighted centers of points                     #
#     Paper: http://people.csail.mit.edu/dannyf/outliers.pdf    #
#     Implemented by Yair Marom. yairmrm@gmail.com              #
#################################################################
import csv
import numpy as np
from set_of_lines import SetOfLines

class ExperimentsKMeansForLines:
    """
    A class that includes all the main experiments of the weighted centers corset API
    """

    #static variables
    current_iteration = 0
    parameters_config = None

    ######################################################################

    @staticmethod
    def EM_estimetor_k_means_for_lines(L, k, is_ground_truth = True, EM_iterations = np.nan, print_iterations=False):
        """
        This method gets a set L of lines, number of centers k>0, and returns the k centers that minimizes the sum
        of squared distances to L.
        Args:
            L (SetOfLines) : a set of lines
            k (int) : number of centers
            max_iterations (int) : number of maximum iterations allowed before convergence
            type: "novel" for our EM estimator, and "sota" for the state of the art EM estimator

        Returns:
            np.ndarray: a set of k centers that minimizes the sum of squared distances to L
        """

        size = L.get_size()
        dim = L.dim
        assert size > 0, "L is empty"
        assert k > 0, "k <= 0"

        parameter_config = ExperimentsKMeansForLines.parameters_config
        if is_ground_truth:
            EM_iterations = L.get_size() * 1
        size_of_sample = int(2 * k)
        if size <= size_of_sample:
            S = L
        else:
            S = L.get_sample_of_lines(size_of_sample)
        P_4_approx_min = S.get_4_approx_points(k)
        cost_min = L.get_sum_of_distances_to_centers(P_4_approx_min)
        for i in range(EM_iterations):
            if print_iterations:
                if i% 30 == 0 and i != 0:
                    print("EM estimator iteration number: ", i, "cost_min by now: ", cost_min)
            if size <= size_of_sample:
                S = L
            else:
                S = L.get_sample_of_lines(size_of_sample)
            P_4_approx_curent = S.get_4_approx_points(k)
            cost_current = L.get_sum_of_distances_to_centers(P_4_approx_curent)
            if cost_current < cost_min:
                P_4_approx_min = P_4_approx_curent
                cost_min = cost_current

        return P_4_approx_min

    ######################################################################

    @staticmethod
    def EM_estimetor_k_means_for_points(P, is_ground_truth=True, EM_iterations=np.nan):
        """
        The state of the art algorithm to finding a robust median without outliers.
        Args:
            P (SetOfPoints) : set of weighted points
            k (int) : number of weighted centers

        Returns:
            np.ndarray: the robust median of P
        """
        parameters_config = ExperimentsKMeansForLines.parameters_config
        centers_number = parameters_config.centers_number
        if is_ground_truth:
            EM_iterations = parameters_config.ground_truth_iterations[ExperimentsKMeansForLines.current_iteration]
        min_k_centers = np.nan
        min_cost = np.infty
        for i in range(EM_iterations):
            k_centers = P.get_sample_of_points(centers_number)
            current_cost = k_centers.get_sum_of_distances_to_set_of_points(P)
            if current_cost < min_cost:
                min_cost = current_cost
                min_k_centers = k_centers
        return min_k_centers

    ######################################################################

    @staticmethod
    def RANSAC_k_means_for_lines(L, k, sample_size):
        """
        TODO: complete
        :param L:
        :param k:
        :return:
        """
        RANSAC_EM_ITERATIONS = ExperimentsKMeansForLines.parameters_config.RANSAC_EM_ITERATIONS
        random_sample_min = L.get_sample_of_lines(sample_size)  # sample m lines from uniform distribution on L
        random_sample_min_centers = ExperimentsKMeansForLines.EM_estimetor_k_means_for_lines(random_sample_min, k, is_ground_truth=False ,EM_iterations=RANSAC_EM_ITERATIONS)  # send the sample to the EM estimator and get the k points that minimizes the sum of squared distances from the lines in the sample
        random_sample_min_cost = L.get_sum_of_distances_to_centers(random_sample_min_centers)
        RANSAC_steps = int(np.sqrt(sample_size))
        for i in range(RANSAC_steps):
            random_sample_current = L.get_sample_of_lines(sample_size)  # sample m lines from uniform distribution on L
            random_sample_current_centers = ExperimentsKMeansForLines.EM_estimetor_k_means_for_lines(random_sample_current, k, is_ground_truth=False, EM_iterations=RANSAC_EM_ITERATIONS)  # send the sample to the EM estimator and get the k points that minimizes the sum of squared distances from the lines in the sample
            current_random_sample_cost = L.get_sum_of_distances_to_centers(random_sample_current_centers)
            if random_sample_min_cost > current_random_sample_cost:
                random_sample_min_cost = current_random_sample_cost  # save the minimum cost
                random_sample_min = random_sample_current
        return random_sample_min

    ######################################################################



    @staticmethod
    def normalized_lines_representation(spans, displacements):
        """
        This method gets a set of n lines represented by an array of n spanning vectors and an array od n displacements
        vectors, and returns these spanning vectors normalized and change each displacements in each line to be the
        closest point on the line to the origin. It is required in order to calculate all the distances later.
        Args:
            spans (np.ndarray) : an array of spanning vectors
            displacements (np.ndarray) : an array of displacements vectors

        Returns:
            spans_normalized, displacements_closest_to_origin (np.ndarray, np,ndarray) : the spanning vectors and the
                displacements vectors normalized and moved as required.
        """

        assert len(spans) > 0, "assert no spanning vectors"
        assert len(displacements) > 0, "assert no displacements vectors"
        assert len(spans) == len(displacements), "number of spanning vectors and displacements vectors not equal"

        dim = np.shape(spans)[1]

        spans_norms = np.sqrt(np.sum(spans ** 2, axis=1))
        spans_norms_repeat = np.repeat(spans_norms, dim, axis=0).reshape(-1, dim)
        spans_normalized = spans / spans_norms_repeat
        # print("spans_normalized: \n", spans_normalized)

        displacements_mul_minus_1 = displacements * -1
        displacements_mul_minus_1_mul_spans_normalized = np.sum(np.multiply(displacements_mul_minus_1, spans), axis=1)
        displacements_mul_minus_1_mul_spans_normalized_repeat_each_col = np.repeat(
            displacements_mul_minus_1_mul_spans_normalized, dim, axis=0).reshape(-1, dim)
        disp_mul_spans_normalized = np.multiply(spans_normalized,
                                                displacements_mul_minus_1_mul_spans_normalized_repeat_each_col)
        displacements_closest_to_origin = disp_mul_spans_normalized + displacements

        # print("displacements_closest_to_origin: \n", displacements_closest_to_origin)

        return spans_normalized, displacements_closest_to_origin

    ######################################################################



    @staticmethod
    def get_means_for_lines_from_file(file_name, k):


        lines_nuber = 500
        lines = []
        with open(file_name, 'rt') as csvfile:
            spamreader = csv.reader(csvfile)
            i = 0
            for row in spamreader:
                row_splited = [element.split(' ') for element in row]
                row_final = [float(entry) for entry in  row_splited[0]]
                lines.append(row_final)
                i += 1
                if i == lines_nuber:
                    break

        L = SetOfLines( lines = lines, is_points = True)
        mens = ExperimentsKMeansForLines.EM_estimetor_k_means_for_lines(L,k)
        return mens
    ######################################################################


