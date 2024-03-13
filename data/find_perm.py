import numpy as np
import numpy as np
import os
from scipy import io as scio
import itertools
from data_utils import DatasetInfo
from sklearn.metrics import pairwise_distances


# following the https://python.plainenglish.io/hungarian-algorithm-introduction-python-implementation-93e7c0890e15
#  Time complexity O(#classes^3) and memory complexity O(#class^2)
def min_zero_row(zero_mat, mark_zero):
    """
    The function can be splitted into two steps:
    #1 The function is used to find the row which containing the fewest 0.
    #2 Select the zero number on the row, and then marked the element corresponding row and column as False
    """

    # Find the row
    min_row = [99999, -1]

    for row_num in range(zero_mat.shape[0]):
        if np.sum(zero_mat[row_num] == True) > 0 and min_row[0] > np.sum(
            zero_mat[row_num] == True
        ):
            min_row = [np.sum(zero_mat[row_num] == True), row_num]

    # Marked the specific row and column as False
    zero_index = np.where(zero_mat[min_row[1]] == True)[0][0]
    mark_zero.append((min_row[1], zero_index))
    zero_mat[min_row[1], :] = False
    zero_mat[:, zero_index] = False


def mark_matrix(mat):
    """
    Finding the returning possible solutions for LAP problem.
    """

    # Transform the matrix to boolean matrix(0 = True, others = False)
    cur_mat = mat
    zero_bool_mat = cur_mat == 0
    zero_bool_mat_copy = zero_bool_mat.copy()

    # Recording possible answer positions by marked_zero
    marked_zero = []
    while True in zero_bool_mat_copy:
        min_zero_row(zero_bool_mat_copy, marked_zero)

    # Recording the row and column positions seperately.
    marked_zero_row = []
    marked_zero_col = []
    for i in range(len(marked_zero)):
        marked_zero_row.append(marked_zero[i][0])
        marked_zero_col.append(marked_zero[i][1])

    # Step 2-2-1
    non_marked_row = list(set(range(cur_mat.shape[0])) - set(marked_zero_row))

    marked_cols = []
    check_switch = True
    while check_switch:
        check_switch = False
        for i in range(len(non_marked_row)):
            row_array = zero_bool_mat[non_marked_row[i], :]
            for j in range(row_array.shape[0]):
                # Step 2-2-2
                if row_array[j] == True and j not in marked_cols:
                    # Step 2-2-3
                    marked_cols.append(j)
                    check_switch = True

        for row_num, col_num in marked_zero:
            # Step 2-2-4
            if row_num not in non_marked_row and col_num in marked_cols:
                # Step 2-2-5
                non_marked_row.append(row_num)
                check_switch = True
    # Step 2-2-6
    marked_rows = list(set(range(mat.shape[0])) - set(non_marked_row))

    return (marked_zero, marked_rows, marked_cols)


def adjust_matrix(mat, cover_rows, cover_cols):
    cur_mat = mat
    non_zero_element = []

    # Step 4-1
    for row in range(len(cur_mat)):
        if row not in cover_rows:
            for i in range(len(cur_mat[row])):
                if i not in cover_cols:
                    non_zero_element.append(cur_mat[row][i])
    min_num = min(non_zero_element)

    # Step 4-2
    for row in range(len(cur_mat)):
        if row not in cover_rows:
            for i in range(len(cur_mat[row])):
                if i not in cover_cols:
                    cur_mat[row, i] = cur_mat[row, i] - min_num
    # Step 4-3
    for row in range(len(cover_rows)):
        for col in range(len(cover_cols)):
            cur_mat[cover_rows[row], cover_cols[col]] = (
                cur_mat[cover_rows[row], cover_cols[col]] + min_num
            )
    return cur_mat


def hungarian_algorithm(mat):
    dim = mat.shape[0]
    cur_mat = mat

    # Step 1 - Every column and every row subtract its internal minimum
    for row_num in range(mat.shape[0]):
        cur_mat[row_num] = cur_mat[row_num] - np.min(cur_mat[row_num])

    for col_num in range(mat.shape[1]):
        cur_mat[:, col_num] = cur_mat[:, col_num] - np.min(cur_mat[:, col_num])
    zero_count = 0
    while zero_count < dim:
        # Step 2 & 3
        ans_pos, marked_rows, marked_cols = mark_matrix(cur_mat)
        zero_count = len(marked_rows) + len(marked_cols)

        if zero_count < dim:
            cur_mat = adjust_matrix(cur_mat, marked_rows, marked_cols)

    return ans_pos


def ans_calculation(mat, pos):
    total = 0
    ans_mat = np.zeros((mat.shape[0], mat.shape[1]))
    for i in range(len(pos)):
        total += mat[pos[i][0], pos[i][1]]
        ans_mat[pos[i][0], pos[i][1]] = mat[pos[i][0], pos[i][1]]
    return total, ans_mat


def linear_assignment(cost_matrix):
    ans_pos = hungarian_algorithm(cost_matrix.copy())  # Get the element position.
    # print(ans_pos)
    ans, ans_mat = ans_calculation(
        cost_matrix, ans_pos
    )  # Get the minimum or maximum value and corresponding matrix.
    ans_pos.sort(key=lambda x: x[0])
    print(ans_pos)
    column = [x[1] for x in ans_pos]
    print("Endermember map:", column)

    # Show the result
    print(f"Linear Assignment problem result: {ans:.0f}\n{ans_mat}")


# serially compute along num_classes and need large memory
def find_perm_deprecated_v1(distance, num_classes):
    ords = list(itertools.permutations([i for i in range(num_classes)]))
    ords_array = np.array(ords)
    index = np.arange(num_classes).reshape(1, num_classes)
    index = np.repeat(index, len(ords), axis=0)
    permutation_errors = np.array(
        [distance[index[:, i], ords_array[:, i]] for i in range(num_classes)]
    ).T
    errors = np.sum(permutation_errors, axis=1)
    print(errors.shape)
    min_ord = ords[np.argmin(errors)]
    print("best permutation:", min_ord)
    max_ord = ords[np.argmax(errors)]
    print("worst permutation:", max_ord)
    print("error for best permutation:", errors[np.argmin(errors)])
    print("error for worst permutation:", errors[np.argmax(errors)])


# serially compute along ords and need large memory and long time
def find_perm_deprecated_v2(distance, num_classes):
    ords = list(itertools.permutations([i for i in range(num_classes)]))
    num_perm = len(ords)
    errors = np.zeros(num_perm)
    true_index = [i for i in range(num_classes)]
    for ord in range(num_perm):
        errors[ord] = sum([distance[i, j] for i, j in zip(true_index, ords[ord])])
    min_ord = ords[np.argmin(errors)]
    print("best permutation:", min_ord)
    max_ord = ords[np.argmax(errors)]
    print("worst permutation:", max_ord)
    print("error for best permutation:", errors[np.argmin(errors)])
    print("error for worst permutation:", errors[np.argmax(errors)])
    return min_ord


def find_perm(dataset):
    # load dataset
    datasetinfo = DatasetInfo(dataset)
    label = scio.loadmat(os.path.join(f"{dataset}/raw", f"{dataset}_gt.mat"))[
        f"{dataset}_gt"
    ]
    label = np.int32(label)
    if dataset == "paviaU":
        unmixing_result = scio.loadmat(
            os.path.join("paviaU/unmixing", "unmixing_result.mat")
        )
        A_pred = unmixing_result["A_MBO_fixed"].T
    elif dataset == "KSC":
        unmixing_result = scio.loadmat(
            os.path.join("KSC/unmixing", "unmixing_result.mat")
        )
        A_pred = unmixing_result["A_MBO_fixed"].T
    elif dataset == "Houston":
        A_pred = scio.loadmat(os.path.join("Houston/unmixing", "A_init.mat"))[
            "A_init"
        ].T
    print("A pred shape", A_pred.shape)
    m, n = datasetinfo.m, datasetinfo.n
    num_classes = datasetinfo.num_classes

    # compute the distance matrix
    # Note considering the difference between matlab and python when reshaping an array
    A_pred = A_pred.reshape(m, n, num_classes, order="F").reshape(
        -1, num_classes, order="C"
    )

    label_1d = label.reshape(
        -1,
    )
    A_ref = np.zeros((label_1d.shape[0], label_1d.max() + 1))
    A_ref[np.arange(label_1d.size), label_1d] = 1
    A_ref = A_ref[:, 1:]
    # only calculate the labeled piexes
    labeled_indices = np.nonzero(label_1d)[0]
    distance = pairwise_distances(A_ref[labeled_indices].T, A_pred[labeled_indices].T)
    linear_assignment(distance)


if __name__ == "__main__":
    dataset = "paviaU"
    find_perm(dataset)
