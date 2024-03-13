from scipy import sparse as sp, io as scio
import os, time
import numpy as np
import faiss
from faiss import normalize_L2
import scipy.sparse as sp
from scipy.sparse import coo_matrix


# cosine_similarity calculation
def cosine_similar(training_vector, k=10):
    d = training_vector.shape[1]  # dimension
    normalize_L2(training_vector)
    index = faiss.IndexFlatIP(d)
    index.train(training_vector)
    print(index.is_trained)
    index.add(training_vector)
    print("search")
    D, I = index.search(training_vector, k)
    print(I[:5, :])  # index of the most 5 nearst nodes
    print(D[:5, :])  # similarity score of the most 5 nearst nodes
    return D, I


def cosine_similar_gpu(training_vector, k=10):
    ngpus = faiss.get_num_gpus()
    print("number of GPUs:", ngpus)
    d = training_vector.shape[1]
    normalize_L2(training_vector)
    cpu_index = faiss.IndexFlatIP(d)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)  # build the index
    gpu_index.add(training_vector)  # add vectors to the index
    print(gpu_index.ntotal)
    D, I = gpu_index.search(training_vector, k)  # actual search
    return D, I


if __name__ == "__main__":
    sigma = 5
    start = time.time()
    data = scio.loadmat(os.path.join("data", "PaviaU.mat"))
    label = scio.loadmat(os.path.join("data", "PaviaU_gt.mat"))
    X = data["paviaU"]
    Y = label["paviaU_gt"]
    X_2d = np.float32(X.reshape(-1, X.shape[-1]))
    # D, I  = cosine_similar(X_2d, 10000)
    D, I = cosine_similar_gpu(X_2d, 10)
    D = D[:, 1:]
    I = I[:, 1:]
    # print('I', I[:5,:])                   # neighbors of the 5 first queries
    # print('D', D[:5,:])

    _row = np.array([i for i in range(I.shape[0]) for j in range(I.shape[1])])
    _col = I.reshape(
        -1,
    )
    _data = np.exp(
        -(
            1
            - D.reshape(
                -1,
            )
        )
        / sigma
    )
    w = coo_matrix(
        (_data, (_row, _col)), shape=(X_2d.shape[0], X_2d.shape[0]), dtype=float
    ).tocsr()
    print(w.shape, np.nonzero(w)[0])
    sp.save_npz("./data/PaviaU_adj.npz", w)
