import numpy as np
from sklearn import manifold
from sklearn.metrics.pairwise import euclidean_distances

# when applied to graphs, one will need to pass in the shortest path distance matrix
def MDS(distances, k=2):
    '''
        Returns features computed with Multidimensional Scaling from pairwise distances stored in distances.
        This is an nxk matrix containing the embeddings of the n points in R^k
    '''

    mds = manifold.MDS(n_components=k, max_iter=3000, eps=1e-12,
                   dissimilarity="precomputed")
    pos = mds.fit(distances).embedding_

    # non-metric case:
    '''
        nmds = manifold.MDS(n_components=k, metric=False, max_iter=3000, eps=1e-4,
                    dissimilarity="precomputed")
        npos = nmds.fit_transform(similarities, init=pos)
    '''
    return pos

def SpectralEmbedding(A, k=1):
    '''
        Returns first k coordinates of the n eigenvectors of the normalized laplacian Matrix
    '''
    D = np.diag(A.sum(axis=1))
    C = np.diag(A.sum(axis=1) ** (-0.5))
    Lap = C@(D-A)@C
    k_eigenvectors = np.linalg.eig(Lap)[1][:,-(k+1):-2]
    k_eigenvalues = np.linalg.eig(Lap)[0][:,-(k+1):-2]
    return k_eigenvectors,k_eigenvalues

if __name__=="__main__":
    X = np.array([[0, 3, 4],[3,0,5],[4,5,0]])
    Y = MDS(X)
    print("Y:{}".format(Y))
    print("type(Y):{}".format(type(Y)))
    print(euclidean_distances(Y))
    print('Spectral Embedding:')
    emb = SpectralEmbedding(np.array([[0,1,1],[1,0,0],[1,0,0]]))
    print(emb)
