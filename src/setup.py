"""

"""

import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import subprocess
import json
from git import Repo
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_score, recall_score
import sklearn.manifold

from IPython.display import display, HTML, display_html

import utils

GIT_REPOS = dict([
    ("deepwalk", "https://github.com/phanein/deepwalk.git"),
    ("node2vec", "https://github.com/kameranis/node2vec.git"),
    ("struc2vec", "https://github.com/leoribeiro/struc2vec.git"),
    ("LINE", "https://github.com/tangjianpku/LINE.git"),
    ("HARP", "https://github.com/GTmac/HARP.git"),
    ("MDeff", "https://github.com/mdeff/cnn_graph"),
    ("Kipf", "https://github.com/tkipf/gcn"),
    ("SAGE", "https://github.com/williamleif/GraphSAGE"),
    ("LGCN", "https://github.com/williamleif/GraphSAGE")
])

GRAPHS = dict([
    ("email-EU-core", {
        "links": "https://snap.stanford.edu/data/email-Eu-core.txt.gz",
        "labels": "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz",
        "edgelist": "email-Eu-core.txt",
    }),
    ("com-Youtube", {
        "links": "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz",
        "labels": "https://snap.stanford.edu/data/bigdata/communities/com-youtube.all.cmty.txt.gz",
        "edgelist": "com-youtube.ungraph.txt",
    }),
    ("ca-AstroPh", {
        "links": "https://snap.stanford.edu/data/ca-AstroPh.txt.gz",
        "edgelist": "ca-AstroPh.txt",
    }),
    ("facebook", {
        "links": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
        "edgelist": "facebook_combined.txt"
    }),
    ("dblp", {
        "links": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz",
        "labels": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.top5000.cmty.txt.gz",
        "edgelist": "com-dblp.ungraph.txt",
    }),
    ("com-amazon", {
        "links": "https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz",
        "labels": "https://snap.stanford.edu/data/bigdata/communities/com-amazon.top5000.cmty.txt.gz",
        "edgelist": "com-amazon.ungraph.txt",
    }),
    ("PPI", {
        "links": "http://snap.stanford.edu/graphsage/ppi.zip",
        "edgelist": "ppi/ppi/ppi-G.json",
        "labels": "ppi/ppi/ppi-class_map.json",
    }),
])

ALGORITHMS = ["deepwalk", "node2vec", "struc2vec", "LINE", "HARP", "LLE", "IsoMap", "MDS", "SpectralEmbedding", "LTSA", "tSNE", "MDeff", "Kipf", "SAGE", "LGCN"]

FILEPATH = os.path.realpath(__file__)
SRC_DIR = os.path.dirname(FILEPATH)
ROOT_DIR = os.path.dirname(SRC_DIR)
GRAPH_DIR = os.path.join(ROOT_DIR, "graphs")
EMBEDDING_DIR = os.path.join(ROOT_DIR, "embeddings")


def download(target="all", verbose=False, **kwargs):
    if target == "all":
        download(target="code", verbose=verbose)
        download(target="data", verbose=verbose)
    elif target == "code":
        for algorithm in GIT_REPOS.keys():
            download(target=algorithm, verbose=verbose)
    elif target in GIT_REPOS.keys():
        download_path = os.path.join(SRC_DIR, target)
        print(download_path)
        if os.path.exists(download_path):
            print("Download path {path} exists. Skipping downloading {algo}".format(path=download_path, algo=target))
            return
        if verbose:
            print("Downloading {algo} code to {path}.".format(algo=target, path=download_path))
        Repo.clone_from(GIT_REPOS[target], download_path)

    elif target == "data":
        for graph in GRAPHS:
            download(graph, verbose=True)
    elif target in GRAPHS.keys():
        for url in GRAPHS[target].values():
            if not url.startswith("http"):
                continue
            download_path = os.path.join(GRAPH_DIR, target, os.path.basename(url))
            unzip_path = os.path.join(GRAPH_DIR, target, os.path.splitext(os.path.basename(url))[0])
            utils.download(url, download_path)
            utils.unzip(download_path, unzip_path)
    else:
        raise ValueError("Unknown target.")


def clean(target, verbose=False, **kwargs):
    """Cleans the target dataset by making the graph undirected and connected"""
    if target == "all":
        for t in GRAPHS.keys():
            clean(t)
        return
    if target not in GRAPHS.keys():
        raise ValueError("Unknown target.")
    print("Cleaning dataset {}...".format(target))
    target_dir = os.path.join(GRAPH_DIR, target)
    edgelist_filename = os.path.join(target_dir, GRAPHS[target]["edgelist"])
    basename, ext = os.path.splitext(edgelist_filename)
    weighted_edgelist_filename = "{}_weighted{}".format(basename, ext)
    if target == "PPI":
        with open(edgelist_filename, 'r') as f:
            G = nx.readwrite.json_graph.node_link_graph(json.load(f))
        M = sorted([a for a in nx.connected_component_subgraphs(G) if len(a) > 100], key=len)
        f = pd.DataFrame(np.load(os.path.join(target_dir, "ppi/ppi/ppi-feats.npy")))
        c = pd.read_json(os.path.join(target_dir, "ppi/ppi/ppi-class_map.json")).T
        for i, a in enumerate(M):
            nx.write_edgelist(a, os.path.join(target_dir, "ppi_{:02}.edgelist".format(i+1)))
            nodes = [n for n in a.nodes()]
            fi = f.loc[nodes, :]
            ci = c.loc[nodes, :]
            ci.to_json(os.path.join(target_dir, "ppi_{:02d}.classes".format(i+1)))
            fi.to_json(os.path.join(target_dir, "ppi_{:02d}.features".format(i+1)))
    else:
        G = nx.read_edgelist(edgelist_filename, nodetype=int)
        G = max(nx.connected_component_subgraphs(G), key=len)
        nx.write_edgelist(G, edgelist_filename)
        for _, _, d in G.edges(data=True):
            if "weight" not in d:
                d["weight"] = 1
        nx.write_weighted_edgelist(G, weighted_edgelist_filename)


def run(algorithm, dataset, **kwargs):
    """Runs the specified algorithm on some dataset"""
    def executor(algorithm, dataset, infile, outfile):
        print(infile)
        if algorithm == "deepwalk":
            print(subprocess.run(['python3', 'deepwalk.py', dataset, infile, outfile]))
        elif algorithm == "node2vec":
            print(subprocess.run(['python3', 'node2vec.py', dataset, infile, outfile]))
        elif algorithm == "struc2vec":
            print(subprocess.run(['python3', 'struc2vec.py', dataset, infile, outfile]))
        elif algorithm == "LINE":
            name, ext = os.path.splitext(infile)
            infile = "{}_weighted{}".format(name, ext)
            raise NotImplementedError
        elif algorithm == "HARP":
            print(subprocess.run(['python', 'harp.py', dataset, infile, outfile]))
        elif algorithm == "MDS":
            G = nx.read_edgelist(infile, nodetype=int)
            A = {}
            for source, lengths in nx.all_pairs_shortest_path_length(G):
                A[source] = lengths
            A = pd.DataFrame(A)
            A = A.loc[A.index, A.index]
            mds = sklearn.manifold.MDS(n_components=128, max_iter=300, eps=1e-3, n_jobs=2, dissimilarity="precomputed")
            mds.fit(A)
            embedding = pd.DataFrame(mds.embedding_, index=G.nodes(data=False))
            with open(outfile, "w") as f:
                print(len(G), 128, file=f)
                for n, e in embedding.iterrows():
                    print("{} {}".format(n, ' '.join(str(i) for i in e)), file=f)
        elif algorithm == "SpectralEmbedding":
            G = nx.read_edgelist(infile, nodetype=int)
            L = nx.normalized_laplacian_matrix(G)
            eigenvalues, eigenvectors = np.linalg.eig(L.todense())
            eigenvalues = np.real(eigenvalues)
            eigenvectors = np.real(eigenvectors)
            idx = np.argsort(eigenvalues)
            embedding = pd.DataFrame(np.multiply(eigenvectors[:, idx[1:129]], eigenvalues[idx[1:129]]), index=G.nodes(data=False))
            with open(outfile, "w") as f:
                print(len(G), 128, file=f)
                for n, e in embedding.iterrows():
                    print("{} {}".format(n, ' '.join(str(i) for i in e)), file=f)
        elif algorithm == "Mdeff":
            raise NotImplementedError
        elif algorithm == "Kipf":
            raise NotImplementedError
        elif algorithm == "SAGE":
            raise NotImplementedError
        elif algorithm == "LGCN":
            raise NotImplementedError

    if algorithm == "all":
        for algorithm in ALGORITHMS:
            run(algorithm, dataset, **kwargs)
        return
    if dataset == "all":
        for dataset in GRAPHS:
            run(algorithm, dataset, **kwargs)
    if dataset not in GRAPHS:
        raise ValueError("Unknown dataset: {}".format(dataset))
    if dataset=="PPI":
        for i in range(1,25):
            infile = os.path.join(GRAPH_DIR, "PPI", "ppi_{num:02d}.edgelist".format(num=i))
            outfile = os.path.join(EMBEDDING_DIR, dataset, "{}_ppi_{num:02d}.embeddings".format(algorithm, num=i))
            executor(algorithm,dataset,infile,outfile)
        return
    infile = os.path.join(GRAPH_DIR, dataset, GRAPHS[dataset]["edgelist"])
    outfile = os.path.join(EMBEDDING_DIR, dataset, "{}_{}.embeddings".format(algorithm, dataset))
    utils.mkdir_p(os.path.dirname(outfile))
    executor(algorithm, dataset, infile, outfile)


def read_labels(dataset, i=None):
    labels = None
    if "labels" not in GRAPHS[dataset]:
        return labels
    if GRAPHS[dataset]["labels"].startswith("http"):
        filename = os.path.join(GRAPH_DIR, dataset, os.path.basename(os.path.splitext(GRAPHS[dataset]["labels"])[0]))
    else:
        filename = os.path.join(GRAPH_DIR, dataset, GRAPHS[dataset]["labels"])
    if not os.path.exists(filename):
        return labels
    if dataset == "email-EU-core":
        labels = pd.read_csv(filename, header=None, index_col=0, sep=' ')
    elif dataset in ["dblp", "com-amazon", "com-Youtube"]:
        labels = []
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                c = [int(i) for i in line.split()]
                if len(c) > 50:
                    labels.append({node: 1 for node in c})
        labels = pd.DataFrame(labels).T
        labels[np.isnan(labels)] = 0
    elif dataset == "PPI":
        filename = os.path.join(GRAPH_DIR, "PPI", "ppi_{:02d}.classes".format(i))
        labels = pd.read_json(filename)

    return labels


def classify(algorithm, dataset, penalty="l2", tol=1e-4, C=1.0, solver="liblinear", **kwargs):
    """Runs classification on the embedding produced by the specified algorithm on some dataset"""
    def executor(data_set, algo, y, embedding_file):
        clf = LogisticRegression(penalty=penalty, tol=tol, C=C, solver=solver, class_weight="balanced")
        X = pd.read_csv(embedding_file, skiprows=1, index_col=0, header=None, sep=' ').sort_index()
        X = (X - X.mean(axis=0)) / np.linalg.norm(X, axis=0)
        if y is None:
            return None

        # Keep only nodes that are in both X and y
        ind = X.index.intersection(y.index)
        X = X.loc[ind, :]
        y = y.loc[ind, :]
        print(y.shape)

        if len(y.shape) > 1 and all(dim > 1 for dim in y.shape):
            res = []
            for _, labels in y.iteritems():
                X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(X, labels, X.index)
                y_pred = clf.fit(X_train, y_train).predict(X_test)
                res.append([precision_score(y_test, y_pred, average="micro"),
                    recall_score(y_test, y_pred, average="micro"),
                    f1_score(y_test, y_pred, average="micro"),
                    precision_score(y_test, y_pred, average="macro"),
                    recall_score(y_test, y_pred, average="macro"),
                    f1_score(y_test, y_pred, average="macro"),
                    f1_score(y_test, y_pred, average="weighted")])
            res = pd.DataFrame(res, columns=["Precision (micro)", "Recall (micro)", "F1 (micro)",
                                      "Precision (macro)", "Recall (macro)", "F1 (macro)",
                                      "F1-weighted"]).mean(axis=0)

        else:
            X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(X, y, X.index)
            y_pred = clf.fit(X_train, y_train).predict(X_test)
            res = ([precision_score(y_test, y_pred, average="micro"),
                    recall_score(y_test, y_pred, average="micro"),
                    f1_score(y_test, y_pred, average="micro"),
                    precision_score(y_test, y_pred, average="macro"),
                    recall_score(y_test, y_pred, average="macro"),
                    f1_score(y_test, y_pred, average="macro"),
                    f1_score(y_test, y_pred, average="weighted")])
        return res


    if algorithm == "all":
        algorithms = ALGORITHMS
    elif algorithm in ALGORITHMS:
        algorithms = [algorithm]
    else:
        raise ValueError("Unknown algorithm: {}.".format(algorithm))

    if dataset == "all":
        datasets = list(GRAPHS.keys())
    elif dataset in GRAPHS:
        datasets = [dataset]
    else:
        raise ValueError("Unknown dataset: {}.".format(dataset))
    results = {}
    for data_set in datasets:
        for algo in algorithms:
            if data_set == "PPI":
                res = []
                for i in range(24):
                    y = read_labels(data_set, i+1)
                    embedding_file = os.path.join(EMBEDDING_DIR, data_set, "{}_{}_{:02d}.embeddings".format(algo, data_set, i+1))
                    if not os.path.exists(embedding_file):
                        continue
                    res.append(executor(data_set, algo, y, embedding_file))
                if len(res) == 0:
                    continue
                res = pd.DataFrame(res).mean(axis=0)
            else:
                embedding_file = os.path.join(EMBEDDING_DIR, data_set, "{}_{}.embeddings".format(algo, data_set))
                if not os.path.exists(embedding_file):
                    continue
                y = read_labels(data_set)
                res = executor(data_set, algo, y, embedding_file)
                if res is None:
                    continue
            results[(data_set, algo)] = res

    results = pd.DataFrame(results, index=["Precision (micro)", "Recall (micro)", "F1 (micro)",
                                          "Precision (macro)", "Recall (macro)", "F1 (macro)",
                                          "F1-weighted"]).T
    results_file_html = os.path.join(ROOT_DIR, "results", "{}_{}_results.html".format(algorithm, dataset))
    results_file_xlsx = os.path.join(ROOT_DIR, "results", "{}_{}_results.xlsx".format(algorithm, dataset))
    with open(results_file_html, "w") as f:
        f.write(results.style.apply(highlight_max)._repr_html_())
    results.style.apply(highlight_max).to_excel(results_file_xlsx)


def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s.groupby(level=0).transform('max') == s
    return ['background-color: yellow' if v else '' for v in is_max]


def parse_args():
    parser = argparse.ArgumentParser(prog="tool")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbosity")
    subparsers = parser.add_subparsers(help="Available commands")

    # download parser
    parser_download = subparsers.add_parser("download", help="Download target")
    parser_download.set_defaults(func=download)
    parser_download.add_argument("target", choices=["all", "code", "data"] + list(GIT_REPOS.keys()) + list(GRAPHS.keys()), default="all", help="What to download")

    # clean parser
    parser_clean = subparsers.add_parser("clean", help="Clean up the data by keeping only the largest connected component")
    parser_clean.set_defaults(func=clean)
    parser_clean.add_argument("target", choices=["all"] + list(GRAPHS.keys()), default="all", help="Which dataset to clean")

    # run parser
    parser_run = subparsers.add_parser("run", help="Run the algorithm(s) on the target dataset(s)")
    parser_run.set_defaults(func=run)
    parser_run.add_argument("algorithm", choices=["all"] + ALGORITHMS, default="all", help="Which algorithm to run")
    parser_run.add_argument("dataset", choices=["all"] + list(GRAPHS.keys()), default="all", help="On which datasets to run the algorithm")

    # classify parser
    parser_classify = subparsers.add_parser("classify", help="Run Logistic Regression on some ebedding")
    parser_classify.set_defaults(func=classify)
    parser_classify.add_argument("algorithm", choices=["all"] + ALGORITHMS, default="all", help="Which algorithm's embeddings to use")
    parser_classify.add_argument("dataset", choices=["all"] + list(GRAPHS.keys()), default="all", help="Whose dataset's embedding to use as input")
    parser_classify.add_argument("--penalty", choices=["l1", "l2"], default="l2",
            help="Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties.")
    parser_classify.add_argument("--tol", type=float, default=1e-4, help="Tolerance for stopping criteria.")
    parser_classify.add_argument("-C", type=float, default=1.0,
            help="Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.")
    parser_classify.add_argument("--solver", choices=["newton-cg", "lbfgs", "liblinear", "sag", "saga"], default="liblinear", help="Algorithm to use in the optimization problem.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.func(**vars(args))


if __name__ == "__main__":
    main()
