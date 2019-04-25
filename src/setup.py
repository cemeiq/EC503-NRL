"""

"""

import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import subprocess
from git import Repo
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_score, recall_score

from IPython.display import display, HTML

import utils

GIT_REPOS = dict([
    ("deepwalk", "https://github.com/phanein/deepwalk.git"),
    ("node2vec", "https://github.com/aditya-grover/node2vec.git"),
    ("struc2vec", "https://github.com/leoribeiro/struc2vec.git"),
    ("LINE", "https://github.com/tangjianpku/LINE.git"),
    ("HARP", "https://github.com/GTmac/HARP.git"),
])

GRAPHS = dict([
    ("email-EU-core", {
        "links": "https://snap.stanford.edu/data/email-Eu-core.txt.gz",
        "labels": "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gza",
        "edgelist": "email-Eu-core-department-labels.txt",
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
        "labels": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.all.cmty.txt.gz",
        "edgelist": "com-dblp.ungraph.txt",
    }),
    ("com-amazon", {
        "links": "https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz",
        "labels": "https://snap.stanford.edu/data/bigdata/communities/com-amazon.all.cmty.txt.gz",
        "edgelist": "com-amazon.ungraph.txt",
    })
])

ALGORITHMS = ["deepwalk", "node2vec", "struc2vec", "LINE", "HARP", "LLE", "IsoMap", "MDS", "SpectralEmbedding", "LTSA", "tSNE"]

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
    G = nx.read_edgelist(edgelist_filename)
    G = max(nx.connected_component_subgraphs(G), key=len)
    for _, _, d in G.edges(data=True):
        if "weight" not in d:
            d["weight"] = 1
    nx.write_edgelist(G, edgelist_filename)
    nx.write_weighted_edgelist(G, weighted_edgelist_filename)


def run(algorithm, dataset, **kwargs):
    """Runs the specified algorithm on some dataset"""
    if algorithm == "all":
        for algorithm in ALGORITHMS:
            run(algorithm, dataset, **kwargs)
        return
    if dataset == "all":
        for dataset in GRAPHS:
            run(algorith, dataset, **kwargs)
    if dataset not in GRAPHS:
        raise ValueError("Unknown dataset: {}".format(dataset))
    infile = os.path.join(GRAPH_DIR, target, GRAPHS[dataset]["edgelist"])
    outfile = os.path.join(EMBEDDING_DIR, target, "{}_{}.embeddings".format(algorithm, dataset))
    mkdir_p(os.path.basename(outfile))

    elif algorithm == "deepwalk":
        p = subprocess.run(['python3', 'deepwalk.py', dataset])
        print(p)
    elif algorithm == "node2vec":
        raise NotImplementedError
    elif algorithm == "struc2vec":
        raise NotImplementedError
    elif algorithm == "LINE":
        name, ext = os.path.splitext(infile)
        infile = "{}_weighted{}".format(name, ext)
        raise NotImplementedError
    elif algorithm == "HARP":
        raise NotImplementedError
    elif algorithm == "LLE":
        raise NotImplementedError
    elif algorithm == "MDS":
        raise NotImplementedError
    elif algorithm == "SpectralEmbedding":
        raise NotImplementedError
    elif algorithm == "LTSA":
        raise NotImplementedError
    elif algorithm == "tSNE":
        raise NotImplementedError


def classify(algorithm, dataset, penalty="l2", tol=1e-4, C=1.0, solver="liblinear", **kwargs):
    """Runs classification on the embedding produced by the specified algorithm on some dataset"""
    if algorithm == "all":
        algorithms = list(ALGORITHMS.keys())
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
    for dataset in datasets:
        for algorithm in algorithms:
        clf = LogisticRegression(penalty=penalty, tol=tol, C=C, solver=solver)
        embedding_file = os.path.join(EMBEDDING_DIR, target, "{}_{}.embeddings".format(algorithm, dataset))
        prediction_file = os.path.join(EMBEDDING_DIR, target, "{}_{}.predictions".format(algorithm, dataset))
        X = pd.read_csv(embedding_file, skiprows=1, index_col=0, header=None, sep=' ').sort_index()
        X = (X - X.mean(axis=0)) / np.linalg.norm(X, axis=0)
        y = read_labels(label_filename)
        X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(X, y, X.index)
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        results[(dataset, algorithm)] = ([precision_score(y_test, y_pred, average="micro"),
                    recall_score(y_test, y_pred, average="micro"),
                    f1_score(y_test, y_pred, average="micro"),
                    precision_score(y_test, y_pred, average="macro"),
                    recall_score(y_test, y_pred, average="macro"),
                    f1_score(y_test, y_pred, average="macro"),
                    f1_score(y_test, y_pred, average="weighted")])
    results = pd.DataFrame(results, index=["Precision (micro)", "Recall (micro)", "F1 (micro)",
                                          "Precision (macro)", "Recall (macro)", "F1 (macro)",
                                          "F1-weighted"]).T
    results_file = os.path.join(ROOT_DIR, "results.html")
    with open(results_file, "w") as f:
        f.write(display_html(results.style.apply(highlight_max)._repr_html_(), raw=True))


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
    parser_classify.set_defaults(func="classify")
    parser_classify.add_argument("algorithm", choices=["all"] + ALGORITHMS, default="all", help="Which algorithm's embeddings to use")
    parser_classify.add_argument("dataset", choices=["all"] + list(GRAPHS.keys()), default="all", help="Whose dataset's embedding to use as input")
    parser_classify.add_argument("--penalty", choice=["l1", "l2"], default="l2",
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
