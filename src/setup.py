"""

"""

import warnings
warnings.filterwarnings("ignore")

import os
import argparse
from git import Repo
import networkx as nx

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
        "labels": "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz"
    }),
    ("com-Youtube", {
        "links": "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz",
        "labels": "https://snap.stanford.edu/data/bigdata/communities/com-youtube.all.cmty.txt.gz"
    }),
    ("ca-AstroPh", {
        "links": "https://snap.stanford.edu/data/ca-AstroPh.txt.gz"
    }),
    ("facebook", {
        "links": "https://snap.stanford.edu/data/facebook_combined.txt.gz"
    }),
    ("dblp", {
        "links": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz",
        "labels": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.all.cmty.txt.gz"
    }),
    ("com-amazon", {
        "links": "https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz",
        "labels": "https://snap.stanford.edu/data/bigdata/communities/com-amazon.all.cmty.txt.gz"
    })
])

ALGORITHMS = ["deepwalk", "node2vec", "struc2vec", "LINE", "HARP", "LLE", "IsoMap", "MDS", "SpectralEmbedding", "LTSA", "tSNE"]

FILEPATH = os.path.realpath(__file__)
SRC_DIR = os.path.dirname(FILEPATH)
ROOT_DIR = os.path.dirname(SRC_DIR)


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
            download_path = os.path.join(ROOT_DIR, "graphs", target, os.path.basename(url))
            unzip_path = os.path.join(ROOT_DIR, "graphs", target, os.path.splitext(os.path.basename(url))[0])
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
    target_dir = os.path.join(ROOT_DIR, "graphs", target)
    edgelist_filename = os.path.join(target_dir, os.path.splitext(os.path.basename(GRAPHS[target]["links"]))[0])
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
    elif algorithm == "deepwalk":
        raise NotImplementedError
    elif algorithm == "node2vec":
        raise NotImplementedError
    elif algorithm == "struc2vec":
        raise NotImplementedError
    elif algorithm == "LINE":
        raise NotImplementedError
    elif algorithm == "HARP":
        raise NotImplementedError
    elif algorithm == "LLE":
        raise NotImplementedError
    elif algorithm == "IsoMap":
        raise NotImplementedError
    elif algorithm == "MDS":
        raise NotImplementedError
    elif algorithm == "SpectralEmbedding":
        raise NotImplementedError
    elif algorithm == "LTSA":
        raise NotImplementedError
    elif algorithm == "tSNE":
        raise NotImplementedError



def parse_args():
    parser = argparse.ArgumentParser(prog="tool")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbosity")
    subparsers = parser.add_subparsers(help="Available commands")

    # download parser
    parser_download = subparsers.add_parser("download", help="Downloads target")
    parser_download.set_defaults(func=download)
    parser_download.add_argument("target", choices=["all", "code", "data"] + list(GIT_REPOS.keys()) + list(GRAPHS.keys()), default="all", help="What to download")

    # clean parser
    parser_clean = subparsers.add_parser("clean", help="Cleans up the data")
    parser_clean.set_defaults(func=clean)
    parser_clean.add_argument("target", choices=["all"] + list(GRAPHS.keys()), default="all", help="Which dataset to clean")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.func(**vars(args))


if __name__ == "__main__":
    main()

