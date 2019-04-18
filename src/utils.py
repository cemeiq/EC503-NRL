import os
import sys
import errno
import random

import requests

from zipfile import ZipFile
import gzip
import shutil
from progressbar import ProgressBar, widgets
from os.path import basename, realpath, dirname, isfile, isdir


# https://stackoverflow.com/a/600612
def mkdir_p(path):
    """Attempts to create path. If path exists does nothing."""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and isdir(path):
            pass
        else:
            raise


def download(url, filename):
    """Attempts to download url to filename unless the two files have the same size"""
    r = requests.get(url, stream=True)
    size = int(r.headers.get('Content-Length'))
    bname = basename(filename)

    if size and isfile(filename) and os.path.getsize(filename) == size:
        print('File %s already exists, skipping download' % bname, file=sys.stderr)
        return

    currdown = 0.0
    fmt = [
        'Downloading %s: ' % bname, widgets.Bar(), ' ',
        widgets.RotatingMarker(), ' ', widgets.Percentage(), ' ',
        widgets.FileTransferSpeed(), ' ', widgets.ETA()
    ]

    progress = ProgressBar(maxval=size or 100, widgets=fmt)
    progress.start()

    # https://stackoverflow.com/a/5137509
    mkdir_p(dirname(realpath(filename)))

    # https://stackoverflow.com/a/16696317
    with open(filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)

            currdown += len(chunk)

            if size:
                progress.update(currdown)
            else:
                progress.update(random.randint(0, 100))

    progress.finish()


def unzip(filename, folder=None):
    """Unzips filename to folder"""
    print('Decompressing {}...'.format(basename(filename)))

    if filename.endswith("tar.gz"):
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall()

    elif filename.endswith(".gz"):
        with open(os.path.splitext(filename)[0], 'wb') as f_out:
            with gzip.GzipFile(filename, 'rb') as f_in:
                f_out.write(f_in.read())

    elif filename.endswith(".zip"):
        with ZipFile(filename) as zf:
            zf.extractall(path=folder)

    else:
        raise ValueError("Was expecting a .zip or .gz file. Instead got {}".format(filename))
