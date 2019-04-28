import os
import argparse
import subprocess
import virtualenv
from ipython_genutils.py3compat import execfile
import git
#import shutil

# https://stackoverflow.com/a/17271444/
graphs = {
    "facebook": 'facebook_combined.txt',
    "email-EU-core": "email-Eu-core.txt",
    "dblp": "com-dblp.ungraph",
    "com-Youtube": "com-youtube.ungraph",
    "com-amazon": "com-amazon.ungraph.txt",
    "ca-AstroPh": "ca-AstroPh.txt"
}

def execute(dataset, infile, outfile='.'):
    os.chdir("HARP")
    print('Current directory is', os.getcwd())
    # create and activate the virtual environment
    print('\nRunning in a virtual environment')
    venv_dir = os.path.join(os.getcwd(), 'project', '.venv')
    if not os.path.exists(venv_dir):
        virtualenv.create_environment(venv_dir)
    execfile(os.path.join(venv_dir, "bin", "activate_this.py"), globals())

    magic_graph_dir = os.path.join(os.getcwd(), 'magic-graph')
    if not os.path.exists(magic_graph_dir):
        git.Git(".").clone("git://github.com/phanein/magic-graph")
        os.chdir("magic-graph")
        setup = subprocess.call(os.path.join(venv_dir, 'local', 'bin', 'python2.7') + ' setup.py install', shell=True)
        print(setup)
        os.chdir("..")

    print('\nSetting up harp ...\n')

    # pip install the requirements of harp in the virtual environment
    print('\nInstalling requirements of harp ...\n')
    # https://stackoverflow.com/a/17271444/
    from pip._internal import main as pip
    pip(['install', '--prefix', venv_dir, '-r', 'requirements.txt'])
    #pip.main(['install', '--prefix', venv_dir, '-r', 'requirements.txt'])

    print('\nRunning HARP using', dataset, '...\n')
    command = os.path.join(venv_dir, 'local', 'bin', 'python2.7') + ' ' + os.path.join('src', 'harp.py') + ' ' \
            '--format edgelist ' \
            '--input "' + infile + '" ' \
            '--model deepwalk' \
            '--sfdp-path sfd_osx' \
            '--number-walks 10 ' \
            '--walk-length 40 ' \
            '--workers 1 ' \
            '--output "' + outfile + '"'

    run = subprocess.call(command, shell=True)
    print(run)

def main():
    parser = argparse.ArgumentParser("HARP")
    parser.add_argument("dataset", help="Name of the dataset", type=str)
    parser.add_argument("infile", help="Input file", type=str)
    parser.add_argument("outfile", help="Output file", type=str)
    args = parser.parse_args()
    execute(args.dataset, args.infile, args.outfile)


if __name__ == "__main__":
    main()
