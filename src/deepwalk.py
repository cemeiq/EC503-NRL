import os
import argparse
import subprocess
import virtualenv
import pip
from ipython_genutils.py3compat import execfile

graphs = {
    "facebook": 'facebook_combined.txt',
    "email-EU-core": "email-Eu-core.txt",
    "dblp": "com-dblp.ungraph",
    "com-Youtube": "com-youtube.ungraph",
    "com-amazon": "com-amazon.ungraph.txt",
    "ca-AstroPh": "ca-AstroPh.txt"
}

def execute(dataset, infile, outfile='.'):
    os.chdir("deepwalk")
    print('Current directory is', os.getcwd())
    # create and activate the virtual environment
    print('\nRunning in a virtual environment')
    venv_dir = os.path.join(os.getcwd(), 'project', '.venv')
    if not os.path.exists(venv_dir):
        virtualenv.create_environment(venv_dir)
    execfile(os.path.join(venv_dir, "bin", "activate_this.py"), globals())

    # pip install the requirements of deepwalk in the virtual environment
    print('\nInstalling requirements of deepwalk ...\n')
    pip.main(['install', '--prefix', venv_dir, '-r', 'requirements.txt'])

    # Setup deepwalk
    print('\nSetting up deepwalk ...\n')
    setup = subprocess.run("python setup.py install", shell=True)
    print(setup)

    path = os.path.join('..', '..', 'graphs', dataset)
    print('\nRunning deepwalk using', dataset, '...\n')
    command = 'deepwalk --format edgelist ' \
              '--input "' + infile + '" ' \
              '--max-memory-data-size 0 ' \
              '--number-walks 10 ' \
              '--representation-size 128 ' \
              '--walk-length 40 ' \
              '--workers 1 ' \
              '--output "' + outfile + '"'
    run = subprocess.run(command, shell=True)
    print(run)

def main():
    parser = argparse.ArgumentParser("deepwalk")
    parser.add_argument("dataset", help="Name of the dataset", type=str)
    parser.add_argument("infile", help="Input file", type=str)
    parser.add_argument("outfile", help="Output file", type=str)
    args = parser.parse_args()
    execute(args.dataset, args.infile, args.outfile)


if __name__ == "__main__":
    main()
