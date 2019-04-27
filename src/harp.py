import os
import argparse
import subprocess
import virtualenv
import pip
from ipython_genutils.py3compat import execfile
import git
import shutil

graphs = {
    "facebook": 'facebook_combined.txt',
    "email-EU-core": "email-Eu-core.txt",
    "dblp": "com-dblp.ungraph",
    "com-Youtube": "com-youtube.ungraph",
    "com-amazon": "com-amazon.ungraph.txt",
    "ca-AstroPh": "ca-AstroPh.txt"
}

def execute(dataset, output='.'):
    os.chdir("harp")
    print('Current directory is', os.getcwd())
    # create and activate the virtual environment
    print('\nRunning in a virtual environment')
    venv_dir = os.path.join(os.getcwd(), 'project', '.venv')
    if not os.path.exists(venv_dir):
        virtualenv.create_environment(venv_dir)
    execfile(os.path.join(venv_dir, "bin", "activate_this.py"), globals())

   
    print('\nSetting up harp ...\n')
    git.Git("magic-graph").clone("git://github.com/phanein/magic-graph")
    setup = subprocess.run("python magic-graph/setup.py install", shell=True)
    print(setup)

    # pip install the requirements of harp in the virtual environment
    print('\nInstalling requirements of harp ...\n')
    pip.main(['install', '--prefix', venv_dir, '-r', 'requirements.txt'])

    path = os.path.join('..', '..', 'graphs', dataset)
    print('\nRunning HARP using', dataset, '...\n')
    command = 'python ' + os.path.join('src', 'harp.py') + ' ' \
            '--format edgelist ' \
            '--input ' + os.path.join(path, graphs[dataset]) + ' ' \
            '--model deepwalk' \
            '--sfdp-path sfd_osx' \
            '--number-walks 10 ' \
            '--walk-length 40 ' \
            '--workers 1 ' \
            '--output ' + os.path.join(output, 'harp_' + dataset + '.embeddings')              
              
    run = subprocess.run(command, shell=True)
    print(run)

def main():
    parser = argparse.ArgumentParser("HARP")
    parser.add_argument("dataset", help="Name of the dataset", type=str)
    args = parser.parse_args()
    execute(args.dataset)
    

if __name__ == "__main__":
    main()
