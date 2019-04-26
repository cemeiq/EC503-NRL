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

def execute(dataset, output='.'):
    os.chdir("struc2vec")
    print('Current directory is', os.getcwd())
    # create and activate the virtual environment
    print('\nRunning in a virtual environment')
    venv_dir = os.path.join(os.getcwd(), 'project', '.venv')
    if not os.path.exists(venv_dir):
        virtualenv.create_environment(venv_dir)
    execfile(os.path.join(venv_dir, "bin", "activate_this.py"), globals())

    # pip install the requirements of struc2vec in the virtual environment
    print('\nInstalling requirements of struc2vec ...\n')
    pip.main(['install', '--prefix', venv_dir, 'figures'])
    pip.main(['install', '--prefix', venv_dir, 'fastdtw'])
    pip.main(['install', '--prefix', venv_dir, 'gensim'])

    path = os.path.join('..', '..', 'graphs', dataset)
    print('\nRunning struc2vec using', dataset, '...\n')
    command = 'python src/main.py ' \
              '--input ' + os.path.join(path, graphs[dataset]) + ' ' \
              '--num-walks 80 ' \
              '--dimensions 128 ' \
              '--walk-length 40 ' \
              '--OPT1 True ' \
              '--OPT2 True ' \
              '--OPT3 True ' \
              '--output ' + os.path.join(output, 'struc2vec_' + dataset + '.embeddings')
    run = subprocess.run(command, shell=True)
    print(run)

def main():
    parser = argparse.ArgumentParser("struc2vec")
    parser.add_argument("dataset", help="Name of the dataset", type=str)
    args = parser.parse_args()
    execute(args.dataset)
    

if __name__ == "__main__":
    main()
