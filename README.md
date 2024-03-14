# active_learning
I am creating this repository for demonstrate Active Learning Strategy using simple example.

# Usage : 
conda create -n active_learning python=3.11

conda activate active_learning

git clone https://github.com/at110/active_learning.git

cd active_learning

pip install -r requirements.txt 

export PYTHONPATH="${PYTHONPATH}:./src"

# testing 

python tests/test_train.py

python tests/test_data_loader.py 
