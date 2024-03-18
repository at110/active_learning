# active_learning
I am creating this repository for demonstrate Active Learning Strategy using simple example.

# Usage : 


conda create -n active_learning python=3.11

conda activate active_learning

export PYTHONPATH="${PYTHONPATH}:./src"

export AWS_SECRET_ACCESS_KEY=""

export AWS_ACCESS_KEY_ID=""

git clone https://github.com/at110/Spleen-stratified.git

cd Spleen-stratified

pip install -r requirements.txt 

dvc pull -r storage

cd ..

git clone https://github.com/at110/active_learning.git

cd active_learning

pip install -r requirements.txt 





# testing 

python tests/test_train.py

python tests/test_data_loader.py 
