#/bin/bash
# before running the script, please change identity to be root and set password of user ubuntu.

sudo apt-get update
sudo apt-get upgrade
sudo apt install zsh -y

chsh -s $(which zsh)
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# install anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
chmod 755 Anaconda3-2021.11-Linux-x86_64.sh
source ~/.zshrc

# install tensorflow==2.5.0 with conda
conda create -n tf tensorflow==2.5.0
echo "# Automate init for tf" >> ~/.zshrc
echo "conda activate tf" >> ~/.zshrc
source ~/.zshrc
pip install --upgrade tensorflow-hub

# clone api repo
git clone https://github.com/quminzhi/DevAPI.git
python -m pip install -r DevAPI/requirements.txt

# interupt manually and install packages left
pip install Django==3.2.9
pip install djangorestframework
pip install django-filter
pip install django-cors-headers
pip install opencv-python
sudo apt-get install ffmpeg libsm6 libxext6 -y
pip install pillow

# django
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
