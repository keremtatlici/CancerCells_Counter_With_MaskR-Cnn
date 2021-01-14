##Cihaz Özellikleri##
Ubuntu 18.04.05


##Kurulumlar##
Klasörde gerekli dosyalar mevcuttur.

###Conda Kurulumu###
#Kaynak : https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart

#Adımlar:
bash gerekliSetuplarVeDosyalar/Anaconda3-2020.07-Linux-x86_64.sh

#Kurulum tamamlandıktan sonra : 

source ~/.bashrc

###Conda Virtual Env Kurulumu###
conda create --name env369 --file gerekliSetuplarVeDosyalar/virtualEnvSetup.txt

conda activate env369

sudo apt-get update && sudo apt-get upgrade

pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

sudo apt install gcc

pip install pycocotools==2.0.2

pip install tensorboard==2.3.0

pip install qhoptim

pip install ipython==5.5.0

sudo ubuntu-drivers autoinstall


 #Bu linkteki adımları gerçekleştirdim : https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130

#o linkteki adımlar : 

sudo rm /etc/apt/sources.list.d/cuda*
sudo apt remove --autoremove nvidia-cuda-toolkit
sudo apt remove --autoremove nvidia-*

sudo apt update
sudo add-apt-repository ppa:graphics-drivers
sudo apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'

sudo apt update
sudo apt install cuda-10-1
sudo apt install libcudnn7

sudo nano ~/.profile
#burada root ve kullanıcının .profile dosya çiftine yazıyoruz!

# set PATH for cuda 10.1 installation
if [ -d "/usr/local/cuda-10.1/bin/" ]; then
    export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi


#artık şu iki komutu çalıştırarak kurulumun başarılı olup olmadığını anlayabiliriz.ama önce bilgisayarı yeniden başlatalım.

nvidia-smi

 #ve 

nvcc -V

#ve 

/sbin/ldconfig -N -v $(sed ‘s/:/ /’ <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn




