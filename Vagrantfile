# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "generic/ubuntu1804"

  config.vm.provider "virtualbox" do |vb|
    vb.memory = "5000"
    vb.cpus = "4"
  end

  config.vm.synced_folder ".", "/vagrant"

  config.vm.provision "shell", inline: <<-SHELL
    swapoff -a
    apt-get -y update
    apt -y install python3.8 python3-distutils coinor-cbc python2.7 python-pip
    wget https://bootstrap.pypa.io/get-pip.py
    python3.8 get-pip.py
    pip3 install pyomo PyUtilib cloudpickle python-dateutil pandas np
    pip2 install tornado==4.5.3
  SHELL
end
