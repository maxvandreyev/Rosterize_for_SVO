#!/bin/bash
cd $HOME
git clone https://gitlab.com/everest/agent.git
mkdit -p agent/conf
cp -f /vagrant/agent.conf agent/conf
