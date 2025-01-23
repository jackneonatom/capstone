ssh -i id_rsa.pub ubuntu@164.152.104.200  #ssh into public server

#update package list
sudo apt update 
#update system
sudo apt upgrade
#install nginx and view status
sudo apt install nginx
systemctl status nginx
#install text editor vim
sudo apt install vim
#edit nginx config file
sudo vim /etc/nginx/sites-available/default
cd /var/www/html
ls
#change permissions so file can be added
sudo chmod 777 /var/www/html/
#install docker
curl -fsSL https://get.docker.com -o get-docker.sh

sh get-docker.sh
# installing mongodb on standard linux system
docker run --restart=always -d -p 27017:27017 --name mongodb mongo

# install mongo on raspberry pi 3b
sudo docker pull mongo:3.6
sudo docker run -d --name my-mongo -p 27017:27017 mongo:3.6
# ---------------------------------------------------------------------------
#setting up fastapi 
pip install fastAPI uvicorn motor datetime
#setting up raspberry pi gpio

pip3 install RPi.GPIO
#check docker containers
sudo docker ps
#########
cat default
cd ~
sudo raspi-config #then change the hostname to whatever you want itsudo -H ./install
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

logout #have to logout to reload node js when installed
# install node.js and setup contant running of python file
nvm install 20
npm install pm2 -g
pm2 start app.py
# check pm2 logs
pm2 monit
#python 3 installed version
python3 --version
#stop constant runnug app
pm2 stop app.py
# install python packages system wide on linux system
sudo apt install python3-fastapi
sudo apt install python3-motor
#run program on linux system
python3 app.py
#check status of constantly runnig=ng apps
pm2 status
#delete constantly runnign program
pm2 delete app
#start it back up choosing interpreter 
pm2 start app.py --interpreter python3
#check pm2 logs
pm2 logs app
pm2 startup  #only need to run once
# startup should then return a path and a command that you need to run to actually do the startup
sudo env PATH=$PATH:/home/demo/.nvm/versions/node/v20.15.0/bin /home/demo/.nvm/versions/node/v20.15.0/lib/node_modules/pm2/bin/pm2 startup systemd -u demo --hp /home/demo
pm2 save


# when CORS policy still wasnt working after putting it in orgins make sure to do a pm2 stop and start again,


# to access database through docker

sudo docker exec -it 30f30a01f072 bash
mongo PVdemo
show dbs
 use PVdemo
 show collections
 db.readings.find()

 sudo apt update
sudo apt install vlc -y
vlc v4l2:///dev/video0

#install postgres database in docker container

docker run --name postgresdb -e POSTGRES_USER=admin -e POSTGRES_PASSWORD=admin123 -e POSTGRES_DB=vehiclecounter -p 5432:5432 -d postgres:latest -e TZ=America/Bogota

# frequency over a time period or different periods and then compare data
#have drop down menu on each icon that shows different frequecnies in different time periods
