############
Installation
############

Fedstellar is a modular, adaptable and extensible platform for creating centralized and decentralized architectures using Federated Learning. Also, the platform enables the creation of a standard approach for developing, deploying, and managing federated learning applications.

The platform enables developers to create distributed applications that use federated learning algorithms to improve user experience, security, and privacy. It provides features for managing data, managing models, and managing federated learning processes. It also provides a comprehensive set of tools to help developers monitor and analyze the performance of their applications.

Prerequisites
=============
* Python 3.8 or higher (3.11 recommended)
* pip3
* Docker Engine 24.0.4 or higher (24.0.7 recommended, https://docs.docker.com/engine/install/)
* Docker Compose 2.19.0 or higher (2.19.1 recommended, https://docs.docker.com/compose/install/)

.. _deploy_venv:

Deploy a virtual environment
===================================

Virtualenv is a tool to build isolated Python environments.

It's a great way to quickly test new libraries without cluttering your
global site-packages or run multiple projects on the same machine which
depend on a particular library but not the same version of the library.

Since Python version 3.3, there is also a module in the standard library
called `venv` with roughly the same functionality.

Create virtual environment
--------------------------
In order to create a virtual environment called e.g. fedstellar using `venv`, run::

  $ python3 -m venv fedstellar-venv

Activate the environment
------------------------
Once the environment is created, you need to activate it. Just change
directory into it and source the script `Scripts/activate` or `bin/activate`.

With bash::

  $ cd fedstellar-venv
  $ . Scripts/activate
  (fedstellar-venv) $

With csh/tcsh::

  $ cd fedstellar-venv
  $ source Scripts/activate
  (fedstellar-venv) $

Notice that the prompt changes once you are activate the environment. To
deactivate it just type deactivate::

  (fedstellar-venv) $ deactivate
  $

After you have created the environment, you can install fedstellar following the instructions below.

Building from source
====================

Obtaining the platform
--------------------

You can obtain the source code from https://github.com/enriquetomasmb/fedstellar

Or, if you happen to have git configured, you can clone the repository::

    git clone https://github.com/enriquetomasmb/fedstellar.git


Now, you can move to the source directory::

        cd fedstellar

Dependencies
------------

Fedstellar requires the additional packages in order to be able to be installed and work properly.

You can install them using pip::

    pip3 install -r requirements.txt



Checking the installation
-------------------------
Once the installation is finished, you can check
by listing the version of the Fedstellar with the following command line::

    python app/main.py --version


Building the fedstellar participant
====================================

Docker image (CPU version)
-------------------------
You can build the docker image using the following command line in the root directory::

    docker build -t fedstellar -f Dockerfile-cpu .

Docker image (GPU version)
-------------------------
You can build the docker image using the following command line in the root directory::

    docker build -t fedstellar-gpu -f Dockerfile-gpu .

Also, you have to follow the instructions in the following link to install nvidia-container-toolkit::

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

Checking the docker images
==========================
You can check the docker images using the following command line::

        docker images

Running Fedstellar
==================
To run Fedstellar, you can use the following command line::

    python app/main.py [PARAMS]

The first time you run the platform, the fedstellar-frontend docker image will be built. This process can take a few minutes.
    
You can show the PARAMS using::

    python app/main.py --help

The frontend will be available at http://127.0.0.1:5000 (by default)

To change the default port of the frontend, you can use the following command line::

    python app/main.py --webport [PORT]
To change the default port of the statistics endpoint, you can use the following command line::

    python app/main.py --statsport [PORT]

Fedstellar Frontend
==================
You can login with the following credentials::

- User: admin
- Password: admin

If not working the default credentials, send an email to `Enrique Tomás Martínez Beltrán <mailto:enriquetomas@um.es>`_ to get the credentials.


Stop Fedstellar
==================
To stop Fedstellar, you can use the following command line::

    python app/main.py --stop

Be careful, this command will stop all the containers related to Fedstellar: frontend, controller, and participants.


Possible issues during the installation or execution
====================================================

If frontend is not working, check the logs in app/logs/server.log

If any of the following errors appear, take a look at the docker logs of the fedstellar-frontend container::

docker logs fedstellar-frontend

===================================

Network fedstellar_X  Error failed to create network fedstellar_X: Error response from daemon: Pool overlaps with other one on this address space

Solution: Delete the docker network fedstellar_X

    docker network rm fedstellar_X

===================================

Error: Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?

Solution: Start the docker daemon

    sudo dockerd

Solution: Enable the following option in Docker Desktop

Settings -> Advanced -> Allow the default Docker socket to be used
    
    .. image:: _static/docker-required-options.png
        :align: center
        :alt: Docker required options


===================================

Error: Cannot connect to the Docker daemon at tcp://X.X.X.X:2375. Is the docker daemon running?

Solution: Start the docker daemon

    sudo dockerd -H tcp://X.X.X.X:2375

===================================

If frontend is not working, restart docker daemon

    sudo systemctl restart docker