#!/bin/bash
set -euo pipefail

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Please install Docker before running this script."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null
then
    echo "Docker Compose is not installed. Please install Docker Compose before running this script."
    exit 1
fi

# Check if Python is installed
if ! command -v python &> /dev/null
then
    echo "Python is not installed. Please install Python before running this script."
    exit 1
fi

# Check if Make is installed
if ! command -v make &> /dev/null
then
    echo "Make is not installed. Please install Make before running this script."
    exit 1
fi

# Check if poetry is installed
if ! command -v poetry &> /dev/null
then
    echo "poetry is not installed. Please install poetry before running this script."
    exit 1
fi

# Check the version of Docker
docker_version=$(docker --version | awk '{print $3}' | sed 's/,//')
echo "Docker version $docker_version ✅"

# Check the version of Docker Compose
docker_compose_version=$(docker-compose --version | awk '{print $4}')
echo "Docker Compose version $docker_compose_version ✅"

# Check the version of Python
python_version=$(python --version | awk '{print $2}')
echo "Python version $python_version ✅"

# Check the version of Make
make_version=$(make --version | awk 'NR==1{print $3}')
echo "Make version $make_version ✅"

# Check the version of poetry
poetry_version=$(poetry --version | awk '{print $3}')
echo "poetry version $poetry_version ✅"

# Check the version of virtualenv
virtualenv_version=$(virtualenv --version | awk '{print $2}')
echo "virtualenv version $virtualenv_version ✅"

echo "All required dependencies (Docker, docker-compose, Python, Make, poetry, virtualenv) are installed correctly."

# if yolo is installed, check the version
if command -v yolo &> /dev/null
then
    yolo checks
fi

