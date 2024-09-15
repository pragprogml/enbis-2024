#!/bin/bash
set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

command_exists() {
    command -v "$1" &> /dev/null
}

check_version() {
    local command=$1
    local version_command=$2
    local name=$3

    if command_exists "$command"; then
        version=$(eval "$version_command")
        echo -e "${GREEN}$name version $version ✅${NC}"
    else
        echo -e "${RED}$name is not installed. Please install $name before running this script.${NC}"
        return 1
    fi
}

check_min_version() {
    local version=$1
    local min_version=$2

    if [ "$(printf '%s\n' "$min_version" "$version" | sort -V | head -n1)" = "$min_version" ]; then
        return 0
    else
        return 1
    fi
}

dependencies=(
    "docker:docker --version | awk '{print \$3}' | sed 's/,//':Docker:20.10.0"
    "docker:docker compose version --short 2>/dev/null || docker-compose --version | awk '{print \$3}' | sed 's/,//':Docker Compose:1.27.0"
    "python:python --version 2>&1 | awk '{print \$2}':Python:3.7.0"
    "make:make --version | awk 'NR==1{print \$3}':Make:4.0"
    "poetry:poetry --version | awk '{print \$3}':poetry:1.0.0"
)

for dep in "${dependencies[@]}"; do
    IFS=':' read -r command version_command name min_version <<< "$dep"
    if ! check_version "$command" "$version_command" "$name"; then
        exit 1
    fi
done

echo -e "${GREEN}All required dependencies are installed correctly.${NC}"

if command_exists yolo; then
    echo -e "${YELLOW}YOLO is installed. Running YOLO checks...${NC}"
    yolo checks
else
    echo -e "${YELLOW}YOLO is not installed. Skipping YOLO checks.${NC}"
fi
