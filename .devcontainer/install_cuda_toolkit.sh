#!/usr/bin/env bash

function detect_distro_arch() {
  local arch=$(uname -m)
  local distrib_id=$(source /etc/lsb-release; echo ${DISTRIB_ID})
  local distrib_release=$(source /etc/lsb-release; echo ${DISTRIB_RELEASE})

  if [[ "${distrib_id}" == "Ubuntu" ]]; then
    if [[ "${arch}" == "x86_64" ]]; then
      local final_arch="x86_64"
    else
      local final_arch="sbsa"
    fi
    if [[ "${distrib_release}" == '20.04' ]]; then
      local final_distro="ubuntu2004"
    elif [[ "${distrib_release}" == '22.04' ]]; then
      local final_distro="ubuntu2204"
    elif [[ "${distrib_release}" == '24.04' ]]; then
      local final_distro="ubuntu2404"
    else
      echo "Unsupported Ubuntu release." >&2
      exit 1
    fi
    echo "${final_distro}/${final_arch}"
    exit 0
  fi

  echo "Unsupported OS." >&2
  exit 1
}

distro_arch="$(detect_distro_arch)"
if [[ $? -ne 0 ]]; then
  echo "Could not auto-detect the distibution and architecture to add CUDA repository." >&2
  return
fi

tmpdir="$(mktemp -d)"
wget "https://developer.download.nvidia.com/compute/cuda/repos/${distro_arch}/cuda-keyring_1.1-1_all.deb" -O "${tmpdir}/cuda-keyring_1.1-1_all.deb"
sudo dpkg -i "${tmpdir}/cuda-keyring_1.1-1_all.deb"
rm -Rf ${tmpdir}

sudo apt update

if [[ -z "${CUDA_VERSION}" ]]; then
    printf "\n\nPlease run the script as follow:\n"
    printf "  \$ CUDA_VERSION=12.6.3-1 $0\n"
    printf " with a version selected from this list:\n"
    apt-cache madison cuda-toolkit
else
    sudo apt install -y "cuda-toolkit=${CUDA_VERSION}"
    printf "\n\nPlease export in your environment:\n  \$ PATH=/usr/local/cuda/bin:\$PATH\n\n"
fi
