"""Deploy training to DigitalOcean"""

import requests
import os

def deploy_do_training(
    api_token: str,
    region: str = "nyc3",
    size: str = "gpu-h100x8-640gb",
    image: str = "ubuntu-22-04-x64"
):
    """Deploy SXLM training to DigitalOcean GPU Droplet"""

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    droplet_config = {
        "name": "sxlm-training",
        "region": region,
        "size": size,
        "image": image,
        "user_data": """#!/bin/bash
apt-get update
apt-get install -y python3-pip git
git clone https://github.com/rand0mdevel0per/sxlm.git
cd sxlm
pip3 install -r requirements.txt
python3 scripts/train_sft.py
"""
    }

    response = requests.post(
        "https://api.digitalocean.com/v2/droplets",
        headers=headers,
        json=droplet_config
    )

    return response.json()

if __name__ == "__main__":
    api_token = os.getenv("DO_API_TOKEN")
    result = deploy_do_training(api_token)
    print(f"Droplet created: {result}")
