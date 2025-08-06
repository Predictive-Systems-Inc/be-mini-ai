#!/usr/bin/env python3
"""
Script to download all audio files from the conversation tree JSON
"""

import json
import requests
import os
import re
from urllib.parse import urlparse
from pathlib import Path

def extract_audio_links(json_file_path):
    """Extract all audio links from the JSON file"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    audio_links = []
    
    def extract_links_recursive(obj):
        if isinstance(obj, dict):
            if "audio link" in obj:
                audio_links.append(obj["audio link"])
            for value in obj.values():
                extract_links_recursive(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_links_recursive(item)
    
    extract_links_recursive(data)
    return audio_links

def download_audio_file(url, output_dir):
    """Download a single audio file"""
    try:
        # Extract filename from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if '?' in filename:
            filename = filename.split('?')[0]
        
        output_path = os.path.join(output_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(output_path):
            print(f"File already exists: {filename}")
            return True
        
        print(f"Downloading: {filename}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded: {filename}")
        return True
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main():
    # Create voice directory if it doesn't exist
    voice_dir = "voice"
    os.makedirs(voice_dir, exist_ok=True)
    
    # Extract audio links from the JSON file
    json_file = "data/conversation-tree-v2.json"
    print(f"Extracting audio links from {json_file}...")
    
    audio_links = extract_audio_links(json_file)
    print(f"Found {len(audio_links)} audio links")
    
    # Download each audio file
    successful_downloads = 0
    failed_downloads = 0
    
    for i, link in enumerate(audio_links, 1):
        print(f"\nProgress: {i}/{len(audio_links)}")
        if download_audio_file(link, voice_dir):
            successful_downloads += 1
        else:
            failed_downloads += 1
    
    print(f"\nDownload complete!")
    print(f"Successful downloads: {successful_downloads}")
    print(f"Failed downloads: {failed_downloads}")
    print(f"Total files: {len(audio_links)}")

if __name__ == "__main__":
    main() 