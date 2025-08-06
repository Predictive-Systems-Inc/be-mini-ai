#!/usr/bin/env python3
"""
Script to download all audio files from the conversation tree JSON
"""

import json
import os
import urllib.request
import re
from urllib.parse import urlparse
from pathlib import Path

def extract_audio_urls(data, urls=None):
    """Recursively extract all audio URLs from the JSON structure"""
    if urls is None:
        urls = []
    
    if isinstance(data, dict):
        # Check for audio link
        if "audio link" in data:
            urls.append({
                "url": data["audio link"],
                "tag": data.get("tag", "unknown"),
                "speaker": data.get("speaker", "unknown")
            })
        
        # Recursively process all values
        for value in data.values():
            extract_audio_urls(value, urls)
    
    elif isinstance(data, list):
        # Recursively process all items
        for item in data:
            extract_audio_urls(item, urls)
    
    return urls

def download_audio_file(url, tag, speaker, voice_dir):
    """Download a single audio file"""
    try:
        # Create filename from tag
        filename = f"{tag}.wav"
        filepath = os.path.join(voice_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(filepath):
            print(f"Skipping {filename} (already exists)")
            return True
        
        print(f"Downloading {filename} ({speaker})...")
        
        # Download the file using urllib
        urllib.request.urlretrieve(url, filepath)
        
        print(f"✓ Downloaded {filename}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download {tag}: {e}")
        return False

def main():
    # Load the conversation tree
    with open('conversation-tree-simplified-with-audio.json', 'r') as f:
        data = json.load(f)
    
    # Extract all audio URLs
    print("Extracting audio URLs...")
    audio_urls = extract_audio_urls(data)
    
    print(f"Found {len(audio_urls)} audio files to download")
    
    # Create voice directory
    voice_dir = "./voice"
    os.makedirs(voice_dir, exist_ok=True)
    print(f"Created voice directory: {voice_dir}")
    
    # Download all audio files
    successful = 0
    failed = 0
    
    for audio_info in audio_urls:
        success = download_audio_file(
            audio_info["url"], 
            audio_info["tag"], 
            audio_info["speaker"], 
            voice_dir
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\nDownload complete!")
    print(f"✓ Successfully downloaded: {successful}")
    print(f"✗ Failed downloads: {failed}")
    print(f"Total files: {len(audio_urls)}")

if __name__ == "__main__":
    main() 