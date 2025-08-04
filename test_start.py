#!/usr/bin/env python3
"""
Test script for the start() function
"""

from parse import start

def test_start_function():
    """Test the start function with different speakers"""
    
    print("🧪 Testing start() function with different speakers\n")
    
    # Test with 'question' speaker (first level questions)
    print("📢 Results for speaker 'question':")
    results = start('question')
    print(f"Found {len(results)} questions")
    for i, result in enumerate(results[:3]):  # Show first 3
        print(f"  {i+1}. {result['message']}")
        print(f"     Audio: {result['audio'] if result['audio'] else 'No audio'}")
    
    print("\n" + "-"*50)
    
    # Test with a non-existent speaker
    print("📢 Results for speaker 'non_existent':")
    results = start('non_existent')
    print(f"Found {len(results)} results")
    
    print("\n" + "-"*50)
    
    # Test with empty speaker
    print("📢 Results for speaker '':")
    results = start('')
    print(f"Found {len(results)} results")

if __name__ == "__main__":
    test_start_function() 