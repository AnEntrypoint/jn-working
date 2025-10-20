"""
Test client for Jet-Nemotron API
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_generation():
    """Test text generation endpoint"""
    print("\nTesting text generation...")

    request_data = {
        "prompt": "Explain quantum computing in simple terms:",
        "max_new_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json=request_data
        )

        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Generated text: {result['generated_text']}")
            print(f"Tokens: {result['tokens_generated']}")
            print(f"Time: {result['generation_time']:.2f}s")
            print(f"Speed: {result['tokens_per_second']:.1f} tok/s")
        else:
            print(f"Error: {response.text}")

        return response.status_code == 200

    except Exception as e:
        print(f"Generation test failed: {e}")
        return False

def test_batch_generation():
    """Test batch generation endpoint"""
    print("\nTesting batch generation...")

    requests_data = [
        {
            "prompt": "Write a haiku about artificial intelligence:",
            "max_new_tokens": 30,
            "temperature": 0.8
        },
        {
            "prompt": "What are the benefits of renewable energy?",
            "max_new_tokens": 40,
            "temperature": 0.6
        }
    ]

    try:
        response = requests.post(
            f"{API_BASE_URL}/generate-batch",
            json=requests_data
        )

        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            results = response.json()
            for i, result in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"Generated: {result['generated_text'][:100]}...")
                print(f"Tokens: {result['tokens_generated']}")
                print(f"Time: {result['generation_time']:.2f}s")
        else:
            print(f"Error: {response.text}")

        return response.status_code == 200

    except Exception as e:
        print(f"Batch generation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Jet-Nemotron API Test Client")
    print("=" * 50)

    # Wait a bit for server to start
    print("Waiting for server to start...")
    time.sleep(5)

    # Run tests
    tests = [
        ("Health Check", test_health),
        ("Text Generation", test_generation),
        ("Batch Generation", test_batch_generation)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)

        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")

    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

if __name__ == "__main__":
    main()