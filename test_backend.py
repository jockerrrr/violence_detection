import requests
import os
import time

def test_health_endpoint():
    """Test the health endpoint"""
    try:
        response = requests.get('http://localhost:5001/health')
        if response.status_code == 200:
            print("✅ Health endpoint is working")
            return True
        else:
            print(f"❌ Health endpoint returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error connecting to health endpoint: {str(e)}")
        return False

def main():
    """Main test function"""
    print("Testing backend API...")
    
    # Test health endpoint
    if not test_health_endpoint():
        print("Backend server may not be running. Please start it first.")
        return
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
