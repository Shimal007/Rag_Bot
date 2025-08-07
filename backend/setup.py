import requests
import json
import subprocess
import time
import sys

OLLAMA_BASE_URL = "http://localhost:11434"

def check_ollama_service():
    """Check if Ollama service is running"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service is running")
            return True
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama at http://localhost:11434")
        print("   Make sure Ollama is running with: ollama serve")
        return False
    except requests.exceptions.Timeout:
        print("❌ Ollama connection timed out")
        return False
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return False

def get_available_models():
    """Get list of available models"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"📋 Available models ({len(models)}):")
            if models:
                for model in models:
                    name = model.get('name', 'Unknown')
                    size = model.get('size', 0)
                    size_gb = size / (1024**3) if size else 0
                    print(f"   - {name} ({size_gb:.1f}GB)")
            else:
                print("   No models installed")
            return [model.get('name') for model in models]
        return []
    except Exception as e:
        print(f"❌ Error getting models: {e}")
        return []

def test_model_generation(model_name="llama3"):
    """Test model generation"""
    print(f"\n🧪 Testing {model_name} generation...")
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": "Hello! Please respond with just 'Hi there!' to test the connection.",
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("response", "").strip()
            print(f"✅ {model_name} response: {generated_text}")
            return True
        else:
            print(f"❌ Generation failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ {model_name} generation timed out (model might be loading)")
        return False
    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        return False

def install_recommended_models():
    """Install recommended models for RAG"""
    recommended_models = ["llama3", "mistral"]
    
    print("\n📦 Installing recommended models...")
    for model in recommended_models:
        print(f"Installing {model}...")
        try:
            result = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            if result.returncode == 0:
                print(f"✅ Successfully installed {model}")
            else:
                print(f"❌ Failed to install {model}: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout installing {model} (it might still be downloading)")
        except Exception as e:
            print(f"❌ Error installing {model}: {e}")

def main():
    print("🔍 Ollama Health Check\n" + "="*30)
    
    if not check_ollama_service():
        print("\n💡 To start Ollama:")
        print("   1. Open a terminal and run: ollama serve")
        print("   2. Keep that terminal open")
        print("   3. Run this script again")
        return False
    
    # Get available models
    models = get_available_models()
    
    # Install models if none available
    if not models:
        print("\n🤔 No models found. Installing recommended models...")
        install_recommended_models()
        time.sleep(2)
        models = get_available_models()
    
    # Test a model
    if models:
        test_model = models[0]
        if "llama3" in [m.split(':')[0] for m in models]:
            test_model = next(m for m in models if m.startswith("llama3"))
        elif "mistral" in [m.split(':')[0] for m in models]:
            test_model = next(m for m in models if m.startswith("mistral"))
        
        success = test_model_generation(test_model)
        
        if success:
            print("\n🎉 Ollama is working correctly!")
            print(f"   You can now start the RAG bot with model: {test_model}")
        else:
            print("\n⚠️  Ollama is running but model generation failed")
            print("   The model might still be loading. Try again in a moment.")
    else:
        print("\n❌ No models available for testing")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
