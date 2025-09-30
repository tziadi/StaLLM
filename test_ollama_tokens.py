#!/usr/bin/env python3
"""
Test script to verify Ollama token counting is working properly.
Run this to debug token extraction from your Ollama models.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from StaLLM_llm import ChatModel, LLMConfig, debug_ollama_response

def test_ollama_token_counting():
    """Test token counting with different Ollama models."""
    
    # Test models (use the ones you have available)
    test_models = [
        "gpt-oss:20b",
        "gemma3:1b", 
        "phi3:mini",
        "deepseek-r1:14b",
        "deepseek-coder-v2:latest",
        "llama2:latest",
        "mistral:latest"
    ]
    
    # Test message
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes code for potential issues."},
        {"role": "user", "content": "Analyze this Python function for potential bugs:\n\ndef calculate_average(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total / len(numbers)"}
    ]
    
    print("🧪 Testing Ollama Token Counting")
    print("=" * 50)
    
    for model_name in test_models:
        print(f"\n🔍 Testing model: {model_name}")
        print("-" * 30)
        
        try:
            # Create ChatModel instance
            config = LLMConfig(
                provider="ollama",
                model=model_name,
                api_base="http://localhost:11434"
            )
            
            chat_model = ChatModel(config)
            
            # Make a test request
            print("📤 Sending test request...")
            text, meta = chat_model.chat(
                messages=test_messages,
                temperature=0.1,
                max_tokens=200,
                return_meta=True
            )
            
            # Display results
            print(f"✅ Response received ({len(text)} characters)")
            print(f"📊 Token metrics:")
            print(f"   • Prompt tokens: {meta.get('prompt_tokens', 0)}")
            print(f"   • Completion tokens: {meta.get('completion_tokens', 0)}")
            print(f"   • Total tokens: {meta.get('total_tokens', 0)}")
            print(f"   • Model: {meta.get('model', 'unknown')}")
            
            # Show first 100 characters of response
            response_preview = text[:100] + "..." if len(text) > 100 else text
            print(f"📝 Response preview: {response_preview}")
            
        except Exception as e:
            print(f"❌ Error testing {model_name}: {str(e)}")
            continue
    
    print("\n" + "=" * 50)
    print("🏁 Token counting test completed!")

def test_single_model_debug(model_name: str = "phi3:mini"):
    """Test a single model with detailed debugging."""
    
    print(f"🐛 Debug test for {model_name}")
    print("=" * 40)
    
    try:
        config = LLMConfig(
            provider="ollama",
            model=model_name,
            api_base="http://localhost:11434"
        )
        
        chat_model = ChatModel(config)
        
        # Simple test message
        test_messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        # Make request and capture raw response for debugging
        import ollama
        original_host = os.environ.get("OLLAMA_HOST")
        try:
            os.environ["OLLAMA_HOST"] = "http://localhost:11434"
            
            # Get raw response for debugging
            raw_response = ollama.chat(
                model=model_name,
                messages=test_messages,
                options={"temperature": 0.1}
            )
            
            print("🔍 Raw Ollama response structure:")
            debug_info = debug_ollama_response(raw_response)
            for key, value in debug_info.items():
                print(f"   • {key}: {value}")
            
            # Test our token extraction
            print("\n📊 Token extraction test:")
            text, meta = chat_model.chat(
                messages=test_messages,
                temperature=0.1,
                return_meta=True
            )
            
            print(f"   • Prompt tokens: {meta.get('prompt_tokens', 0)}")
            print(f"   • Completion tokens: {meta.get('completion_tokens', 0)}")
            print(f"   • Total tokens: {meta.get('total_tokens', 0)}")
            
        finally:
            if original_host is not None:
                os.environ["OLLAMA_HOST"] = original_host
            elif "OLLAMA_HOST" in os.environ:
                del os.environ["OLLAMA_HOST"]
                
    except Exception as e:
        print(f"❌ Debug test failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Ollama Token Counting Test")
    print("Choose test mode:")
    print("1. Test all models")
    print("2. Debug single model")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_ollama_token_counting()
    elif choice == "2":
        model = input("Enter model name (default: phi3:mini): ").strip() or "phi3:mini"
        test_single_model_debug(model)
    else:
        print("Invalid choice. Running full test...")
        test_ollama_token_counting()

