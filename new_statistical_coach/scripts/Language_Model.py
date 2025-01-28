from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

def get_llama_feedback(
    user_content, 
    system_content="You are a helpful fitness coach providing precise, constructive feedback.",
    model="llama-3.1-8b-instant", 
    temperature=0, 
    max_completion_tokens=1024, 
    top_p=1
):
    """
    Generate feedback using Groq's Llama API.
    
    Args:
        user_content (str): The main content/prompt to send to the model
        system_content (str, optional): System message to set model context. Defaults to fitness coach prompt.
        model (str, optional): Model to use. Defaults to "llama-3.1-8b-instant".
        temperature (float, optional): Sampling temperature. Defaults to 1.
        max_completion_tokens (int, optional): Maximum tokens in response. Defaults to 1024.
        top_p (float, optional): Nucleus sampling parameter. Defaults to 1.
    
    Returns:
        str: Generated model feedback
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))  

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": system_content
            },
            {
                "role": "user", 
                "content": user_content
            }
        ],
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        top_p=top_p,
        stream=True,
        stop=None,
    )

    model_feedback = ""
    for chunk in completion:
        content = chunk.choices[0].delta.content or ""
        print(content, end="")
        model_feedback += content
    
    return model_feedback


import requests

def get_local_language_model_feedback(
    user_content, 
    system_content="You are a helpful fitness coach providing precise, constructive feedback.",
    model="local-model", 
    temperature=0, 
    max_tokens=1024, 
    top_p=1,
    base_url="http://localhost:1234/v1"
):
    """
    Generate feedback using LM Studio's local server.
    
    Args:
        user_content (str): The main content/prompt to send to the model
        system_content (str, optional): System message to set model context. 
        model (str, optional): Model identifier. Defaults to "local-model".
        temperature (float, optional): Sampling temperature. Defaults to 1.
        max_tokens (int, optional): Maximum tokens in response. Defaults to 1024.
        top_p (float, optional): Nucleus sampling parameter. Defaults to 1.
        base_url (str, optional): Base URL for LM Studio server. Defaults to local endpoint.
    
    Returns:
        str: Generated model feedback
    """
    try:
        # Prepare the payload for the API request
        payload = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "model": model
        }

        # Send POST request to LM Studio server
        response = requests.post(
            f"{base_url}/chat/completions", 
            json=payload
        )

        # Check if the request was successful
        response.raise_for_status()

        # Extract and return the model's response
        model_response = response.json()
        model_feedback = model_response['choices'][0]['message']['content'].strip()

        # Optional: Print the feedback as it's generated
        print(model_feedback)
        
        return model_feedback

    except requests.RequestException as e:
        print(f"Error communicating with LM Studio server: {e}")
        return ""

