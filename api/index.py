import os, re, io, json          # ← json тоже нужен
from flask import Flask, request, jsonify   # ← ЭТО добавь
import re
import replicate
import requests
from PIL import Image
import io

# --- ЭТО ВАЖНАЯ ЧАСТЬ ---
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))


app = Flask(__name__, template_folder=template_dir)

try:
    replicate_api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not replicate_api_token:
        raise ValueError("FATAL: REPLICATE_API_TOKEN not found in environment variables.")
    
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("FATAL: GOOGLE_API_KEY not found in environment variables.")
    
    print(f"✓ Replicate token found.")
    print(f"✓ Google API key found.")
    
    replicate_client = replicate.Client(api_token=replicate_api_token)
    
    genai.configure(api_key=google_api_key)
    
    print("✓ API keys configured successfully")

except Exception as e:
    print(f"FATAL: API key configuration failed: {e}")
    exit(1)

def create_ascii_art(image_bytes, width=70):
    ascii_chars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w_percent = (width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))
        image = image.resize((width, int(h_size * 0.5)))
        
        ascii_image = ""
        for y in range(image.height):
            for x in range(image.width):
                r, g, b = image.getpixel((x, y))
                gray = int(0.21 * r + 0.72 * g + 0.07 * b)
                index = int((gray / 255) * (len(ascii_chars) - 1))
                ascii_image += ascii_chars[index]
            ascii_image += "\n"
        return ascii_image
    except Exception as e:
        print(f"ASCII conversion failed: {e}")
        return "[ ASCII CONVERSION FAILED ]"


SYSTEM_PROMPT = """
You are a creative director for a text-based adventure game. Your output MUST be a single, valid JSON object.
The JSON must have: "genre", "image_prompt", "story_scenes", "choices", "is_final_scene".

- "genre": A single word for the story's genre (e.g., Sci-Fi, Fantasy, Noir, Horror).
- "image_prompt": A prompt for a visually striking, high-contrast, cartoon-style image with a solid black background. IMPORTANT: Provide a prompt periodically throughout the story (e.g., every 1-2 story segments) to keep it visually engaging. Do not provide a prompt for every single scene. If no image is needed, this MUST be an empty string "".
- "story_scenes": An array of 1 to 3 strings. Each string is a paragraph of the story. The player will press Enter to advance through them one by one.
- "choices": An array of 3 player actions. This array applies ONLY to the VERY LAST scene in the `story_scenes` array. If this segment of the story does not end with a choice, this MUST be an empty array `[]`.
- "is_final_scene": A boolean (true/false). Set to `true` ONLY if this is the absolute end of the entire story.

IMPORTANT: Adhere to the requested story length (short, medium, or long) passed by the user. A 'short' story should have around 2-3 choice moments, 'medium' 4-5, and 'long' 6 or more.
Your goal is to create a flowing narrative in chunks. You are responsible for bringing the story to a natural conclusion.
"""
model = genai.GenerativeModel("gemini-2.5-flash")


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    history = data.get('history', [])
    user_input = data.get('userInput', '')

    try:
        print("--- Gemini Stage: Requesting scene data... ---")
        full_context = [
            {'role': 'user', 'parts': [SYSTEM_PROMPT]},
            {'role': 'model', 'parts': ["OK. I will only respond with a valid JSON object containing 'genre', 'image_prompt', 'story_scenes', 'choices', and 'is_final_scene'."]},
            *history,
            {'role': 'user', 'parts': [user_input]}
        ]
        
        response = model.generate_content(full_context)
        
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON found in Gemini response")
        
        gemini_text = json_match.group(0)
        scene_data = json.loads(gemini_text)
        print("--- Gemini Stage: Scene data received. ---")

        ascii_art = None
        image_prompt = scene_data.get("image_prompt") 
        
        if image_prompt: 
            print(f"--- Replicate Stage: Requesting image for prompt: '{image_prompt}' ---")
            try:
                output = replicate_client.run(
                    "black-forest-labs/flux-schnell",
                    input={"prompt": image_prompt}
                )
            except Exception:
                print("--- Flux Schnell failed, trying SDXL ---")
                try:
                    output = replicate_client.run(
                        "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                        input={"prompt": image_prompt}
                    )
                except Exception as e2:
                     print(f"--- SDXL also failed: {e2}. No image will be generated. ---")
                     output = None

            if output:
                image_url = output[0] if isinstance(output, list) and output else output
                print(f"--- Replicate Stage: Image URL received: {image_url} ---")
                
                print("--- Downloading and converting image... ---")
                image_response = requests.get(image_url, timeout=30)
                image_response.raise_for_status()
                ascii_art = create_ascii_art(image_response.content)
                print("--- ASCII art created successfully ---")
        else:
            print("--- Image not requested for this scene. ---")


        final_data = {
            "genre": scene_data.get("genre", "Unknown"),
            "art": ascii_art,
            "story_scenes": scene_data.get("story_scenes", ["The system is silent."]),
            "choices": scene_data.get("choices", []),
            "is_final_scene": scene_data.get("is_final_scene", False)
        }
        
        final_history = history + [
            {'role': 'user', 'parts': [user_input]},
            {'role': 'model', 'parts': [response.text]}
        ]
        
        return jsonify({
            'status': 'success',
            'data': final_data,
            'history': final_history
        })
        
    except replicate.exceptions.ReplicateError as e:
        print(f"REPLICATE ERROR: {e}")
        return jsonify({'status': 'error', 'message': f'Replicate API error: {str(e)}.'}), 500
    except requests.exceptions.RequestException as e:
        print(f"HTTP ERROR: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to download image: {str(e)}'}), 500
    except json.JSONDecodeError as e:
        print(f"JSON ERROR: {e}\nGemini response was: {response.text}")
        return jsonify({'status': 'error', 'message': f'Invalid JSON from Gemini: {str(e)}'}), 500
    except Exception as e:
        print(f"FATAL ERROR in /generate: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Unexpected error: {str(e)}'}), 500


@app.route('/test-replicate', methods=['GET'])
def test_replicate():
    """Тестирование Replicate API"""
    try:
        print("--- Testing Replicate API ---")
        
        models_to_test = [
            ("Flux Schnell", "black-forest-labs/flux-schnell"),
            ("SDXL", "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"),
            ("Stable Diffusion", "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf"),
            ("Kandinsky", "ai-forever/kandinsky-2.2:ad9d7879fbffa2874e1d909d1d37d9bc682889cc65b31f7bb00d2362619f194a")
        ]
        
        results = {}
        working_model = None
        
        for model_name, model_path in models_to_test:
            try:
                print(f"--- Testing {model_name}: {model_path} ---")
                
                output = replicate_client.run(
                    model_path,
                    input={"prompt": "test image"}
                )
                
                results[model_name] = {
                    "status": "success", 
                    "model": model_path,
                    "output": str(output)
                }
                working_model = model_path
                print(f"✓ {model_name} works!")
                break  
                
            except Exception as e:
                results[model_name] = {
                    "status": "error", 
                    "model": model_path,
                    "error": str(e)
                }
                print(f"✗ {model_name} failed: {e}")
                continue
        
        return jsonify({
            'status': 'test_complete',
            'working_model': working_model,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'type': type(e).__name__
        }), 500

def test_manual_replicate_request(token):
    """Тестируем Replicate API вручную через HTTP запрос"""
    import requests
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    data = {
        "version": "3d8650e9965a34f47535136453a242f388836551b3272d491c6e11894982a88d",
        "input": {"prompt": "test image"}
    }
    
    response = requests.post('https://api.replicate.com/v1/predictions', 
                           headers=headers, json=data, timeout=30)
    
    if response.status_code == 401:
        raise Exception(f"Manual request also failed with 401. Token is definitely invalid. Response: {response.text}")
    
    response.raise_for_status()
    return response.json()

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка работоспособности API"""
    try:
        replicate_token = os.environ.get("REPLICATE_API_TOKEN")
        google_key = os.environ.get("GOOGLE_API_KEY")
        
        return jsonify({
            'status': 'healthy',
            'replicate_configured': bool(replicate_token),
            'google_configured': bool(google_key)
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

