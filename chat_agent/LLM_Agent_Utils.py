import ast
import _ast
import contextlib
import io
import os
import requests
import openai
import yaml
from transformers import pipeline
from diffusers import AutoPipelineForText2Image
import torch

base_dir = os.getcwd()
# Initialize the pipeline with the multimodal LLM model
if torch.cuda.is_available():
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")
else:
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float32, variant="fp32")
    pipe.to("cpu")


def generate_image(prompt: str):
    try:
        # Generate text based on the user's prompt
        image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        safe_filename = f"generated_file_{prompt}".replace(" ", "_").replace("/", "_") + ".png"

        file_path = os.path.join(
            base_dir,
            safe_filename
        )
        save_image(image, file_path)
        return {"image_path": file_path}  # It might be more useful to return the file path

    except Exception as e:
        return {"error": str(e)}

# Write a function to save an image to a location
def save_image(image, file_path: str):
    image.save(file_path)

# Write a function to read an image from a location based on a prompt
def read_image(file_path: str):
    # safe_filename = f"generated_file_{prompt}".replace(" ", "_").replace("/", "_") + ".png"
    # file_path = os.path.join(
    #     base_dir,
    #     safe_filename
    # )
    with open(file_path, 'rb') as file:
        return file.read()

from IPython.display import display
from PIL import Image
# Write a function to display an image
def display_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    display(image)

def parse_yaml(file_path: str) -> dict:
    """Parse a YAML string and return a dictionary."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config_file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'config.yaml'
)

with open(config_file_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
weather_api_key = config['weather_api_key']


def execute_python_code(code: str) -> str:
    """Execute Python code and return the output as a string."""
    # Create a string buffer to capture stdout
    buffer = io.StringIO()

    # Use contextlib to redirect stdout to the buffer
    with contextlib.redirect_stdout(buffer):
        try:
            # Parse the code to ensure it's a simple expression for safety
            parsed_code = ast.parse(code, mode='exec')
            if not all(isinstance(node, (_ast.Expr, _ast.Import, _ast.ImportFrom)) for node in parsed_code.body):
                return "Error: Unsupported operation. Only simple expressions and imports are allowed."

            # Execute the code
            exec(code)
        except Exception as e:
            return f"Error executing code: {str(e)}"

    # Get the contents of the buffer
    output = buffer.getvalue()

    return output if output else "No output produced or error occurred."


def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


def duckduckgo_search(query: str):
    """ Perform a search on DuckDuckGo and return the result """
    api_url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
       "pretty": 1
    }
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        results = response.json()
        return results
    except requests.RequestException as e:
        raise Exception(str(e))


def get_weather(city_name):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    # Get Weather api key from environment
    # weather_api_key = os.getenv['weather_api_key']
    complete_url = f"{base_url}appid={weather_api_key}&q={city_name}"
    response = requests.get(complete_url)
    weather_data = response.json()

    if weather_data['cod'] != "404":
        # Parsing the weather_data dict for information
        weather_description = weather_data['weather'][0]['description']
        temperature = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        print(f"Weather in {city_name}: {weather_description}")
        print(f"Temperature: {temperature}K, Humidity: {humidity}%")
    else:
        print("City not found.")
    return weather_data
