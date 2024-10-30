import argparse
import gradio as gr
from openai import OpenAI

# Argument parser setup
parser = argparse.ArgumentParser(
    description='Chatbot Interface with Customizable Parameters')
parser.add_argument('--model-url',
                    type=str,
                    default='http://localhost:8000/v1',
                    help='Model URL')
parser.add_argument('-m',
                    '--model',
                    type=str,
                    required=True,
                    help='Model name for the chatbot')
parser.add_argument('--temp',
                    type=float,
                    default=0.8,
                    help='Temperature for text generation')
parser.add_argument('--stop-token-ids',
                    type=str,
                    default='',
                    help='Comma-separated stop token IDs')
parser.add_argument("--host", type=str, default=None)
parser.add_argument("--port", type=int, default=8001)

# Parse the arguments
args = parser.parse_args()

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = args.model_url

# Create an OpenAI client to interact with the API server
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Function to log user inputs and responses
def log_interaction(user_input, response):
    with open("chat_log.txt", "a") as log_file:
        log_file.write(f"User: {user_input}\n")
        log_file.write(f"Response: {response}\n\n")

def predict(system_prompt, message, history):
    # Convert chat history to OpenAI format
    if history is None:
        history = []
        
    history_openai_format = [{"role": "system", "content": system_prompt}]
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": message})

    # Create a chat completion request and send it to the API server
    stream = client.chat.completions.create(
        model=args.model,  # Model name to use
        messages=history_openai_format,  # Chat history
        temperature=args.temp,  # Temperature for text generation
        stream=True,  # Stream response
        extra_body={
            'repetition_penalty': 1,
            'stop_token_ids': [
                int(id.strip()) for id in args.stop_token_ids.split(',')
                if id.strip()
            ] if args.stop_token_ids else []
        })

    # Read and return generated text from response stream
    partial_message = ""
    for chunk in stream:
        partial_message += (chunk.choices[0].delta.content or "")
        yield partial_message

    # Log the interaction
    log_interaction(message, partial_message)

# Create Gradio inputs for the system prompt and chat interface
system_prompt_input = gr.Textbox(lines=2, placeholder="Enter system prompt here...", label="System Prompt")
chat_message_input = gr.Textbox(lines=2, placeholder="Enter your message here...", label="Message")

# Create and launch a chat interface with Gradio
chat_interface = gr.Interface(fn=predict, inputs=[system_prompt_input, chat_message_input], outputs="text")
chat_interface.queue().launch(server_name=args.host,
                              server_port=args.port,
                              share=True)
