
import openai

class GPT4Responder:
    def __init__(self, api_key=None):
        if api_key:
            openai.api_key = api_key
        else:
            raise ValueError("OpenAI API key must be provided.")

    def generate_response(self, prompt, context=[]):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages += context
        messages.append({"role": "user", "content": prompt})

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        return response['choices'][0]['message']['content']
