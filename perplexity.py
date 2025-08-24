import requests
import os

class PerplexityClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('PERPLEXITY_API_KEY')
        self.api_url = 'https://api.perplexity.ai/chat/completions'

    def send_prompt(self, prompt, model='sonar-pro', json_schema=None, system_prompt=None):

        messages = []
        
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })
            
        messages.append({
            'role': 'user',
            'content': prompt
        })
        
        payload = {
            'model': model,
            'messages': messages
        }

        if json_schema is not None :
            payload['response_format'] = {
                                            "type": "json_schema",
                                                "json_schema": {
                                                    "schema": json_schema
                                                }
                                        }
                                          

        response = requests.post(
            self.api_url,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            },
            json=payload
        )
        return response.json()
