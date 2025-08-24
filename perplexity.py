import requests
import os

api_key = os.getenv('PERPLEXITY_API_KEY')

response = requests.post(
    'https://api.perplexity.ai/chat/completions',
    headers={
        'Authorization': 'Bearer {0}'.format(api_key),
        'Content-Type': 'application/json'
    },
    json={
        'model': 'sonar-pro',
        'messages': [
            {
                'role': 'user',
                'content': "What are the major AI developments and announcements from today across the tech industry?"
            }
        ]
    }
)

print(response.json())