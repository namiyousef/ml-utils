import os

# MICROSOFT API
MICROSOFT_TRANSLATE_URL = 'https://api.cognitive.microsofttranslator.com'
MICROSOFT_TRANSLATE_API_KEY = os.environ.get('MICROSOFT_TRANSLATE_API_KEY', 'key')
MICROSOFT_TRANSLATE_LOCATION = os.environ.get('MICROSOFT_TRANSLATE_LOCATION', 'uksouth')