from mlutils.external_apis.config import MICROSOFT_TRANSLATE_URL, MICROSOFT_TRANSLATE_API_KEY, MICROSOFT_TRANSLATE_LOCATION
import requests, uuid
import logging
from typing import Tuple, Union, List, Dict, Optional

# -- setup logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
date_format = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s', datefmt=date_format)
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

HEADERS = {
    'Ocp-Apim-Subscription-Key': MICROSOFT_TRANSLATE_API_KEY,
    'Ocp-Apim-Subscription-Region': MICROSOFT_TRANSLATE_LOCATION,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}

CHARACTER_LIMITS = 50000

# impement autobatch_size based on limits

def batch_by_size(sizes: Dict[int, int], limit: int) -> List[Dict[str, Union[int, List[int]]]]:
    """Given a size mapping such {document_id: size_of_document}, batches documents such that the total size of a batch of documents does not exceed pre-specified limit

    :param sizes: mapping that gives document size for each document_id
    :param limit: size limit for each batch
    :return: [{'idx': [ids_for_batch], 'total_size': total_size_of_documents_in_batch}, ...]

    Example:
        >>> documents = ['Joe Smith is cool', 'Django', 'Hi']
        >>> sizes = {i: len(doc) for i, doc in enumerate(documents)}
        >>> limit = 10
        >>> batch_by_size(sizes, limit)
        [{'idx': [0], 'total_size': 17}, {'idx': [1, 2], 'total_size': 8}]
    """

    batched_items = []
    sizes_iter = iter(sizes)
    key = next(sizes_iter)
    while key is not None:
        if not batched_items:
            batched_items.append({
                'idx': [key],
                'total_size': sizes[key]
            })
        else:
            size = sizes[key]
            total_size = batched_items[-1]['total_size'] + size
            if total_size > limit:
                batched_items.append({
                    'idx': [key],
                    'total_size': size
                })
            else:
                batched_items[-1]['idx'].append(key)
                batched_items[-1]['total_size'] = total_size
        key = next(sizes_iter, None)
    
    return batched_items

def build_url(url: str, param_name: str, param_value: Union[int, float, str, List[Union[int, float, str]]]) -> str:
    """Updates a request URL by parameters depending on the parameter type

    :param url: request URL ending with "?"
    :param param_name: name of the parameter
    :param param_value: value of the parameter
    :return: Updated url: f"{url}&{param_name}={param_value}"
    """
    if isinstance(param_value, str):
        url = f'{url}&{param_name}={param_value}'
    elif isinstance(param_value, list):
        for param_value_ in param_value:
            url = f'{url}&{param_name}={param_value_}'
    return url

def is_response_valid(status_code: int) -> bool:
    """Check if API response is valid

    :param status_code: status code from a HTTP request
    :return: `True` if status code is success, `False` otherwise
    """
    if str(status_code).startswith('2'):
        return True
    else:
        return False

def standardise_request_output(resp: requests.Response) -> Tuple[int, dict]:
    """Standardises the output of an API response

    :param resp: response from an API call
    :return: (status_code, response.json() if success call response.text otherwise)
    """
    status_code = resp.status_code
    if is_response_valid(status_code):
        msg = resp.json()
    else:
        msg = resp.text
    return status_code, msg

def _aggregate_translation_by_language(request_outputs):
    aggregated_output = dict(
        status_code=request_outputs[0]['status_code'],
        msg=[dict(translations=[])]
    )
    for request_output in request_outputs:
        translation = request_output['msg'][0]['translations'][0]
        aggregated_output['msg'][0]['translations'].append(translation)
    
    return aggregated_output

def _translate_batch(url, texts, target_languages, translation_dict=dict()):

    # -- batch texts based on global character limit
    n_langs = len(target_languages)
    total_chars_per_text = {text_id: len(text)*n_langs for text_id, text in texts.items()}
    batched_requests = batch_by_size(total_chars_per_text, CHARACTER_LIMITS)

    LOGGER.debug(f'Split {len(texts)} texts into {len(batched_requests)} batches')
    for batched_request in batched_requests:
        total_chars = batched_request['total_size']
        batch_idx = batched_request['idx']
        batched_texts = [texts[idx] for idx in batch_idx]
        batch_idx_key = tuple(batch_idx)

        if total_chars > CHARACTER_LIMITS:
            if len(batched_texts) == 1:
                idx = batch_idx[0]
                LOGGER.debug(f'Text with idx={idx} is too large for {len(target_languages)} languages. Further batching by language...')
                batch_size = CHARACTER_LIMITS // len(texts[idx])
                if not batch_size:
                    LOGGER.error(f'Text with idx={idx} exceeds character limit ({total_chars} > {CHARACTER_LIMITS}). Cannot be translated using this engine')
                    raise Exception(f'Text with idx={idx} exceeds character limit ({total_chars} > {CHARACTER_LIMITS}). Cannot be translated using this engine')
                else:
                    batch_range = range(0, n_langs, batch_size)
                    n_batches = len(batch_range)
                    _request_outputs = []
                    for batch_id, start_lang_idx in enumerate(batch_range):
                        end_lang_idx = start_lang_idx + batch_size
                        target_languages_ = target_languages[start_lang_idx: end_lang_idx]
                        url_ = build_url(url, 'to', target_languages_)
                        batched_body = [{'text': batched_text} for batched_text in batched_texts]
                        LOGGER.debug(f'Translating batch {batch_id+1}/{n_batches} of text with idx={idx}. Target languages: {target_languages_}')
                        resp = requests.post(url_, headers=HEADERS, json=batched_body)
                        status_code, msg = standardise_request_output(resp)
                        _request_outputs.append(dict(status_code=status_code, msg=msg))
                    
                    
                    _request_outputs = cleanup_multiple_requests_output(_request_outputs, errors='raise')
                    aggregated_requests = _aggregate_translation_by_language(_request_outputs)
                    translation_dict[batch_idx_key] = aggregated_requests
            else:
                translation_dict = _translate_batch(url, batched_texts, target_languages, translation_dict)
        else:
            batched_body = [{'text': batched_text} for batched_text in batched_texts]
            url_ = build_url(url, 'to', target_languages)
            LOGGER.debug(f'Translating texts with idx={batch_idx_key}. Total characters: {total_chars}')
            resp = requests.post(url_, headers=HEADERS, json=batched_body)
            status_code, msg = standardise_request_output(resp)
            translation_dict[batch_idx_key] = dict(status_code=status_code, msg=msg)
        
    return translation_dict

def cleanup_multiple_requests_output(request_outputs, errors):
    unsuccess_status_codes = [not is_response_valid(request_output['status_code']) for request_output in request_outputs]
    if any(unsuccess_status_codes):
        if errors == 'raise':
            raise Exception('Found at least 1 unsuccessful request. Failing all calls.')
        elif errors == 'remove':
            request_outputs = [request_output for request_output, unsuccess_status_code in zip(request_outputs, unsuccess_status_codes) if not unsuccess_status_code]
    return request_outputs

def translate_text_v2(
    text: Union[str, List[str]],
    target_language: Union[str, List[str]],
    source_language: Optional[str] = None,
) -> Tuple[int, list]:  # TODO change the output type
    """Translates text(s) to target_language(s) using Microsoft translate API v3. The translation process automatically batches the data where possible to avoid hitting the APIs character limit.

    :param text: text to be translated. Either single or multiple (stored in a list)
    :param target_language: ISO format of target translation languages
    :param source_language: ISO format of source language. If not provided is inferred by the translator, defaults to None
    :return: for successful response, (status_code, [{"translations": [{"text": translated_text_1, "to": lang_1}, ...]}, ...]))        
    """
    url = f'{MICROSOFT_TRANSLATE_URL}/translate?api-version=3.0'
    if source_language:
        url = f'{url}&from={source_language}'

    if isinstance(target_language, str):
        target_language = [target_language]
    
    if isinstance(text, str):
        text = [text]

    LOGGER.info(f'Translating {len(text)} texts to {len(target_language)} languages')

    translation_dict = dict()
    texts = {i: text_ for i, text_ in enumerate(text)}

    translation_dict = _translate_batch(url, texts, target_language, translation_dict)
    # ensure no missing translations
    _ = cleanup_multiple_requests_output(translation_dict.values(), errors='raise')

    # flatten the outputs
    translated_texts = [None] * len(texts)
    for idx, translation_outputs in translation_dict.items():
        translations = translation_outputs['msg']
        for index, translation in zip(idx, translations):
            translated_texts[index] = translation
    
    status_code = translation_outputs['status_code']

    return status_code, translated_texts


def translate_text(text, target_language, source_language=None):
    url = f'{MICROSOFT_TRANSLATE_URL}/translate?api-version=3.0'

    if isinstance(target_language, str):
        url = f'{url}&to={target_language}'
    elif isinstance(target_language, list):
        for lang in target_language:
            url = f'{url}&to={lang}'

    if source_language:
        url = f'{url}&from={source_language}'

    if isinstance(text, str):
        body = [{'text': text}]
    elif isinstance(text, list):
        body = [{'text': text_} for text_ in text]

    resp = requests.post(url, headers=HEADERS, json=body)

    if str(resp.status_code).startswith('2'):
        return resp.status_code, resp.json()
    
    return resp.status_code, resp.text

if __name__ == '__main__':
    print(translate_text('test', 'de'))