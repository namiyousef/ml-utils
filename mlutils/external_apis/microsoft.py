from mlutils.external_apis.config import MICROSOFT_TRANSLATE_URL, MICROSOFT_TRANSLATE_API_KEY, MICROSOFT_TRANSLATE_LOCATION
import requests, uuid
import logging
from typing import Tuple, Union, List, Dict, Optional
from mlutils.external_apis.utils import is_request_valid, add_array_api_parameters

# -- setup logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
date_format = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s', datefmt=date_format)
ch.setFormatter(formatter)
LOGGER.addHandler(ch)
# -- define global variables
HEADERS = {
    'Ocp-Apim-Subscription-Key': MICROSOFT_TRANSLATE_API_KEY,
    'Ocp-Apim-Subscription-Region': MICROSOFT_TRANSLATE_LOCATION,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}

CHARACTER_LIMITS = 50000

def batch_by_size_min_buckets(sizes: Dict[Union[int, str], int], limit: int, sort_docs: bool = True, on_encounter_doc_exceed_limit: Optional[str] = None) -> List[Dict[str, Union[int, List[int]]]]:
    """Given dictionary of documents and their sizes {doc_id: doc_size}, batch documents such that the total size of each batch <= limit. Algorithm designed to decrease number of batches, but does not guarantee that it will be an optimal fit

    :param sizes: mapping that gives document size for each document_id, {doc_1: 10, doc_2: 20, ...}
    :param limit: size limit for each batch
    :sort_doc: if True sorts `sizes` in descending order
    :on_encounter_doc_exceed_limit: strategy to deal with documents that exceed max size. If `warn` then issues warning, if `raise` raises error, else does nothing
    :return: [{'idx': [ids_for_batch], 'total_size': total_size_of_documents_in_batch}, ...]

    Example:
        >>> documents = ['Joe Smith is cool', 'Django', 'Hi']
        >>> sizes = {i: len(doc) for i, doc in enumerate(documents)}
        >>> limit = 10
        >>> batch_by_size(sizes, limit)
        [{'idx': [0], 'total_size': 17}, {'idx': [1, 2], 'total_size': 8}]
    """
    if sort_docs:
        sizes = {key: size for key, size in sorted(sizes.items(), key=lambda x: x[1], reverse=True)}

    
    batched_items = []
    sizes_iter = iter(sizes)
    key = next(sizes_iter)  # doc_id

    # -- helpers
    def _add_doc(key):
        batched_items.append({
            'idx': [key],
            'total_size': sizes[key]
        })

    def _append_doc_to_batch(batch_id, key):
        batched_items[batch_id]['idx'].append(key)
        batched_items[batch_id]['total_size'] += sizes[key]
    
    while key is not None:

        # initial condition
        if not batched_items:
            _add_doc(key)
        else:
            size = sizes[key]

            if size > limit:
                msg = f'Document {key} exceeds max limit size: {size}>{limit}'
                if on_encounter_doc_exceed_limit == 'raise':
                    raise Exception(msg)
                elif on_encounter_doc_exceed_limit == 'warn':
                    LOGGER.warning(msg)
                
                _add_doc(key)
            else:
                # find the batch that fits the current doc best
                batch_id = -1
                total_capacity = limit - size  # how much we can still fit
                min_capacity = total_capacity
                for i, batched_item in enumerate(batched_items):
                    total_size = batched_item['total_size']
                    remaining_capacity =  total_capacity - total_size  # we want to minimise this

                    # current batch too large for doc, go to next batch
                    if remaining_capacity < 0:
                        continue
                    # current batch is a better fit for doc, save batch_id
                    elif remaining_capacity < min_capacity:
                        min_capacity = remaining_capacity
                        batch_id = i

                    # if perfect fit, break loop
                    if remaining_capacity == 0:
                        break
                

                if batch_id == -1:
                    _add_doc(key)
                else:
                    _append_doc_to_batch(batch_id, key)

        key = next(sizes_iter, None) 
    return batched_items

# impement autobatch_size based on limits

def batch_by_size(sizes: Dict[int, int], limit: int, sort_docs: bool = False) -> List[Dict[str, Union[int, List[int]]]]:
    """Given a size mapping such {document_id: size_of_document}, batches documents such that the total size of a batch of documents does not exceed pre-specified limit

    :param sizes: mapping that gives document size for each document_id
    :param limit: size limit for each batch
    :sort_doc: if True sorts `sizes` in descending order
    :return: [{'idx': [ids_for_batch], 'total_size': total_size_of_documents_in_batch}, ...]

    Example:
        >>> documents = ['Joe Smith is cool', 'Django', 'Hi']
        >>> sizes = {i: len(doc) for i, doc in enumerate(documents)}
        >>> limit = 10
        >>> batch_by_size(sizes, limit)
        [{'idx': [0], 'total_size': 17}, {'idx': [1, 2], 'total_size': 8}]
    """
    if sort_docs:
        sizes = {key: size for key, size in sorted(sizes.items(), key=lambda x: x[1], reverse=True)}

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
            if size > limit:
                LOGGER.warning(f'Document {key} exceeds max limit size: {size}>{limit}')
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

def standardise_request_output(resp: requests.Response) -> Tuple[int, dict]:
    """Standardises the output of an API response

    :param resp: response from an API call
    :return: (status_code, response.json() if success call response.text otherwise)
    """
    status_code = resp.status_code
    if is_request_valid(status_code):
        msg = resp.json()
    else:
        msg = resp.text
    return msg, status_code

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
                        msg, status_code = standardise_request_output(resp)
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
            msg, status_code = standardise_request_output(resp)
            translation_dict[batch_idx_key] = dict(status_code=status_code, msg=msg)
        
    return translation_dict

def cleanup_multiple_requests_output(request_outputs, errors):
    unsuccess_status_codes = [not is_request_valid(request_output['status_code']) for request_output in request_outputs]
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
    api_version: str = '3.0'
) -> Tuple[int, list]:  # TODO change the output type
    """Translates text(s) to target_language(s) using Microsoft translate API v3. The translation process automatically batches the data where possible to avoid hitting the APIs character limit.

    :param text: text to be translated. Either single or multiple (stored in a list)
    :param target_language: ISO format of target translation languages
    :param source_language: ISO format of source language. If not provided is inferred by the translator, defaults to None
    :param api_version: api version to use, defaults to "3.0"
    :return: for successful response, (status_code, [{"translations": [{"text": translated_text_1, "to": lang_1}, ...]}, ...]))        
    """
    url = f'{MICROSOFT_TRANSLATE_URL}/translate?api-version={api_version}'
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

    return translated_texts, status_code


def translate_text(
    text: Union[str, list],
    target_language: Union[str, list],
    source_language: Optional[str] = None,
    api_version: str = '3.0') -> tuple:
    """translates txt using the microsoft translate API

    :param text: text to be translated. Either single or multiple (stored in a list)
    :param target_language: ISO format of target translation languages
    :param source_language: ISO format of source language. If not provided is inferred by the translator, defaults to None
    :param api_version: api version to use, defaults to "3.0"
    :return: for successful response, (status_code, [{"translations": [{"text": translated_text_1, "to": lang_1}, ...]}, ...]))        
    """

    url = f'{MICROSOFT_TRANSLATE_URL}/translate?api-version={api_version}'

    if isinstance(target_language, str):
        target_language = [target_language]
    
    url = add_array_api_parameters(url, param_name='to', param_values=target_language)

    if source_language:
        url = f'{url}&from={source_language}'

    if isinstance(text, str):
        text = [text]
    
    body = [{'text': text_} for text_ in text]

    LOGGER.info(f'Translating {len(text)} texts to {len(target_language)} languages')
    resp = requests.post(url, headers=HEADERS, json=body)
    status_code = resp.status_code
    print(resp.reason, resp.text)
    print(resp.headers.get('Retry-After'))
    if is_request_valid(status_code):
        return resp.json(), status_code

    return resp.text, status_code

def translate_text_v3(
    text: Union[str, list],
    target_language: Union[str, list],
    source_language: Optional[str] = None,
    api_version: str = '3.0', raise_error_on_translation_failure=True, include_partials_in_output=False) -> tuple:
    """translates txt using the microsoft translate API

    :param text: text to be translated. Either single or multiple (stored in a list)
    :param target_language: ISO format of target translation languages
    :param source_language: ISO format of source language. If not provided is inferred by the translator, defaults to None
    :param api_version: api version to use, defaults to "3.0"
    :param raise_error_on_translation_failure: if `True`, raises errors on translation failure. If `False` ignores failed translations in output
    :pram include_partials_in_output: if `True` includes partially translated texts in output, otherwise ignores them
    :return: for successful response, (status_code, [{"translations": [{"text": translated_text_1, "to": lang_1}, ...]}, ...]))        
    """

    url = f'{MICROSOFT_TRANSLATE_URL}/translate?api-version={api_version}'

    if isinstance(target_language, str):
        target_language = [target_language]

    if source_language:
        url = f'{url}&from={source_language}'

    if isinstance(text, str):
        text = [text]
    
    n_target_langs = len(target_language)
    
    # get maximum size of each document based on target languages
    sizes_dict = {i: len(text_)*n_target_langs for i, text_ in enumerate(text)}
    batched_texts = batch_by_size_min_buckets(sizes_dict, CHARACTER_LIMITS)

    # for each batch, translate the texts. If successful append to list
    translation_outputs = []

    for batch in batched_texts:
        translation_output = []
        doc_ids = batch['idx']
        total_size = batch['total_size']

        if total_size > CHARACTER_LIMITS:
            doc_size = sizes_dict[doc_ids[0]]
            batch_size = CHARACTER_LIMITS // doc_size
            
            # case when a single doc is too large to be translated
            if not batch_size:
                # TODO the ID of this should be saved!
                msg = f'Text `{doc_ids[0]}` too large to be translated'
                if raise_error_on_translation_failure:
                    raise Exception(msg)
                LOGGER.error(msg)
            
            # case when single doc too big to be translated to all language, but small enough that it can be translated to some languages
            else:
                _translation_output = dict(translations=[])
                batch_range = range(0, n_target_langs, batch_size)
                n_batches = len(batch_range)
                
                _translation_failed = False
                # batch by target languages
                for batch_id, start_lang_idx in enumerate(batch_range):
                    end_lang_idx = start_lang_idx + batch_size
                    target_languages_ = target_language[start_lang_idx: end_lang_idx]
                    
                    # rebuild the url for subset of langs
                    url_ = add_array_api_parameters(url, param_name='to', param_values=target_languages_)
                    body = [{'text': text[doc_ids[0]]}]
                    
                    LOGGER.info(f'Translating batch {batch_id+1}/{n_batches} of text with idx={doc_ids[0]}. Target languages: {target_languages_}')
                    resp = requests.post(url_, headers=HEADERS, json=body)
                    status_code = resp.status_code
                    if not is_request_valid(status_code):
                        msg = f'Partial translation of text `{doc_ids[0]}` to languages {target_languages_} failed.'
                        if raise_error_on_translation_failure:
                            raise Exception(msg)
                        LOGGER.error(msg)
                        _translation_failed = True
                        if not include_partials_in_output:
                            break
                    
                    partial_translation_output = resp.json()
                    # concatenate outputs in correct format
                    _translation_output['translations'] += partial_translation_output['translations']
                
                if not _translation_failed or include_partials_in_output:
                    translation_output.append(_translation_output)
                
        else:
            # -- code as before, except translation_output now part of else
            batch_texts = [text[doc_id] for doc_id in doc_ids]
            body = [{'text': text_} for text_ in batch_texts]
            LOGGER.info(f'Translating {len(text)} texts to {len(target_language)} languages')
            # rebuild url for all languages
            url_ = add_array_api_parameters(
                url,
                param_name='to',
                param_values=target_language
            )
            resp = requests.post(url_, headers=HEADERS, json=body)
            status_code = resp.status_code
            if not is_request_valid(status_code):
                msg = f'Translation failed for texts {doc_ids}. Reason: {resp.text}'
                if raise_error_on_translation_failure:
                    raise Exception(msg)
                LOGGER.error(msg)
            else:
                translation_output += resp.json()

        translation_outputs += translation_output
    
    return translation_outputs, status_code


MAX_CHARACTER_LIMITS_PER_HOUR = 2000000

class MicrosoftTranslator:
    """Class for translating text using the Microsoft Translate API

    :param api_version: api version to use, defaults to "3.0"
    :param ignore_on_translation_failure: if `False`, returns failed translations with error and status code. If `False` ignores failed translations in output, defaults to False
    :param include_partials_in_output: if `True` includes partially translated texts in output, otherwise ignores them, defaults to False

    """
    def __init__(
        self,
        api_version: str = '3.0',
        ignore_on_translation_failure: bool = False,
        include_partials_in_output: bool = False
    ):
        
        base_url = f'{MICROSOFT_TRANSLATE_URL}/translate?api-version={api_version}'

        self.base_url = base_url
        self.api_version = api_version
        self.ignore_on_translation_failure = ignore_on_translation_failure
        self.include_partials_in_output = include_partials_in_output
    
    def translate_text(
        self,
        texts: Union[str, list],
        target_languages: Union[str, list],
        source_language: Optional[str] = None
    ) -> tuple:
        """translates txt using the microsoft translate API

        :param texts: text(s) to be translated. Can either be a single text (str) or multiple (list)
        :param target_languages: ISO format of target translation language(s). Can be single lang (str) or multiple (list)
        :param source_language: ISO format of source language. If not provided is inferred by the translator, defaults to None
        :return: for successful response, (status_code, [{"translations": [{"text": translated_text_1, "to": lang_1}, ...]}, ...]))         
        """

        # -- create storage for translation failures and flag for failed translation
        self._set_request_default

        # add source language to url
        if source_language:
            base_url = f'{self.base_url}&from={source_language}'
        else:
            base_url = self.base_url

        # standardise target_languages and texts types
        if isinstance(target_languages, str):
            target_languages = [target_languages]
        
        if isinstance(texts, str):
            texts = [texts]



        # batch texts for translation, based on doc_size = len(doc)*n_target_langs
        n_target_langs = len(target_languages)
        sizes_dict = {i: len(text)*n_target_langs for i, text in enumerate(texts)}

        profile_texts = self._profile_texts(sizes_dict)
        if profile_texts:
            return profile_texts

        batched_texts = batch_by_size_min_buckets(sizes_dict, CHARACTER_LIMITS, sort_docs=True)

        translation_outputs = []  # variable to store all translation outputs

        for batch in batched_texts:
            batch_translation_output = []  # variable to store translation output for single batch
            doc_ids = batch['idx']
            total_size = batch['total_size']
            texts_in_batch = [texts[doc_id] for doc_id in doc_ids]
            # -- case when batch exceeds character limits
            if total_size > CHARACTER_LIMITS:
                assert len(doc_ids) == 1, 'Critical error: batching function is generating batches that exceed max limit with more than 1 text. Revisit the function and fix this.'

                doc_id = doc_ids[0]
                doc_size = sizes_dict[doc_id] // n_target_langs
                batch_size = CHARACTER_LIMITS // doc_size
                
                # -- case when a single doc is too large to be translated
                if not batch_size:
                    process_error = self._process_error(f'Text idx={doc_id} too large to be translated', 'Max character limit for request', 400, doc_ids, target_languages)
                    if process_error:
                        return process_error
                
                # -- case when single doc too big to be translated to all language, but small enough that it can be translated to some languages
                else:
                    _translation_output = dict()  # variable to store translations for single text but different language batches
                    _translation_failed = False  # variable to track if translation is partial

                    batch_range = range(0, n_target_langs, batch_size)
                    n_batches = len(batch_range)
                    
                    # batch by target languages
                    for batch_id, start_lang_idx in enumerate(batch_range):
                        end_lang_idx = start_lang_idx + batch_size
                        target_languages_ = target_languages[start_lang_idx: end_lang_idx]
                        
                        response_output, status_code = self._post_request(
                            f'Translating batch {batch_id+1}/{n_batches} of text with idx={doc_id}. Target languages: {target_languages_}',
                            base_url, target_languages_, texts_in_batch
                        )
                        if not is_request_valid(status_code):
                            process_error = self._process_error(
                                f'Partial translation of text idx={doc_id} to languages {target_languages_} failed. Reason: {response_output}',
                                response_output, status_code, doc_ids, target_languages_)
                            if process_error:
                                return process_error
                            
                            # failure indicates translation is partial. Break loop if we don't case about partials in output
                            _translation_failed = True
                            if not self.include_partials_in_output:
                                break
                        else:
                            self._update_partial_translation(response_output, _translation_output, source_language)
                    
                    if not _translation_failed or self.include_partials_in_output:
                        if _translation_output:
                            batch_translation_output.append(_translation_output)

            # -- case when batch does not exceed character limits
            else:
                response_output, status_code = self._post_request(
                    f'Translating {len(texts)} texts to {len(target_languages)} languages',
                    base_url, target_languages, texts_in_batch
                )

                if not is_request_valid(status_code):
                    process_error = self._process_error(
                        f'Translation failed for texts {doc_ids}. Reason: {response_output}',
                        response_output, status_code, doc_ids, target_languages
                    )
                    if process_error:
                        return process_error
                else:
                    batch_translation_output += response_output

            translation_outputs += batch_translation_output
        
        if self.no_failures:
            status_code = 200
        # case when all translations failed, so return translation errors instead
        elif not translation_outputs:
            translation_outputs = self.translation_errors
            status_code = 400
        # case when translations partially failed, modify status code
        else:
            status_code = 206

        return translation_outputs, status_code
    
    @property
    def _set_request_default(self):
        """Function for resetting translation errors and no_failures flag
        """

        self.translation_errors = {}
        self.no_failures = True

    @property
    def _set_no_failures_to_false(self):
        """Function to explicitly set no failures to False
        """
        self.no_failures = False
    
    def _update_partial_translation(self, response_output, partial_translation_output, source_language):

        # concatenate outputs in correct format
        if 'translations' not in partial_translation_output:
            partial_translation_output['translations'] = response_output['translations']
        else:
            partial_translation_output['translations'] += response_output['translations']

        if not source_language:
            if 'detectedLanguage' not in partial_translation_output:
                partial_translation_output['detectedLanguage'] = response_output['detectedLanguage']
            else:
                partial_translation_output['detectedLanguage'] += response_output['detectedLanguage']
        
        return partial_translation_output

    def _update_translation_errors(self, response_text: str, status_code: int, doc_ids: list, target_languages: list):
        """Add failed translation to errors dictionary

        :param response_text: response text from failed request (status_code not beginning with 2)
        :param status_code: status code from failed request
        :param doc_ids: documents that were to be translated
        :param target_languages: target languages used in request
        """
        doc_ids = tuple(doc_ids)
        if doc_ids not in self.translation_errors:
            self.translation_errors[doc_ids] = dict(
                reason=response_text,
                status_code=status_code,
                target_languages=target_languages
            )
        else:
            self.translation_errors[doc_ids]['target_languages'] += target_languages
            self.translation_errors[doc_ids]['status_code'] = status_code
            self.translation_errors[doc_ids]['reason'] = response_text
        
    def _process_error(self, msg: str, response_text: str, status_code: int, doc_ids: list, target_languages: list):
        """Processes failed request based on `ignore_on_translation_failure` strategy

        :param msg: message to return or log depending on `ignore_on_translation_failure` strategy
        :param response_text: response text from failed request (status_code not beginning with 2)
        :param status_code: status code from failed request
        :param doc_ids: documents that were to be translated
        :param target_languages: target languages used in request
        """

        self._set_no_failures_to_false
        
        self._update_translation_errors(response_text, status_code, doc_ids, target_languages)

        if not self.ignore_on_translation_failure:
            return response_text, status_code

        LOGGER.error(msg)

    def _profile_texts(self, sizes: Dict[Union[int, str], int]):
        """Profiles texts to see if the request can be translated

        :param sizes: size mapping for each document, {doc_id_1: size, ...}
        """
        num_texts = len(sizes)
        total_request_size = sum(sizes.values())
        if total_request_size > MAX_CHARACTER_LIMITS_PER_HOUR:
            return 'Your texts exceed max character limits per hour', 400
        
        LOGGER.info(f'Detected `{num_texts}` texts with total request size `{total_request_size}`')

    def _post_request(self, msg: str, base_url: str, target_languages: list, texts: List[str]) -> Tuple[Union[dict, str], int]:
        """Internal function to post requests to microsoft API

        :param msg: message to log
        :param base_url: base url for making the request
        :param target_languages: list of target languages to translate text to
        :param texts: texts to translate
        :return: for successful response, (status_code, [{"translations": [{"text": translated_text_1, "to": lang_1}, ...]}, ...]))
        """

        url = add_array_api_parameters(base_url, param_name='to', param_values=target_languages)
        LOGGER.info(msg)
        body = [{'text': text} for text in texts]
        resp = requests.post(url, headers=HEADERS, json=body)
        status_code = resp.status_code
        if is_request_valid(status_code):
            return resp.json(), status_code

        return resp.text, status_code

# TODO need to ave a check on the total number of characters, if exceeds 2 million then depending on fail translation raise error or not
# you cannot really guarantee that it will not cause a 429 error...
if __name__ == '__main__':
    translator = MicrosoftTranslator(ignore_on_translation_failure=True, include_partials_in_output=True)
    outputs, status_code = translator.translate_text(['text'*5000, 'text', 'text'*10000, 'text'*100000], ['de', 'en'])
    print(outputs)
    print(status_code)
    '''import matplotlib.pyplot as plt
    import numpy as np
    import time

    def get_stats(batched_items):
        num_batches = len(batched_items)
        total_sizes = np.array([batch['total_size'] for batch in batched_items])
        mean = total_sizes.mean()
        std = total_sizes.std()
        return num_batches, mean, std
        
    limit = 50000
    sizes = 60000
    num_docs = 1000
    num_iter = 1000
    time_taken_sorted = np.zeros((num_docs, num_iter))
    time_taken = np.zeros((num_docs, num_iter))

    num_batches_sorted = np.zeros((num_docs, num_iter))
    num_batches = np.zeros((num_docs, num_iter))

    for i in range(1, num_docs+1):
        for j in range(num_iter):
            document_sizes = np.random.choice(np.arange(1, sizes), size=i)
            document_sizes = {i: size for i, size in enumerate(document_sizes)}
            # sorted
            start = time.time()
            batched_items_sorted = batch_by_size_min_buckets(document_sizes, limit, sort_docs=True)
            end = time.time()
            n_batches, mean, std = get_stats(batched_items_sorted)
            time_taken_sorted[i-1, j] = end-start
            num_batches_sorted[i-1, j] = n_batches
            # unsorted
            start = time.time()
            batched_items = batch_by_size_min_buckets(document_sizes, limit, sort_docs=False)
            end = time.time()
            n_batches, mean, std = get_stats(batched_items_sorted)

            time_taken[i-1, j] = end-start
            num_batches[i-1, j] = n_batches
    plt.figure()
    plt.plot(range(1, num_docs+1), time_taken_sorted.mean(axis=1), '.')
    plt.fill_between(range(1, num_docs+1), time_taken_sorted.mean(axis=1)- time_taken_sorted.std(axis=1), time_taken_sorted.mean(axis=1)+time_taken_sorted.std(axis=1), alpha=0.1)
    plt.plot(range(1, num_docs+1), time_taken.mean(axis=1), '.')
    plt.fill_between(range(1, num_docs+1), time_taken.mean(axis=1)- time_taken.std(axis=1), time_taken.mean(axis=1)+time_taken.std(axis=1), alpha=0.1)

    plt.figure()
    plt.plot(range(1, num_docs+1), num_batches_sorted.mean(axis=1), '.')
    plt.plot(range(1, num_docs+1), num_batches.mean(axis=1), '.')
    plt.show()
    '''