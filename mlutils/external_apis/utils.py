def add_array_api_parameters(url: str, param_name: str, param_values: list) -> str:
    """Expands API url with array parameters

    :param url: API url in the form {base_url}/{endpoint}?
    :param param_name: name of the parameter, e.g. {base_url}/{endpoint}?{param_name}=
    :param param_values: values that parameter takes
    :return: updated url {base_url}/{endpoint}?{param_name}={param_value}&...
    """
    for param_value in param_values:
        url = f'{url}&{param_name}={param_value}'
    
    return url

def is_request_valid(status_code: int) -> bool:
    """checks if request was valid

    :param status_code: status_code from request
    :return: True if valid, False otherwise
    """
    if str(status_code).startswith('2'):
        return True
    
    return False