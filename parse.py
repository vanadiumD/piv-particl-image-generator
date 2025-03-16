import re

def preprocess_expression(expression):

    processed = expression
    processed = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', processed)
    processed = re.sub(r'([a-zA-Z)])(\d)', r'\1*\2', processed)
    processed = processed.replace('^', '**')
    return processed