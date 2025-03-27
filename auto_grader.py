'''
Main script for testing the assignment.
Runs the tests on the results json file.
'''

import argparse
import json
import re

def get_args():
    parser = argparse.ArgumentParser(description='Language Modeling')
    parser.add_argument('test', type=str, help='The test to perform.')
    return parser.parse_args()

def test_link():
    # Read link
    link_file = "notebook_link.txt"
    with open(link_file, 'r') as f:
        link = f.read().strip()
    
    # Make sure link contains usp=sharing
    if "usp=sharing" not in link:
        return f"Link {link} doesn't seem to have share access"
    return 1
    

def test_preprocess(results):
    if results["vocab_length"] != 1804:
        return f"Vocab length is {results['vocab_length']}, expected 1804"
    return 1

def test_build_lm(results):
    if results["english_2_gram_length"] != 748:
        return f"English 2-gram length is {results['english_2_gram_length']}, expected 748"
    if results["english_3_gram_length"] != 8239:
        return f"English 3-gram length is {results['english_3_gram_length']}, expected 8239"
    if results["french_3_gram_length"] != 8286:
        return f"French 3-gram length is {results['french_3_gram_length']}, expected 8286"
    if results["spanish_3_gram_length"] != 8469:
        return f"Spanish 3-gram length is {results['spanish_3_gram_length']}, expected 8469"
    return 1
    
def relative_difference(expected, actual):
    """Calculate the relative difference between expected and actual values."""
    return abs(expected - actual) / expected

def test_eval(results):
    perplexity_en_on_en = float(results["en_on_en"])  
    perplexity_en_on_fr = float(results["en_on_fr"])  
    perplexity_en_on_tl = float(results["en_on_tl"])  
    perplexity_en_on_nl = float(results["en_on_nl"])  

    perplexities = [
        perplexity_en_on_en,
        perplexity_en_on_fr,
        perplexity_en_on_tl,
        perplexity_en_on_nl
    ]

    if min(perplexities) != perplexity_en_on_en:
        return f"English model should perform best on English text. Results: {results}"
    
    if not (perplexity_en_on_en <= perplexity_en_on_fr <= max(perplexity_en_on_tl, perplexity_en_on_nl)):
        return f"Expected increasing perplexity from English to other languages. Results: {results}"

    return 1

    
def test_match(results):
    perplexity_en_on_en = int(results["en_en_3"])  
    perplexity_en_on_tl = int(results["en_tl_3"])  
    perplexity_en_on_nl = int(results["en_nl_3"])  

    perplexities = [
        perplexity_en_on_en,
        perplexity_en_on_tl,
        perplexity_en_on_nl
    ]

    if min(perplexities) != perplexity_en_on_en:
        return f"English model should perform best on English text. Results: {results}"

    if not (perplexity_en_on_en <= max(perplexity_en_on_tl, perplexity_en_on_nl)):
        return f"Expected increasing perplexity from English to other languages. Results: {results}"

    return 1


def test_generate(results):
    if not results["english_2_gram"].startswith("I am"):
        return f"English 2-gram does not start with 'I am', but with {results['english_2_gram']}"
    if not results["french_3_gram"].startswith("Je suis"):
        return f"French 3-gram does not start with 'Je suis', but with {results['french_3_gram']}"
    return 1

def main():
    # Get command line arguments
    args = get_args()

    # Read results.json
    with open('results.json', 'r') as f:
        results = json.load(f)

    # Initialize the result variable
    result = None

    # Switch between the tests
    match args.test:
        case 'test_link':
            result = test_link()
        case 'test_preprocess':
            result = test_preprocess(results["test_preprocess"])
        case 'test_build_lm':
            result = test_build_lm(results["test_build_lm"])
        case 'test_eval':
            result = test_eval(results["test_eval"])
        case 'test_match':
            result = test_match(results["test_match"])
        case 'test_generate':
            result = test_generate(results["test_generate"])
        case _:
            print('Invalid test.')

    # Print the result for the autograder to capture
    if result is not None:
        print(result)

if __name__ == '__main__':
    main()
