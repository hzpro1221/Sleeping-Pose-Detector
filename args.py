import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--PATH',
        type=str,
    )
    parsed_args = parser.parse_args()
    return parsed_args