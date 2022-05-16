from termcolor import colored

# Print error message
def print_error(str_in):
    print(colored(str_in, 'red'))

# Print ongoing message
def print_ongoing(str_in):
    print(colored(str_in, 'blue'))

# Print success message
def print_success(str_in):
    print(colored(str_in, 'green'))