import time
import requests
import subprocess
import getpass
from bs4 import BeautifulSoup
import argparse

parser = argparse.ArgumentParser(
    prog='Web-CAT Submitter',
    description='Submits files on the command line to Web-CAT for grading. '
                'Requirements: python, requests, BeautifulSoup, getpass',
)

parser.add_argument('file', type=str, nargs='+', help='A file to submit (ex. "p03.py" "lexpress.fr.json")')
parser.add_argument('-a', '--assignment', type=str, nargs='?', help='Assignment name (ex. "CS 3435/Beautiful Soup")')
parser.add_argument('-u', '--username', type=str, nargs='?', help='User name')
args = parser.parse_args()

file_names = args.file
assignment = args.assignment
user_name = args.username

print('file_names:', file_names)

if assignment is None:

    cmd = ['java', '-jar', 'webcat-submitter-1.0.5.jar', '-t', 'http://webcatvm.cs.appstate.edu:8080/Web-CAT/WebObjects/Web-CAT.woa/wa/assignments/eclipse', '-l']

    result = subprocess.run(cmd, capture_output=True)

    open_submissions = [x for x in result.stdout.decode('ascii').split('\n') if '/' in x]

    for i, s in enumerate(open_submissions):
        print(f'{i:2d}: {s}')

    if not open_submissions:
        print('There are no open submissions.')
        exit(0)


    index = None
    while index is None:
        try:
            index = int(input(f'Enter the submission index (0-{len(open_submissions)-1}): '))
        except ValueError:
            print("Not an integer")

    assignment = open_submissions[index]

print('assignment:', assignment)

if user_name is None:
    user_name = getpass.getuser()

print('username:', user_name)

password = getpass.getpass()

cmd = [
    'java',
    '-jar',
    'webcat-submitter-1.0.5.jar',
    '-t',
    'http://webcatvm.cs.appstate.edu:8080/Web-CAT/WebObjects/Web-CAT.woa/wa/assignments/eclipse',
    '-u',
    user_name,
    '-p',
    password,
    '-a',
    assignment
] + file_names

result = subprocess.run(cmd, capture_output=True)
html_text = result.stdout.decode('ascii')
soup = BeautifulSoup(html_text, 'html.parser')

href = soup.body.a['href']
print(href)

delay = 15
while True:
    result = requests.get(href)
    soup = BeautifulSoup(result.text, 'html.parser')
    summary = soup.find('div', title='Result Summary')
    score = summary.find(text='Total Score').find_next('b').text.strip()

    # print result summary
    table = soup.find('div', title='Result Summary').table
    results_str = '\nResult Summary:\n--------------\n'
    for tr in table.find_all('tr'):
        k = tr.th.text.strip()
        v = tr.td.text.strip()
        results_str += f'{k}: {v}\n'
    results_str += '\n'

    if score != '<Queued>':
        break

    print(results_str)
    print(f'Waiting {delay} seconds\n')
    time.sleep(delay)

results = soup.find('div', title=lambda t: t and t.startswith("Results From Running Your Instructor's Tests"))
print(results.text.strip())
print(results_str)
