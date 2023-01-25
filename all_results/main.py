# replace '}' with '}\n' in a file and save it
# the file is pretty big, so it needs to be done in chunks

import os

def main():
    with open('test.txt', 'r') as f:
        data = f.read()
    data = data.replace('}', '}\n')
    with open('test.txt', 'w') as f:
        f.write(data)

if __name__ == '__main__':
    main()