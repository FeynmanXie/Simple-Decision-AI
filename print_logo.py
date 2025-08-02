import os

script_dir = os.path.dirname(__file__)
logo_path = os.path.join(script_dir, 'logo.txt')

try:
    with open(logo_path, 'r', encoding='utf-8') as f:
        print(f.read())
except FileNotFoundError:
    print("Error: logo.txt not found.")
except Exception as e:
    print(f"An error occurred: {e}")