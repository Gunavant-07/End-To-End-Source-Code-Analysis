import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.pt",
    ".env",
    "setup.py",
    "app.py",
    "research/trials.ipynb"
]

for file_path in list_of_files:
    file_path = Path(file_path)
    filedir, filename = os.path.split(file_path)
    
    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir}")
    
    if not os.path.exists(file_path) or os.path.getsize(file_path)  == 0:
        with open(file_path,'w') as f:
            pass
        logging.info(f"Created empty file: {file_path}")
    
    else:
        logging.info(f"File already exists and is not empty: {file_path}")