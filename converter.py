import json

def convert_to_ipynb(py_file, ipynb_file):
    with open(py_file, 'r', encoding='utf-8') as f:
        src = f.read()
    
    cells = []
    blocks = src.split('# ─────────────────────────────────────────────\n')
    
    for block in blocks:
        if not block.strip():
            continue
        
        # Splitting keeping the original lines and reconstructing array of strings
        source_lines = [line + "\n" for line in block.split('\n')]
        
        # fix the last newline
        if len(source_lines) > 0 and source_lines[-1] == "\n":
            source_lines.pop()
        elif len(source_lines) > 0 and source_lines[-1].endswith("\n"):
            source_lines[-1] = source_lines[-1][:-1]
            
        # extract any title and turn into markdown cell if it starts with # BAGIAN
        lines_to_check = [l.strip() for l in block.split('\n') if l.strip()]
        if lines_to_check and lines_to_check[0].startswith("# BAGIAN"):
            # just treat entire block as code, it's safer
            pass
            
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source_lines
        })
        
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(ipynb_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

if __name__ == '__main__':
    convert_to_ipynb('svqr.py', 'svqr.ipynb')
    print("Conversion finished.")
