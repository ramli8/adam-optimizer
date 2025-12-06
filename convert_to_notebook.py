import json
import re

# Read the Python file
with open('SVQR_Fix.py', 'r', encoding='utf-8') as f:
    py_content = f.read()

# Split by sections based on comments
cells = []

# Split content into cells
lines = py_content.split('\n')
current_cell = []
current_type = 'code'

for line in lines:
    # Check if line is a markdown comment (starts with # ###)
    if line.strip().startswith('# ###'):
        # Save previous cell
        if current_cell:
            cells.append({
                "cell_type": current_type,
                "metadata": {},
                "source": current_cell if current_type == 'code' else [l.lstrip('# ') for l in current_cell]
            })
            if current_type == 'code':
                cells[-1]["outputs"] = []
                cells[-1]["execution_count"] = None
        
        # Start new markdown cell
        current_cell = [line.lstrip('# ') + '\n']
        current_type = 'markdown'
    elif current_type == 'markdown' and line.strip().startswith('#') and not line.strip().startswith('##'):
        # Continue markdown cell
        current_cell.append(line.lstrip('# ') + '\n')
    else:
        # Switch to code cell if needed
        if current_type == 'markdown' and line.strip():
            # Save markdown cell
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": current_cell
            })
            current_cell = [line + '\n']
            current_type = 'code'
        else:
            current_cell.append(line + '\n')

# Add last cell
if current_cell:
    cells.append({
        "cell_type": current_type,
        "metadata": {},
        "source": current_cell if current_type == 'code' else [l.lstrip('# ') for l in current_cell]
    })
    if current_type == 'code':
        cells[-1]["outputs"] = []
        cells[-1]["execution_count"] = None

# Create notebook structure
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
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write to .ipynb file
with open('SVQR_Fix.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("✅ Successfully converted SVQR_Fix.py to SVQR_Fix.ipynb")
