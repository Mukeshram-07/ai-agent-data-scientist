import nbformat as nbf

def generate_notebook(df):

    nb = nbf.v4.new_notebook()

    code = f"""
import pandas as pd

df = pd.read_csv("your_dataset.csv")

print(df.head())

print(df.describe())
"""

    nb.cells.append(nbf.v4.new_code_cell(code))

    with open("analysis_notebook.ipynb", "w") as f:

        nbf.write(nb, f)

    return "analysis_notebook.ipynb"
