def build():
    from subprocess import run
    from shutil import rmtree, copytree, copyfile
    from os.path import exists
    from pathlib import Path
    
    if exists("docs"):
        rmtree("docs")
    result = run(["sphinx-build", ".", "docs"],capture_output=True)
    print(result.stdout.decode("utf-8"))
    if exists("jupyter_execute"):
        rmtree("jupyter_execute")
    Path("docs/.nojekyll").touch()
    
    result = run("jupyter-nbconvert --to slides critique_paper2.ipynb --output docs/critique_paper2 --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='remove-nb-cell'",capture_output=True, shell=True)
    print(result.stdout.decode("utf-8"))
    
    result = run("jupyter-nbconvert --to slides present_paper10.ipynb --output docs/present_paper10 --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='remove-nb-cell'",capture_output=True, shell=True)
    print(result.stdout.decode("utf-8"))
    
    result = run("jupyter-nbconvert --to slides critique_50y_fairness.ipynb --output docs/critique_50y_fairness --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='remove-nb-cell'",capture_output=True, shell=True)
    print(result.stdout.decode("utf-8"))
    
    result = run("jupyter-nbconvert --to slides survey_of_datasets.ipynb --output docs/survey_of_datasets --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='remove-nb-cell'",capture_output=True, shell=True)
    print(result.stdout.decode("utf-8"))
    
    result = run("jupyter-nbconvert --to slides failing_loudly.ipynb --output docs/failing_loudly --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='remove-nb-cell' --TagRemovePreprocessor.remove_input_tags='remove-nb-input'",capture_output=True, shell=True)
    print(result.stdout.decode("utf-8"))
    
if __name__ == '__main__':
    build()