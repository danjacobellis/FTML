def build():
    from subprocess import run
    from shutil import rmtree, copytree, copyfile
    from os.path import exists
    from pathlib import Path
    
    if exists("docs"):
        rmtree("docs")
    result = run("sphinx-build . docs",capture_output=True)
    print(result.stdout.decode("utf-8"))
    Path("docs/.nojekyll").touch()
    
    result = run("jupyter-nbconvert --to slides critique_paper2.ipynb --output docs\critique_paper2 --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='remove-nb-cell'",capture_output=True)
    print(result.stdout.decode("utf-8"))
    
    result = run("jupyter-nbconvert --to slides present_paper10.ipynb --output docs\present_paper10 --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='remove-nb-cell'",capture_output=True)
    print(result.stdout.decode("utf-8"))
    
if __name__ == '__main__':
    build()