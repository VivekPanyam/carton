This runner can run arbitrary python code (including PyTorch models).

A carton containing a python "model" contains the original code along with a lockfile for depdendencies.

TODO: add more details on how to actually create a model for this runner

# How it works

## Packing process

Packing a python carton works as follows:

1. Get the reqirements.txt and compute a sha256 for it
2. If all the following are true, we don't need to do anything else:
    - There's already a lockfile and the requirements hash in the lockfile matches the hash from above
    - The lockfile contains a set of requirements for the current [environment](https://peps.python.org/pep-0508/#environment-markers) (OS, arch, etc).
3. Otherwise, we get a list of dependencies and transitive dependencies from `pip install --report`
4. For each (transitive) dependency:
    - If it's a wheel on PyPi, include its URL and sha256 in the lockfile
    - If it's a non-pypi wheel, download it and save it to `{input_dir}/.carton/bundled_wheels/{sha256}/{filename}` in the input folder
    - If it's not a wheel, build a wheel for it using `pip wheel` and then save that wheel in bundled_wheels as above
5. Create a lockfile with the above information int `{input_dir}/.carton/carton.lock`

## Loading process

1. If the model contains a `.carton/carton.lock` file and we have a lockfile for the current [environment](https://peps.python.org/pep-0508/#environment-markers) (OS, arch, etc), do the following for each dependency:
    - If we already have a package with the specified sha256 in `~/.carton/pythonpackages/py{MAJOR}{MINOR}/{sha256}`, add it to `sys.path`
    - Otherwise, if it's a PyPi wheel URL, download and unzip it to `~/.carton/pythonpackages/py{MAJOR}{MINOR}/{sha256}`. Add that path to `sys.path`
    - If it's a bundled wheel, unzip it to a temp directory and add that directory to `sys.path`
2. If we don't have a lockfile for the current environment, log a warning message and then run the packing process followed by step 1 above. This could be made more efficient, but it's mostly for convenience.
3. Run the model's entrypoint

There are some details skipped above (like making subprocesses work), but that's the gist of how it works.