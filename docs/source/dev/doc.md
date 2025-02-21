# Write documentation

To update the documentation, add or edit pages in `.rst`(reStructuredText) or `*.md` (markdown) format
within the `docs/source` directory. Remember to add those new files in the table of content in `docs/source/index.rst`.

Note that whenever you open a Pull Request on the project [main repository](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms),
the documentation will be automatically built and deployed to `org.readthedocs.build` (see the exact link automatically defined at the end of your PR descreiption).
This lets you check
that your updates are taken into account as expected.

## Build documentation

To build the documentation locally, you first need to install the project with its documentation dependencies:

```bash
$ pip install -e '.[doc, testing]'
```

Then, you can build the documentation with:

```bash
$ sphinx-build -b html docs/source docs/build
```

The documentation is then available at `docs/build/index.html`. If you want to open
it locally, you may start a local HTTP server with

```bash
$ python -m http.server -d docs/build/
Serving HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...
```

You may then open the documentation in your browser at `http://localhost:8000/index.html`.


## Test the documentation

### Tutorials

By default, when building the documentation locally, notebook tutorials are executed if you have `ipykernel` installed, and errors on execution are ignored.

You may force their execution by adding the option `--define nbsphinx_execute=always` to the `sphinx-build` command.
You may also require the tutorials to run without error by adding the option `--define nbsphinx_allow_errors=0`.

When ReadTheDocs build the documentation, tutorials are always executed and documentation fails to build on errors.

### Code examples

You may embed python code in the documentation and have it tested automatically. This is done in `rST` mode by either defining plain python-terminal like commands and their (optional) expected output, outside of a code-block:

```rst
>>> import few
>>> cfg = few.get_config()
>>> cfg.log_level
30
>>> cfg.file_integrity_check
'once'
```

Each of these command will be tested automatically and the output will be compared to the expected output. In a markdown file, just embed this `rST` fragment inside a ` ```{eval-rst} ... ``` ` block

````markdown
```{eval-rst}
>>> import few
>>> cfg = few.get_config()
>>> cfg.log_level
30
>>> cfg.file_integrity_check
'once'
```
````

You may also define multiple blocks using the `testcode` and optional `testoutput` directives.
These directives can be grouped together using a group name. See the [sphinx doctest documentation](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html) for more details, and `docs/source/user/cfg.md` for an example.

To run the tests, simply build the documentation using the `doctest` builder:

```bash
$ sphinx-build -M doctest docs/source docs/build
...
Doctest summary
===============
    7 tests
    0 failures in tests
    0 failures in setup code
    0 failures in cleanup code
build succeeded.
```

## Add a tutorial

Tutorials (and example notebooks in general) are added to the `docs/source/tutorial` directory in the `-- Copy example notebook --` section of `docs/source/conf.py`. Adapt it to your needs to add supplementary tutorials and example notebooks.

Remember to also declare these new notebooks in the documentation index (`index.rst`).
