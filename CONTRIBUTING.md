# CONTRIBUTING GUIDELINES

## Do I need requirement.txt for uv?

No, you don't need requirements.txt for uv. Since you're already using pyproject.toml, that's your source of truth for dependencies.

uv uses pyproject.toml directly. The requirements.txt is optional and only useful if you:

Need to lock exact versions for reproducibility (use uv lock instead, which creates uv.lock)
Want to share requirements with tools that only support requirements.txt
You can stick with just pyproject.toml and uv sync. If you want reproducible builds across environments, run:

This creates a uv.lock file that pins exact versions, similar to what requirements.txt would do but better managed by uv.


## import numpy as np takes forever, without any error after exectuing the code
One of the common problem - it is missing ipykernel. Check if uv.lock has this as dependencies

```bash
uv add ipykernel
```