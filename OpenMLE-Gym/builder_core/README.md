# Builder Core

This directory contains the LangGraph pipeline used by the public `openmle-task build` command.

Most users should run the top-level CLI instead of importing these modules directly:

```bash
uv run openmle-task build --slugs-file examples/slugs.txt --execute
```

The wrapper in `openmle_gym/build.py` sets paths, credentials, batch names, and raw-data retention options consistently for release use.
