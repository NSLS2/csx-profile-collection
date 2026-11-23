# csx_tools

Tools for CSX beamline at NSLS-II

## Usage

The repository root is an IPython profile directory: `startup/` runs in the `terminal`,
`qs`, and `test` environments, which do not install the `csx_tools` package.

| Command | Purpose |
| ------------------------- | ------------------------------------------------------- |
| `pixi run start` | Start IPython with the profile (`terminal` environment) |
| `pixi run qs-backend` | Start the Queue Server RE Manager (`qs` environment) |
| `pixi run qs-server` | Start the Queue Server HTTP server (`qs` environment) |
| `pixi run tests` | Test the profile (`test` environment) |
| `pixi run -e dev lint` | Run the pre-commit hooks |
| `pixi run -e dev tests` | Test the `csx_tools` package |
