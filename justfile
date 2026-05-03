set windows-shell := ["powershell", "-NoProfile", "-Command"]

RM_R := if os_family() == "windows" { "Remove-Item -Recurse" } else { "rm -r" }
ON_ERROR_CONTINUE := if os_family() == "windows" { '; $ErrorActionPreference = "Continue"' } else { "|| true" }
RM_PYCACHE := if os_family() == "windows" { 'Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force' } else { 'find . -depth -type d -name "__pycache__" -exec rm -r {} +' }

DEFAULT_VENV_NAME := ".venv"

# interactive menu to select a command
_default:
	@ just --choose --quiet || :

[private]
_check_arg arg:
    @ if [ -z "{{arg}}" ]; then echo "error: missing argument"; exit 1; fi

alias default := _default

# install all dependencies and sync the virtual environment
install:
	uv sync --all-groups

# convert notebook to pdf
topdf input="" output=(if input == "" { "" } else { file_stem(input) }): (_check_arg input)
	uv run jupyter nbconvert --to webpdf --allow-chromium-download --no-prompt {{input}} --output {{output}}

# create a fresh virtual environment (defaults to .venv)
venv name=DEFAULT_VENV_NAME:
	uv venv {{ name }}

# format codebase
format:
	uv tool run ruff format

# cleans up the project
clean:
	uv clean
	{{ RM_R }} .ruff_cache 		{{ ON_ERROR_CONTINUE }}
	{{ RM_R }} .venv 			{{ ON_ERROR_CONTINUE }}
	{{ RM_PYCACHE }} 			{{ ON_ERROR_CONTINUE }}