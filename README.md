# DATA7001 Group 15 Project

## Prerequisites

You need to install two core tools to get started.

### 1. Install `uv` (Package Manager)
Open your terminal:

```bash
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Mac/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
wget -qO- https://astral.sh/uv/install.sh | sh
```

Look at the official **uv** [documentation](https://docs.astral.sh/uv/getting-started/installation/) for more help.

### 2. Install `just` (Command Runner)
Once `uv` is installed, run:

```bash
uv tool install just
```

## Getting Started

### 1. Initialize the Environment
Clone the repository and run the following in the project root:

```bash
just install
```

This syncs all dependencies (Pandas, Numpy, Matplotlib, etc.) into a local `.venv` folder.

### 2. Configure VS Code
- Open a .ipynb notebook.
- Click Select Kernel in the top-right corner.
- Choose Python Environments...
- Select the interpreter located at `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Mac/Linux). It should be also called `data7001-group15-project`.

## Available Commands

Run `just` by itself to see the interactive menu:

| Command | Usage | Description |
| :--- | :--- | :--- |
| **default** | `just` | Opens the interactive command picker menu. |
| **install** | `just install` | Syncs all libraries and prepares the virtual environment. |
| **clean** | `just clean` | Removes the `.venv` and clears the `uv` cache. |
| **venv** | `just venv <name>` | Manually creates a new virtual environment if needed. |
| **topdf** | `just topdf <file>` | Converts a notebook to a clean PDF (auto-downloads Chromium). |
