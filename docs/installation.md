# Installation and Environment Setup Guide

This document describes step-by-step how to configure the local development environment to run the fruit and vegetable classification system.

---

## Prerequisites

Make sure you have the following installed on your system:
* **Python 3.8** or higher (recommended 3.10 or 3.11).
* **Git** to clone the repository.

---

## Installation Steps

### 1. Clone the Repository
Open your terminal (PowerShell, Bash, or CMD) and run:
```bash
git clone https://github.com/SaraLuciaa/fruit-quality-classification.git
cd fruit-quality-classification
```

### 2. Create the Virtual Environment
It is highly recommended to use a virtual environment to avoid conflicts between your system's global dependencies and the project's dependencies.

#### On Windows:
```powershell
python -m venv .venv
```

#### On macOS or Linux:
```bash
python3 -m venv .venv
```

### 3. Activate the Virtual Environment

#### On Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

#### On Windows (CMD):
```cmd
.venv\Scripts\activate.bat
```

#### On macOS or Linux:
```bash
source .venv/bin/activate
```

Once activated, you will see the `(.venv)` prefix at the start of your command prompt.

### 4. Install Dependencies
With the virtual environment active, install the required libraries by running:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> We use `opencv-python-headless` instead of the standard `opencv-python` to avoid issues with system graphics libraries (such as X11 dependencies on Linux/Docker), since the main graphical interface will be fully managed by Streamlit in the browser.

---

## Running the Application

To start the local Streamlit development server:
```bash
streamlit run src/main.py
```

Once the command runs successfully, you will see something like this in your console:
```text
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.XX:8501
```

Open `http://localhost:8501` in your browser to see the interactive interface.

---

## Running Unit Tests
To validate that the environment is properly configured and all modules import without issues, run:
```bash
pytest
```
