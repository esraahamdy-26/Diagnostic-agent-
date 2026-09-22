# Virtual Environment Setup Guide

To run the project and avoid dependency conflicts, it is highly recommended to use a virtual environment (`venv`). Follow these steps according to your operating system:

---

## 1. Create the Virtual Environment

Open your terminal or PowerShell in the root directory of the project and run:

```bash
# On Windows, macOS, or Linux
python -m venv venv
```

*Note: If the command above does not work on macOS/Linux, try using `python3`:*
```bash
python3 -m venv venv
```

---

## 2. Activate the Virtual Environment

You must activate the virtual environment every time you open a new terminal window to work on the project.

### On Windows:
* **Using PowerShell (Recommended):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If you get a script execution policy error, run PowerShell as Administrator and execute `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` first).*
* **Using Command Prompt (CMD):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```

### On macOS / Linux:
```bash
source venv/bin/activate
```

Once activated, you will see `(venv)` prepended to your terminal prompt.

---

## 3. Install Required Dependencies

With the virtual environment active, upgrade `pip` and install the packages specified in `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Link the Virtual Environment to Jupyter Notebook

To make the libraries installed in your virtual environment available inside Jupyter Notebook, you need to register a dedicated Jupyter kernel:

1. Ensure your virtual environment is active.
2. Install `ipykernel`:
   ```bash
   pip install ipykernel
   ```
3. Register the virtual environment as a kernel:
   ```bash
   python -m ipykernel install --user --name=venv --display-name "Python (venv)"
   ```
4. When you open any notebook in Jupyter, select the **"Python (venv)"** kernel from the top menu (`Kernel` -> `Change Kernel`).

---

## 5. Deactivate the Virtual Environment

When you are done working and want to return to your system's global Python environment, run:
```bash
deactivate
```
