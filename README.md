# LingDig_Mushrooms
A research project in language technology on hallucination detection in connection with the international shared task Mu-SHROOM

## How to use this repository:
1. Clone it to your local machine. You can use the command line, GitHub Desktop, Visual Studio Code, etc.

2. Arrange your files structurally. Use separate directories for different experiments.

**Project structure**
```
    lingdig_mushrooms/
    │
    ├── docs/                  # Documentation files (e.g., project overview, setup instructions)
    │   └── setup.md           # Project setup guide
    │
    ├── data/                  # Data storage folder (input/output data, models)
    │   ├── raw/               # Raw input data (e.g., LLM prompts)
    │   ├── processed/         # Preprocessed data (e.g., cleaned data)
    │   └── outputs/           # Outputs of LLM (e.g., responses, analysis results)
    │
    ├── src/                   # Source code for common functionality
    │   ├── __init__.py        # Makes src a Python module
    │   ├── config.py          # Configuration file (e.g., API keys, model parameters)
    │   ├── preprocessing.py    # Common preprocessing code
    │   ├── llm_utils.py       # Utility functions for interacting with LLMs
    │   └── hallucination_analysis.py  # Code for analyzing LLM outputs
    │
    ├── kammerat/              # Directory for Kammerat’s work
    │   ├── task.py            # Script for Kammerat's implementation
    │   └── README.md          # Task-specific documentation
    │
    ├── nicksnlp/              # Directory for NicksNLP’s work
    │   ├── task.py            # Script for NicksNLP's implementation
    │   └── README.md          # Task-specific documentation
    │
    ├── tkzang/                # Directory for TkZang’s work
    │   ├── task.py            # Script for TkZang's implementation
    │   └── README.md          # Task-specific documentation
    │
    ├── XinyuanMO/             # Directory for XinyuanMO’s work
    │   ├── task.py            # Script for XinyuanMO's implementation
    │   └── README.md          # Task-specific documentation
    │
    ├── tests/                 # Unit tests for the shared code and individual tasks
    │   ├── test_preprocessing.py # Tests for common preprocessing functions
    │   └── test_hallucination_analysis.py # Tests for analysis code
    │
    ├── .gitignore             # Files and directories to be ignored by git
    ├── requirements.txt       # Python dependencies
    └── README.md              # Project overview and instructions
```

3. When working with Python and installing packages, use a *virtual environment* (e.g. `mushrooms_env`).  
   ☞ **Note:** Locate it somewhere outside of this repository.

    a. **Create a virtual environment**  
    ```bash
    python3 -m venv mushrooms_env
    ```

    b. **Activate the environment**
      
      - **MacOS/Linux**  
      ```bash
      source mushrooms_env/bin/activate
      ```

      - **Windows**  
      ```bash
      mushrooms_env\Scripts\activate
      ```

    c. **Install packages**  
    ```bash
    pip3 install package_name  # or `pip install package_name`
    ```

    d. Once all necessary packages are installed, generate a `requirements.yourname.date.txt` file to record the environment’s dependencies. This file can be shared with others or used to recreate the environment.

    We can use different names for the requirements file or add a date/your name to avoid conflicts.  
    ```bash
    pip3 freeze > requirements.yourname.date.txt  # or `pip freeze > requirements.yourname.date.txt`
    ```

4. **To recreate the environment on a different machine, follow these steps:**
   
    a. Clone the project folder.  
    b. Create and activate a new virtual environment.  
    c. Install all packages from a `requirements.txt` file by running:  
      ```bash
      pip3 install -r requirements.txt  # or `pip install -r requirements.txt`
      ```
    d. When you're done, deactivate the environment by running:  
      ```bash
      deactivate
      ```
