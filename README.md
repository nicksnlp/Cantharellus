# LingDig_Mushrooms
A research project in language technology on hallucination detection in connection with the international shared task Mu-SHROOM

# Distribution of Work

## 1. **Sofia**
- **Developing Prompts**: Create and optimize various prompts for detecting hallucinations.
- **Documentation**: Maintain README files for task documentation.

## 2. **Zang**
- **Developing Prompts**: Work on developing and refining prompts for detecting hallucinations.
- **Documentation**: Maintain README files for task documentation.

## 3. **Nick**
- **Developing Prompts**: Work on developing and refining prompts for detecting hallucinations.
- **Data Augmentation**: Implement scripts to augment data using LLMs.
- **Combining Outputs**: Develop scripts to combine outputs from different models.
- **Documentation**: Maintain README files for task documentation.

## 4. **Xinyuan**
- **Training Classifier**: Focus on training the classifier to detect hallucinated spans in the outputs.
- **Documentation**: Maintain README files for task documentation.

## Unallocated Tasks
- **Information Retrieval**: Implement a script for retrieving information from Wikipedia to help detect hallucinations in the model outputs.
- **Model Output Combination**: Further explore and implement strategies for combining model outputs if needed.

## How to use this repository:
1. Clone it to your local machine. You can use the command line, GitHub Desktop, Visual Studio Code, etc.

2. Arrange your files structurally. Use separate directories for different experiments. You can add also notebooks inside your folders.

**Initial Project structure**  
(see bash script /LingDig_Mushrooms/Nick/setup_project.sh)
```
 project-root/
│
├── data/                  # Data storage folder
│   ├── raw/               # Raw unlabelled training data
│   ├── generated/         # Generated outputs from LLMs
│   ├── annotated/         # Annotated outputs (with hallucination labels)
│   └── logs/              # Logs for tracking progress
│
├── src/                   # Source code for project
│   ├── __init__.py        # Makes src a Python module
│   ├── data_augmentation.py # Script for augmenting data
│   ├── classifier.py       # Script for training and evaluating the classifier
│   ├── combine_outputs.py   # Script for combining model outputs
│   └── retrieve_info.py     # Script for information retrieval from Wikipedia (to detect hallucinations)
│
├── Sofia/                 # Directory for Sofia’s work
│   ├── develop_prompts.py   # Script for developing prompts
│   └── README.md           # Task documentation
│
├── Zang/                  # Directory for Zang’s work
│   ├── develop_prompts.py    # Script for developing prompts
│   └── README.md           # Task documentation
│
├── Nick/                  # Directory for Nick’s work
│   ├── develop_prompts.py    # Script for developing prompts
│   ├── augment_data.py       # Script for data augmentation
│   ├── combine_outputs.py     # Script for combining model outputs
│   └── README.md             # Task documentation
│
├── Xinyuan/               # Directory for Xinyuan’s work
│   ├── train_classifier.py   # Script for training the classifier
│   └── README.md             # Task documentation
│
├── .gitignore             # Files to ignore in Git
├── requirements.txt       # Project dependencies
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
    b. Create and activate a new virtual environment (outside of the project folder)  
    c. Install all packages from a `requirements.txt` file by running:  
      ```bash
      pip3 install -r requirements.txt  # or `pip install -r requirements.txt`
      ```
    d. When you're done, deactivate the environment by running:  
      ```bash
      deactivate
      ```
