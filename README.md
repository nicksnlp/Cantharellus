# LingDig_Mushrooms
A research project in language technology on hallucination detection in connection with the international shared task Mu-SHROOM

## How to use this repository:
1. Clone it to your local machine. Use command line or GitHub Desktop, Visual Studio Code, etc.

3. Arrange your files structurally, use separate directories for different experiments.

4. When working with python and installing packages, use the environment (for example mushrooms_env),  
☞ **Note:** Locate it somewhere outside of this repository.

    a. **Create a virtual environment**

        'python3 -m venv mushrooms_env'


    b. **Activate the environment**

      - **MacOS/Linux**  

        'source mushrooms_env/bin/activate'

      - **Windows**  

        'mushrooms_env\Scripts\activate'

    c. **Install packages:**

        'pip3 install package_name' / 'pip install package_name'

    d. Once all necessary packages are installed, generate a <strong>requirements.yourname.date.txt</strong> file to record the environment’s dependencies. This file can be shared with others or used to recreate the environment.

    We can use different names for the requirements file, or add a date/your name to avoid conficts.

        'pip3 freeze > requirements.yourname.date.txt / pip freeze > requirements.yourname.date.txt'

  5. **To recreate the environment on a different machine, follow these steps:**

    1.Clone the project folder.
    2.Create and activate a new virtual environment.
    3.Install all packages from a requirements.txt file by running:
   
        'pip3 install -r requirements.txt' / 'pip install -r requirements.txt'
    
    4. When you're done, deactivate the environment by running:

        'deactivate'
