# LingDig_Mushrooms
A research project in language technology on hallucination detection in connection with the international shared task Mu-SHROOM

## How to use this repository:
1. Clone it to your local machine. You can use the command line, GitHub Desktop, Visual Studio Code, etc.

2. Arrange your files structurally. Use separate directories for different experiments.

3. When working with Python and installing packages, use a virtual environment (e.g., `mushrooms_env`).  
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
