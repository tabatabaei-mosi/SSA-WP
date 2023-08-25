# SSA-WP

A research project that explores the promising application of the Sparrow Search Algorithm (SSA) to the challenging problem of well placement optimization.  The objective of this project is to leverage the power of SSA to identify optimal well location and flow rates within a real-world case study.

## How to run the code

1. Create a conda environment with the following command:

    ```bash
    conda create -n <env name> python=3.8
    ```

2. Activate the environment:

    ```bash
    conda activate <env name>
    ```

3. Clone the repository:

    ```bash
    git clone <repo-url>
    ```

4. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

5. Provide following files to the `src/` directory.

    - `.DATA` file to the `src/` directory (Please name this file as `Run.DATA`, or change the name in `.bat` file)
    - `$MatEcl.bat` file in the `src/` directory used to call the simulation (if you want to change the name, please change also in `optimizer` module).
    - You can change `NPV.xlsx` file for constant of NPV formula.

6. Provide following files to the `src/INCLUDE` directory:
    - `.GRDECL` file to the `src/INCLUDE` directory (use this name, `SSA_RQI_0` or change some part of code).
    - Any of following files that you prefer to optimize the variables in should include in the `src/INCLUDE`: 
        - `welspecs.inc`
        - `wconprod.inc`
        - `wconinje.inc`
        - `compdat.inc`

7. Run the code:

    ```bash
    python src/main.py
    ```

## NOTE

If you have any questions, please contact me via email: tabatabaei.mosi@gmail.com