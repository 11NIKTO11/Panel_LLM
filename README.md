## ️ Running the Project

1.  **Download the Dataset**:
    * Navigate to the [Society of Distrust (CZ) dataset](https://archivdv.soc.cas.cz/dataset.xhtml?persistentId=doi:10.14473/CSDA/ZKWKVZ) page.
    * Download the data.
    * Extract the contents and place the `.sav` file in the project's root directory at the following path:
        ```bash
        ./SoD-CZ/Society_of_Distrust - CZ.sav
        ```
    * If you place the dataset elsewhere or with different name, be sure to update the path in the code.


2.  **Set Up OpenAI API Key (Optional)**:
    * To use GPT models (either directly or via AnyLLM), you need to set your OpenAI API key.
    * Save your key as an environment variable named `OPENAI_API_KEY`.

    For **Windows (Command Prompt)**:
    ```bash
    set OPENAI_API_KEY=your_api_key_here
    ```
The main entry point for this project is the Jupyter Notebook `SoD_data_processing.ipynb`.