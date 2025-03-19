import pandas as pd
import json

""" JSON """


def load_json(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, file_name):
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def fill_json_from_excel(product_json: dict, excel_dict: list, REF_code: str) -> dict:
    """
    Fills the 'correct_text' field in the product_json dictionary using data from the matching REF row in excel_dict.

    Args:
        product_json (dict): Dictionary containing product references with 'excel_colomn_name' keys.
        excel_dict (list): List of dictionaries representing rows in an Excel file.
        REF_code (str): The REF code used to find the corresponding row in the Excel data.

    Returns:
        dict: Updated product_json with 'correct_text' fields filled.
    """
    # Find the matching row in the Excel data
    REF_excel_row = find_row_by_ref(excel_dict, REF_code)

    # Handle case when REF is not found
    if not REF_excel_row:
        print(f"⚠ Warning: REF '{REF_code}' not found in Excel data.")
        return product_json  # Return unchanged structure

    # Iterate over product JSON and fill 'correct_text' from the Excel row
    for key, value in product_json.items():
        excel_column_name = value.get("excel_colomn_name", "")
        if excel_column_name:  # If column name exists, fetch value from REF_excel_row
            value["correct_text"] = REF_excel_row.get(excel_column_name, "")

    return product_json  # Return updated structure


""" EXCEL """


def find_ref_row_in_excel(file_path: str, search_value: str) -> dict:
    """
    Opens an Excel file, finds a row with the given value in column A, and returns it as a dictionary.

    Args:
        file_path (str): Path to the Excel file (.xlsx or .xls).
        search_value (str): The value to search for in the first column (A).

    Returns:
        dict: The found row as a dictionary (keys = column names, values = corresponding data).
    """
    try:
        # Load the Excel file
        df = pd.read_excel(
            file_path, dtype=str
        )  # Load as text for more accurate searching

        # Check if the first column exists
        if df.empty or df.columns[0] is None:
            raise ValueError("The Excel file contains no valid data.")

        # Find the row with the matching value in column A (first column)
        row = df[df.iloc[:, 0] == search_value]

        # Check if a result was found
        if row.empty:
            print(f"❌ The value '{search_value}' was not found in the first column.")
            return {}

        # Convert the row to a dictionary
        result_dict = row.iloc[0].to_dict()

        return result_dict

    except Exception as e:
        print(f"⚠ Error processing the file: {e}")
        return {}


def load_excel_to_dict(file_path: str) -> list[dict]:
    """
    Loads an entire Excel file and converts it into a list of dictionaries.

    Args:
        file_path (str): Path to the Excel file (.xlsx or .xls).

    Returns:
        list[dict]: A list of dictionaries, where each dictionary corresponds to a row in the Excel file.
                    Keys = column names, values = corresponding row data.
    """
    try:
        # Load the Excel file into a pandas DataFrame
        df = pd.read_excel(
            file_path, dtype=str
        )  # Load as strings to preserve formatting

        # Check if the file is empty
        if df.empty:
            raise ValueError("❌ The Excel file contains no data.")

        # Convert the entire DataFrame to a list of dictionaries
        data_dict = df.to_dict(orient="records")

        return data_dict

    except Exception as e:
        print(f"⚠ Error loading the file: {e}")
        return []


def find_row_by_ref(excel_dict: list, ref_value: str) -> dict | None:
    """
    Searches for a row in the Excel dictionary where the 'REF' column matches ref_value.

    Args:
        excel_dict (list): List of dictionaries representing Excel rows.
        ref_value (str): Value to search for in the 'REF' column.

    Returns:
        dict | None: The matching row as a dictionary, or None if not found.
    """
    return next((row for row in excel_dict if row.get("REF") == ref_value), None)
