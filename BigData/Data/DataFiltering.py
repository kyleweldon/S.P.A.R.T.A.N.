#############################################
# File created by Kyle Weldon on 06/16/2024 #
#############################################

################# Needs Doing list for entire project ##############
# None
####################################################################


import pandas as pd

# Function to check if a row contains any empty cells
def is_row_complete(row):
    for cell in row:
        # Check if cell is NaN or empty (after stripping whitespace)
        if pd.isna(cell) or str(cell).strip() == '':
            return False
    return True

def main():
    # Step 1: Read the Excel file into a pandas DataFrame
    excel_file = 'RawData.xlsx'  # Replace with your Excel file path
    output_csv = 'FilteredData.csv'  # Replace with your desired CSV output file path

    try:
        df = pd.read_excel(excel_file)
    except FileNotFoundError:
        print(f"Error: The file '{excel_file}' was not found.")
        return
    except Exception as e:
        print(f"Error occurred while reading '{excel_file}': {str(e)}")
        return

    # Step 2: Filter rows based on completeness (non-empty cells)
    complete_rows = []
    for index, row in df.iterrows():
        if is_row_complete(row):
            complete_rows.append(row)

    cleaned_df = pd.DataFrame(complete_rows, columns=df.columns)

    # Step 3: Save the cleaned data to a CSV file
    try:
        cleaned_df.to_csv(output_csv, index=False)
        print(f"Cleaned data saved to '{output_csv}' successfully.")
    except Exception as e:
        print(f"Error occurred while saving to '{output_csv}': {str(e)}")
        return

if __name__ == "__main__":
    main()
