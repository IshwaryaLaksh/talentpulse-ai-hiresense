
import openpyxl
import os

# 1. Define the dataset location
file_path = os.path.join(
    "..", "data", "raw", "candidate_dataset.xlsx"
)

# 2. Check whether the file exists
if not os.path.exists(file_path):
    print("Dataset not found. Check the file location.")
else:

    print("Dataset found successfully!")

    # 3. Open Excel in read-only mode
    workbook = openpyxl.load_workbook(
        file_path,
        read_only=True,
        data_only=True
    )

    # 4. Display worksheet names
    print("\nWORKSHEETS AVAILABLE:")

    for sheet_name in workbook.sheetnames:
        print("-", sheet_name)

    # 5. Inspect each worksheet
    for sheet in workbook.worksheets:

        print("\n------------------------------")
        print("Worksheet:", sheet.title)
        print("Approximate rows:", sheet.max_row)
        print("Approximate columns:", sheet.max_column)

        # 6. Read column headers
        first_row = next(
            sheet.iter_rows(min_row=1, max_row=1, values_only=True)
        )

        print("\nCOLUMN HEADERS:")

        for index, column in enumerate(first_row, start=1):
            print(index, ":", column)

    workbook.close()

    print("\nDataset inspection completed!")
