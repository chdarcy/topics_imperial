import pandas as pd
from pathlib import Path

class DataLoader:
    """Excel sheet loader """

    @staticmethod
    def load_sheet(xlsx_path: Path | str, sheet_name: str):
        xlsx_file = Path(xlsx_path)

        xls = pd.ExcelFile(xlsx_file, engine="openpyxl")
        return pd.read_excel(xls, sheet_name=sheet_name)
