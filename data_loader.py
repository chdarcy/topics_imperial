import pandas as pd
from pathlib import Path

class DataLoader:
    """Excel sheet loader """

    @staticmethod
    def load_sheet(xlsx_path: Path | str, sheet_name: str):
        xlsx_file = Path(xlsx_path)
        xls = pd.ExcelFile(xlsx_file, engine="openpyxl")
        df = pd.read_excel(xls, sheet_name=sheet_name)
        first_col = df.columns[0]
        parsed = pd.to_datetime(df[first_col], errors="coerce")
        if parsed.notna().sum() > 0 and parsed.notna().sum() >= len(df) // 2:
            df[first_col] = parsed
            df = df.set_index(first_col)

        return df
