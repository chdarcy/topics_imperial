from pathlib import Path
import pandas as pd

from data_loader import DataLoader
from curves import interpolate_to_grid


def main():
	base = Path(__file__).resolve().parent
	
	# Load the dataframe
	df = DataLoader.load_sheet(xlsx_path=base / "input.xlsx", sheet_name="gbp ois results")

	# Check if data loaded correctly
	print("*******************Raw Loaded Data*******************")
	print(df.shape)
	print(df.head())
	print()
	
	# interpolate and capture result
	print("*******************Interpolated Data*******************")
	interpolated = interpolate_to_grid(df)
	print("Interpolated shape:", interpolated.shape)
	print(interpolated.head())
	



if __name__ == "__main__":
	main()

