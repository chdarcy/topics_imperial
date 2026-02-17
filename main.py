from pathlib import Path

from data_loader import DataLoader


def main():
	base = Path(__file__).resolve().parent
	
	# Load the dataframe
	df = DataLoader.load_sheet(xlsx_path=base / "input.xlsx", sheet_name="gbp ois results")

    # Check if data loaded correctly
	print(df.shape)
	#print(df.head().to_string(index=False))


if __name__ == "__main__":
	main()

