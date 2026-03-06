from pathlib import Path

import typer


def main(
	data: Path = typer.Option(..., "--data", "-d", help="Path to input data file"),
	outputpath: Path = typer.Option(
		Path("output/predictions.tsv"),
		"--outputpath",
		"--outputfolder",
		help="Path to output file (default: output/predictions.tsv)",
	),
) -> None:
	outputpath.parent.mkdir(parents=True, exist_ok=True)
	typer.echo(f"Input data file: {data} exists") \
		if data.exists() else typer.echo("Input data file does not exist.")
	typer.echo(f"Output data file: {outputpath} exists") \
		if outputpath.exists() else typer.echo("Output data file does not exists, will be created.")
	typer.echo("Entry point is ready. Connect your modeling pipeline here.")
	
	print("This is a template for a Python project using Typer for command-line interfaces.")
	print("Hello world!")

if __name__ == "__main__":
	typer.run(main)
