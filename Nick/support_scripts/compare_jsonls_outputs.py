import json
import argparse

def compare_jsonl_values(file_paths, keys, show_filenames=True): #Added show_filenames parameter
    """Compares values in multiple JSONL files line by line."""
    num_files = len(file_paths)
    try:
        files = [open(file_path, 'r', encoding='utf-8') for file_path in file_paths]
        lines = [file.readlines() for file in files]

        max_lines = max(len(line_list) for line_list in lines)

        for i in range(max_lines):
            print("---")
            try:
                data = []
                for file_index in range(num_files):
                    if i < len(lines[file_index]):
                        try:
                            data.append(json.loads(lines[file_index][i]))
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON in file {file_paths[file_index]} line {i+1}: {lines[file_index][i].strip()}")
                            data.append(None)
                    else:
                        data.append(None)

                differences_found = False
                for key in keys:
                    values_for_key = [d.get(key) if d else None for d in data]

                    if not all(x == values_for_key[0] for x in values_for_key) and not all(x is None for x in values_for_key):
                        differences_found = True
                        print(f"Line {i+1}: Values for '{key}' are different:")
                        for file_index in range(num_files):
                            if values_for_key[file_index] is None:
                                output_string = f"Key '{key}' not found or file has less lines."
                            else:
                                output_string = f"{values_for_key[file_index]}"
                            if show_filenames: #Conditional printing of filenames
                                output_string = f"  {file_paths[file_index]}: {output_string}"
                            print(output_string)

                        break

                if not differences_found and not all(x is None for x in data):
                    print(f"Line {i+1}: Values for all specified keys are identical in all files.")

            except Exception as e:
                print("---")
                print(f"An unexpected error occurred in line {i+1}: {e}")

        for file_index in range(num_files):
            if i+1 < len(lines[file_index]):
                print("---")
                print(f"{file_paths[file_index]} has more lines. Remaining lines:")
                for j in range(i+1, len(lines[file_index])):
                    print(f"Line {j+1}: {lines[file_index][j].strip()}")

        for file in files:
            file.close()

    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare values in multiple JSONL files line by line for multiple keys.")
    parser.add_argument("files", nargs='+', help="Paths to the JSONL files (space-separated).")
    parser.add_argument("-k", "--keys", nargs='+', required=True, help="The keys to compare (space-separated).")
    parser.add_argument("-nf", "--no-filenames", action="store_false", dest="show_filenames", help="Don't show filenames in output.") #Added the argument
    args = parser.parse_args()

    compare_jsonl_values(args.files, args.keys, args.show_filenames) #Pass the show_filenames value
