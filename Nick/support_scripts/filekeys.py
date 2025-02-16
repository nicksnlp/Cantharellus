import json
import argparse

def process_jsonl(filepath, keys):
    """
    Opens a JSONL file, reads it line by line, and prints the values 
    associated with specified keys, separated by '--'.

    Args:
        filepath: The path to the JSONL file.
        keys: A list of keys to extract from each JSON object.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    data = json.loads(line.strip())
                    output = []
                    for key in keys:
                        if key in data:
                            output.append(str(data[key])) #ensure all values are strings for joining
                        else:
                            output.append(f"Key '{key}' not found")
                    print("\n".join(output), end="\n---\n") # Join the collected values with "--"
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {line_number}: {e}", end="--")
                except TypeError as e:
                    print(f"TypeError on line {line_number}: {e}. Line Content: {line.strip()}", end="--")
        print() # Add newline at the end

    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
    except IOError as e:
        print(f"Error reading file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSONL file and print specified keys, separated by '--'.")
    parser.add_argument("filepath", help="Path to the JSONL file")
    parser.add_argument("keys", nargs='+', help="Keys to extract (space-separated)") #nargs='+' to allow multiple keys
    args = parser.parse_args()

    process_jsonl(args.filepath, args.keys)