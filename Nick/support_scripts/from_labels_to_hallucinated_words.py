import json
import argparse

### THIS SCRIPT IS NOT CORRECT

def process_jsonl(input_filepath, output_filepath):
    """
    Processes a JSONL file, extracts hallucinated words based on hard labels,
    and writes the processed data to a new JSONL file.

    Args:
        input_filepath: Path to the input JSONL file.
        output_filepath: Path to the output JSONL file.

    Returns:
        None. Prints error messages to the console if any occur.
    """

    results = []
    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    data = json.loads(line)
                    hard_labels = data.get("hard_labels", [])
                    model_output_tokens = data.get("model_output_tokens", [])
                    model_input = data.get("model_input", "")

                    hallucinated_words = []
                    for start, end in hard_labels:
                        try:
                            word = "".join(model_output_tokens[start:end]).replace("\u0120", " ")
                            if word.lower() not in model_input.lower():
                                hallucinated_words.append(word)
                        except IndexError:
                            print(f"IndexError: Start: {start}, End: {end}, Tokens Length: {len(model_output_tokens)}")
                            hallucinated_words.append("Error: Index out of range")

                    data["hallucinated_words"] = hallucinated_words
                    results.append(data)

                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}")

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_filepath}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    try:
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            for item in results:
                json.dump(item, outfile, ensure_ascii=False)
                outfile.write('\n')  # Crucial for JSONL format
    except Exception as e:
        print(f"Error writing to output file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL file to extract hallucinated words.")
    parser.add_argument("input_file", help="Path to the input JSONL file.")
    parser.add_argument("output_file", help="Path to the output JSONL file.")
    args = parser.parse_args()

    process_jsonl(args.input_file, args.output_file)
    print(f"Processing complete. Output written to {args.output_file}")