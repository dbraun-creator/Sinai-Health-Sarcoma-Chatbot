import pandas as pd
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings_batch(texts, model="text-embedding-3-small"):
    """
    Get embeddings for a batch of texts using OpenAI's embedding model.
    OpenAI API supports up to 2048 inputs per request for embedding models.
    
    Args:
        texts: List of texts to embed
        model: The embedding model to use
    
    Returns:
        List of embeddings (same order as input) or None for failed items
    """
    try:
        # Clean the texts - replace newlines and ensure they're strings
        cleaned_texts = [str(text).replace("\n", " ") for text in texts]
        
        # Call OpenAI API with batch request
        response = client.embeddings.create(
            input=cleaned_texts,
            model=model
        )
        
        # Extract embeddings in order
        embeddings = [item.embedding for item in response.data]
        return embeddings
    
    except Exception as e:
        print(f"Error getting embeddings for batch: {e}")
        return [None] * len(texts)

def process_csv_with_embeddings(input_path, batch_size=50):
    """
    Process a CSV file to add embeddings for the 'questions' column.
    Uses batch processing and incremental writing to handle large files.
    KEEPS ALL ORIGINAL COLUMNS and adds 'embeddings' column.
    
    Args:
        input_path: Path to the input CSV file
        batch_size: Number of rows to process at once (default: 50)
    """
    # Read the input CSV
    print(f"Reading CSV file: {input_path}")
    df = pd.read_csv(input_path)
    
    # Verify 'questions' column exists
    if 'questions' not in df.columns:
        raise ValueError("Input CSV must have a 'questions' column")
    
    # Generate output filename first
    input_file = Path(input_path)
    output_filename = f"{input_file.stem}_with_embd{input_file.suffix}"
    output_path = input_file.parent / output_filename
    
    # Add 'embeddings' column to the existing dataframe (keeps all other columns)
    df['embeddings'] = None
    
    # Counters for success and failure
    success_count = 0
    failure_count = 0
    total_rows = len(df)
    
    print(f"\nProcessing {total_rows} rows in batches of {batch_size}...")
    print(f"Keeping all {len(df.columns)} columns: {list(df.columns)}")
    
    # Process in batches
    for batch_start in range(0, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        batch_num = (batch_start // batch_size) + 1
        total_batches = (total_rows + batch_size - 1) // batch_size
        
        print(f"\nProcessing batch {batch_num}/{total_batches} (rows {batch_start + 1}-{batch_end})...")
        
        # Get questions for this batch
        batch_questions = []
        batch_indices = []
        
        for idx in range(batch_start, batch_end):
            question = df.at[idx, 'questions']
            
            # Skip empty or NaN values
            if pd.isna(question) or str(question).strip() == '':
                print(f"  Row {idx + 1}: Skipping empty question")
                failure_count += 1
            else:
                batch_questions.append(question)
                batch_indices.append(idx)
        
        # Get embeddings for the batch
        if batch_questions:
            embeddings = get_embeddings_batch(batch_questions)
            
            # Store embeddings (this only modifies the 'embeddings' column)
            for idx, embedding in zip(batch_indices, embeddings):
                if embedding is not None:
                    df.at[idx, 'embeddings'] = str(embedding)
                    success_count += 1
                else:
                    failure_count += 1
            
            print(f"  Processed {len(batch_questions)} questions successfully")
        
        # Small delay between batches to respect rate limits
        if batch_end < total_rows:
            time.sleep(0.5)
        
        # Write progress to disk incrementally (optional, for safety)
        # This ensures we don't lose all progress if something fails
        if batch_num % 3 == 0 or batch_end == total_rows:
            print(f"  Saving progress...")
            df.to_csv(output_path, index=False)
    
    # Final save
    print(f"\nSaving final results to: {output_path}")
    df.to_csv(output_path, index=False)
    
    # Display summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total rows processed: {total_rows}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {failure_count}")
    print(f"Total columns in output: {len(df.columns)}")
    print(f"Output file: {output_path}")
    print("="*50)
    
    return output_path

def main():
    """Main function to run the script."""
    print("OpenAI Text Embedding Generator (Batch Processing)")
    print("="*50)
    
    # Get input file path from user
    input_path = input("\nEnter the path to your input CSV file: ").strip()
    
    # Remove quotes if user wrapped the path in quotes
    input_path = input_path.strip('"').strip("'")
    
    # Validate file exists
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return
    
    # Validate file is a CSV
    if not input_path.lower().endswith('.csv'):
        print("Error: Input file must be a CSV file")
        return
    
    # Optional: let user specify batch size
    batch_size = 50  # Default batch size
    use_custom_batch = input(f"\nUse default batch size of {batch_size}? (y/n): ").strip().lower()
    if use_custom_batch == 'n':
        try:
            batch_size = int(input("Enter batch size (recommended: 20-100): ").strip())
        except ValueError:
            print("Invalid input. Using default batch size of 50.")
            batch_size = 50
    
    try:
        # Process the CSV
        output_path = process_csv_with_embeddings(input_path, batch_size)
        print(f"\n✓ Processing complete! Check your output file at:\n  {output_path}")
    
    except Exception as e:
        print(f"\n✗ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()