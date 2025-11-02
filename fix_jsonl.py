#!/usr/bin/env python3
"""
Fix JSONL file with malformed JSON lines.

This script reads a JSONL file, attempts to parse each line,
and writes a properly formatted JSONL file. Lines that can't be fixed are skipped.
"""

import json
import sys
import argparse
from pathlib import Path

def fix_jsonl_line(line: str) -> dict:
    """Attempt to fix a single JSONL line."""
    line = line.strip()
    if not line:
        return None
    
    # Try direct parsing first
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        pass
    
    # Try removing control characters
    try:
        cleaned = ''.join(c if ord(c) >= 32 or c in '\n\t' else ' ' for c in line)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Try to reconstruct JSON by finding the structure
    # This is a heuristic approach - look for opening brace and try to find matching closing
    try:
        # Find the start of the JSON object
        start_idx = line.find('{')
        if start_idx == -1:
            return None
        
        # Find the end by counting braces
        brace_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(line)):
            if line[i] == '{':
                brace_count += 1
            elif line[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if brace_count == 0:
            json_str = line[start_idx:end_idx]
            return json.loads(json_str)
    except:
        pass
    
    return None

def fix_jsonl_file(input_path: Path, output_path: Path):
    """Fix a JSONL file line by line."""
    fixed = 0
    skipped = 0
    errors = []
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, start=1):
            obj = fix_jsonl_line(line)
            
            if obj is not None:
                # Re-encode properly to ensure valid JSON
                outfile.write(json.dumps(obj, ensure_ascii=False) + '\n')
                fixed += 1
            else:
                skipped += 1
                errors.append(line_num)
                if skipped <= 10:
                    print(f"Warning: Skipping malformed line {line_num}: {line[:100]}...", file=sys.stderr)
    
    print(f"Fixed {fixed} lines, skipped {skipped} lines")
    if errors:
        print(f"Lines with errors: {errors[:20]}..." if len(errors) > 20 else f"Lines with errors: {errors}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Fix malformed JSONL file")
    parser.add_argument("input", type=Path, help="Input JSONL file")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output JSONL file (default: input_fixed.jsonl)")
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist", file=sys.stderr)
        sys.exit(1)
    
    output_path = args.output or args.input.parent / f"{args.input.stem}_fixed.jsonl"
    
    fix_jsonl_file(args.input, output_path)
    print(f"Output written to {output_path}")

if __name__ == "__main__":
    main()

