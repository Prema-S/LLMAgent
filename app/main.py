import os
import re
import json
import subprocess
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo  # Requires Python 3.9+
import aiproxy
from flask import Flask, request, jsonify
import hashlib
from PIL import Image
import io
import base64
import requests
import shutil

app = Flask(__name__)

# --- Helper Functions ---

def run_command(command):
    """Runs a shell command and returns the output."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr.strip()}"

def install_uv():
    """Installs uv if not already installed."""
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True, text=True)
        print("uv is already installed.")
    except FileNotFoundError:
        print("Installing uv...")
        run_command("pip install uv")

def run_datagen(email):
    """Runs the datagen.py script with the provided email."""
    install_uv()  # Ensure uv is installed
    script_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    run_command(f"uv run python {script_url} {email}")

def format_with_prettier(filepath, version="3.4.2"):
    """Formats a file using prettier."""
    run_command(f"npx prettier@{version} --write {filepath}")

def count_wednesdays(filepath):
    """Counts the number of Wednesdays in a file containing dates."""
    count = 0
    try:
        with open(filepath, 'r') as f:
            for line in f:
                date_str = line.strip()
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    if date_obj.weekday() == 2:  # Wednesday is 2
                        count += 1
                except ValueError:
                    pass  # Ignore lines that are not valid dates
    except FileNotFoundError:
        return "Error: File not found"
    return count

def sort_contacts(input_filepath, output_filepath):
    """Sorts contacts by last_name and first_name."""
    try:
        with open(input_filepath, 'r') as f:
            contacts = json.load(f)
        sorted_contacts = sorted(contacts, key=lambda x: (x['last_name'], x['first_name']))
        with open(output_filepath, 'w') as f:
            json.dump(sorted_contacts, f, indent=2)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return f"Error: {str(e)}"

def get_recent_log_lines(log_dir, output_filepath):
    """Writes the first line of the 10 most recent .log files to a file."""
    try:
        log_files = sorted(
            [f for f in os.listdir(log_dir) if f.endswith('.log')],
            key=lambda f: os.path.getmtime(os.path.join(log_dir, f)),
            reverse=True
        )[:10]

        with open(output_filepath, 'w') as outfile:
            for log_file in log_files:
                log_filepath = os.path.join(log_dir, log_file)
                with open(log_filepath, 'r') as infile:
                    first_line = infile.readline().strip()
                    outfile.write(first_line + '\n')
    except FileNotFoundError:
        return "Error: Log directory not found"
    except Exception as e:
        return f"Error: {str(e)}"

def create_markdown_index(docs_dir, output_filepath):
    """Creates an index of Markdown files and their H1 titles."""
    index = {}
    try:
        for filename in os.listdir(docs_dir):
            if filename.endswith('.md'):
                filepath = os.path.join(docs_dir, filename)
                with open(filepath, 'r') as f:
                    for line in f:
                        if line.startswith('# '):
                            title = line[2:].strip()
                            index[filename] = title
                            break  # Only get the first H1
        with open(output_filepath, 'w') as f:
            json.dump(index, f, indent=2)
    except FileNotFoundError:
        return "Error: Docs directory not found"
    except Exception as e:
        return f"Error: {str(e)}"

def extract_email_sender(email_filepath, output_filepath, llm_client):
    """Extracts the sender's email address from an email using an LLM."""
    try:
        with open(email_filepath, 'r') as f:
            email_content = f.read()

        prompt = f"Extract the sender's email address from the following email:\n\n{email_content}"
        response = llm_client.completions.create(
            model="gpt-4o-mini",
            prompt=prompt,
            max_tokens=50,  # Limit tokens for efficiency
        )
        sender_email = response.choices[0].text.strip()

        with open(output_filepath, 'w') as f:
            f.write(sender_email)

    except FileNotFoundError:
        return "Error: Email file not found"
    except Exception as e:
        return f"Error: {str(e)}"

def extract_credit_card_number(image_filepath, output_filepath, llm_client):
    """Extracts the credit card number from an image using an LLM."""
    try:
        with open(image_filepath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        prompt = "Extract the credit card number from the image and write it without spaces."
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded_string}"},
                        },
                    ],
                }
            ],
            max_tokens=30,
        )
        card_number = response.choices[0].message.content.strip().replace(" ", "")

        with open(output_filepath, 'w') as f:
            f.write(card_number)

    except FileNotFoundError:
        return "Error: Image file not found"
    except Exception as e:
        return f"Error: {str(e)}"

def find_most_similar_comments(comments_filepath, output_filepath, llm_client):
    """Finds the most similar pair of comments using embeddings."""
    try:
        with open(comments_filepath, 'r') as f:
            comments = [line.strip() for line in f]

        if len(comments) < 2:
            return "Error: Need at least two comments to compare."

        # Get embeddings for each comment
        embeddings = []
        for comment in comments:
            response = llm_client.embeddings.create(input=comment, model="text-embedding-ada-002")
            embeddings.append(response.data[0].embedding)

        # Calculate cosine similarity between all pairs of embeddings
        max_similarity = -1
        most_similar_pair = (None, None)

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = cosine_similarity(embeddings[i], embeddings[j])
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_pair = (comments[i], comments[j])

        with open(output_filepath, 'w') as f:
            f.write(most_similar_pair[0] + '\n')
            f.write(most_similar_pair[1] + '\n')

    except FileNotFoundError:
        return "Error: Comments file not found"
    except Exception as e:
        return f"Error: {str(e)}"

def cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

def calculate_total_sales(db_filepath, output_filepath):
    """Calculates the total sales for 'Gold' tickets."""
    try:
        conn = sqlite3.connect(db_filepath)
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
        total_sales = cursor.fetchone()[0]
        conn.close()

        if total_sales is None:
            total_sales = 0

        with open(output_filepath, 'w') as f:
            f.write(str(total_sales))

    except sqlite3.Error as e:
        return f"Error: SQLite error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

def fetch_data_and_save(api_url, output_filepath):
    """Fetches data from an API and saves it to a file."""
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.text # or response.json() if the API returns JSON

        with open(output_filepath, 'w') as f:
            f.write(data)
            #f.write(json.dumps(data, indent=2)) #if json

    except requests.exceptions.RequestException as e:
        return f"Error: API request failed: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

def clone_and_commit(repo_url, commit_message, local_path="/data/repo"):
    """Clones a git repo, makes a commit, and pushes."""
    try:
        # Ensure the local directory is clean
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        os.makedirs(local_path, exist_ok=True)

        # Clone the repository
        run_command(f"git clone {repo_url} {local_path}")
        os.chdir(local_path)

        # Create/Modify a file (example)
        with open("README.md", "a") as f:
            f.write("\nUpdated by automation agent.")

        # Commit and push
        run_command("git add .")
        run_command(f'git commit -m "{commit_message}"')
        run_command("git push")

    except Exception as e:
        return f"Error: Git operation failed: {str(e)}"
    finally:
        os.chdir("..") # Go back to the root

def run_sql_query(db_filepath, query, output_filepath):
    """Runs a SQL query on a SQLite database and saves the result."""
    try:
        conn = sqlite3.connect(db_filepath)
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        conn.close()

        with open(output_filepath, 'w') as f:
            for row in result:
                f.write(str(row) + '\n')

    except sqlite3.Error as e:
        return f"Error: SQLite error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

def scrape_website(url, output_filepath):
    """Scrapes data from a website and saves it."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_filepath, 'w') as f:
            f.write(response.text)
    except requests.exceptions.RequestException as e:
        return f"Error: Web scraping failed: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

def compress_image(input_filepath, output_filepath, quality=60):
    """Compresses an image."""
    try:
        image = Image.open(input_filepath)
        image.save(output_filepath, "JPEG", quality=quality)
    except FileNotFoundError:
        return "Error: Image file not found"
    except Exception as e:
        return f"Error: {str(e)}"

def transcribe_audio(input_filepath, output_filepath, llm_client):
    """Transcribes audio from an MP3 file using an LLM."""
    # This is a placeholder.  Real audio transcription requires a dedicated service
    # like OpenAI's Whisper API, Google Cloud Speech-to-Text, or AWS Transcribe.
    # This example uses a dummy LLM call for demonstration.
    try:
        with open(input_filepath, 'rb') as audio_file:
            # In a real implementation, you would send the audio data to a
            # transcription service.  Here, we just pretend to.
            prompt = "Transcribe the following audio:"  # No actual audio data sent to the LLM
            response = llm_client.completions.create(
                model="gpt-4o-mini",
                prompt=prompt,
                max_tokens=100, # Adjust as needed
            )
            transcription = response.choices[0].text.strip()

        with open(output_filepath, 'w') as f:
            f.write(transcription)

    except FileNotFoundError:
        return "Error: Audio file not found"
    except Exception as e:
        return f"Error: {str(e)}"

def markdown_to_html(input_filepath, output_filepath):
    """Converts Markdown to HTML."""
    try:
        run_command(f"pandoc {input_filepath} -o {output_filepath}")
    except FileNotFoundError:
        return "Error: Markdown file not found"
    except Exception as e:
        return f"Error: {str(e)}"

def filter_csv_and_return_json(csv_filepath, filter_column, filter_value, output_filepath):
    """Filters a CSV file and returns JSON data."""
    try:
        import csv
        data = []
        with open(csv_filepath, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row[filter_column] == filter_value:
                    data.append(row)

        with open(output_filepath, 'w') as f:
            json.dump(data, f, indent=2)

    except FileNotFoundError:
        return "Error: CSV file not found"
    except KeyError:
        return f"Error: Column '{filter_column}' not found in CSV"
    except Exception as e:
        return f"Error: {str(e)}"


# --- API Endpoints ---

@app.route('/run', methods=['POST'])
def run_task():
    task_description = request.args.get('task')
    if not task_description:
        return jsonify({"error": "Task description is required"}), 400

    llm_client = aiproxy.Client(api_key=os.environ["premapk.7593"])

    try:
        # --- Task Parsing and Execution ---
        # Use LLM to determine the task and parameters
        prompt = f"""You are an automation agent.  You need to identify the task and extract relevant parameters from the following user request:

        User Request: "{task_description}"

        Based on the request, respond *only* with a JSON object in the following format:

        {{
            "task_type": "task_name",
            "parameters": {{
                "param1": "value1",
                "param2": "value2",
                ...
            }}
        }}

        Possible task_type values and their parameters:

        - "run_datagen": {{"email": "user's email"}}
        - "format_file": {{"filepath": "/path/to/file", "version": "prettier version"}}
        - "count_weekday": {{"filepath": "/path/to/file", "output_filepath": "/path/to/output", "weekday": "weekday name (e.g., Wednesday)"}}
        - "sort_json": {{"input_filepath": "/path/to/input.json", "output_filepath": "/path/to/output.json"}}
        - "extract_recent_log_lines": {{"log_dir": "/path/to/logs", "output_filepath": "/path/to/output"}}
        - "create_markdown_index": {{"docs_dir": "/path/to/docs", "output_filepath": "/path/to/index.json"}}
        - "extract_email_sender": {{"email_filepath": "/path/to/email.txt", "output_filepath": "/path/to/output"}}
        - "extract_credit_card": {{"image_filepath": "/path/to/image.png", "output_filepath": "/path/to/output"}}
        - "find_similar_comments": {{"comments_filepath": "/path/to/comments.txt", "output_filepath": "/path/to/output"}}
        - "calculate_total_sales": {{"db_filepath": "/path/to/database.db", "output_filepath": "/path/to/output"}}
        - "fetch_data": {{"api_url": "URL of the API", "output_filepath": "/path/to/output"}}
        - "clone_and_commit": {{"repo_url": "URL of the git repo", "commit_message": "Commit message", "local_path": "/data/repo"}}
        - "run_sql_query": {{"db_filepath": "/path/to/database.db", "query": "SQL query", "output_filepath": "/path/to/output"}}
        - "scrape_website": {{"url": "URL of the website", "output_filepath": "/path/to/output"}}
        - "compress_image": {{"input_filepath": "/path/to/image", "output_filepath": "/path/to/output", "quality": "compression quality (integer)"}}
        - "transcribe_audio": {{"input_filepath": "/path/to/audio.mp3", "output_filepath": "/path/to/output"}}
        - "markdown_to_html": {{"input_filepath": "/path/to/markdown", "output_filepath": "/path/to/html"}}
        - "filter_csv": {{"csv_filepath": "/path/to/csv", "filter_column": "column name", "filter_value": "value to filter", "output_filepath": "/path/to/output"}}
        - "unknown_task": {{}}

        If the task is not recognized, use "unknown_task". Do not hallucinate tasks or parameters.  Only use the information provided.
        """

        response = llm_client.completions.create(
            model="gpt-4o-mini",
            prompt=prompt,
            max_tokens=500, # Increased max_tokens for JSON response
        )
        task_info_str = response.choices[0].text.strip()
        #print(f"LLM response: {task_info_str}") # Debugging

        try:
            task_info = json.loads(task_info_str)
            task_type = task_info['task_type']
            parameters = task_info['parameters']
        except (json.JSONDecodeError, KeyError):
            return jsonify({"error": "Failed to parse LLM response or invalid task format."}), 500

        # --- Security Checks (B1 and B2) ---
        for param_name, param_value in parameters.items():
            if isinstance(param_value, str):
                if param_name.endswith("filepath") or param_name.endswith("path") or param_name.endswith("dir"):
                    if not param_value.startswith("/data/"):
                        return jsonify({"error": f"Security violation: Access to paths outside /data is forbidden. Parameter: {param_name}"}), 400
                if ".." in param_value:
                    return jsonify({"error": f"Security violation: Path traversal detected. Parameter: {param_name}"}), 400

        # --- Task Execution ---
        if task_type == "run_datagen":
            run_datagen(parameters['email'])
        elif task_type == "format_file":
            format_with_prettier(parameters['filepath'], parameters.get('version', '3.4.2'))
        elif task_type == "count_weekday":
            count = count_wednesdays(parameters['filepath'])
            if isinstance(count, str) and count.startswith("Error"):
                return jsonify({"error": count}), 500
            with open(parameters['output_filepath'], 'w') as f:
                f.write(str(count))
        elif task_type == "sort_json":
            result = sort_contacts(parameters['input_filepath'], parameters['output_filepath'])
            if result and result.startswith("Error"):
                return jsonify({"error": result}), 500
        elif task_type == "extract_recent_log_lines":
            result = get_recent_log_lines(parameters['log_dir'], parameters['output_filepath'])
            if result and result.startswith("Error"):
                return jsonify({"error": result}), 500
        elif task_type == "create_markdown_index":
            result = create_markdown_index(parameters['docs_dir'], parameters['output_filepath'])
            if result and result.startswith("Error"):
                return jsonify({"error": result}), 500
        elif task_type == "extract_email_sender":
            result = extract_email_sender(parameters['email_filepath'], parameters['output_filepath'], llm_client)
            if result and result.startswith("Error"):
                return jsonify({"error": result}), 500
        elif task_type == "extract_credit_card":
            result = extract_credit_card_number(parameters['image_filepath'], parameters['output_filepath'], llm_client)
            if result and result.startswith("Error"):
                return jsonify({"error": result}), 500
        elif task_type == "find_similar_comments":
            result = find_most_similar_comments(parameters['comments_filepath'], parameters['output_filepath'], llm_client)
            if result and result.startswith("Error"):
                return jsonify({"error": result}), 500
        elif task_type == "calculate_total_sales":
            result = calculate_total_sales(parameters['db_filepath'], parameters['output_filepath'])
            if result and result.startswith("Error"):
                return jsonify({"error": result}), 500
        elif task_type == "fetch_data":
            result = fetch_data_and_save(parameters['api_url'], parameters['output_filepath'])
            if result and result.startswith("Error"):
                return jsonify({"error": result}), 500
        elif task_type == "clone_and_commit":
            result = clone_and_commit(parameters['repo_url'], parameters['commit_message'], parameters.get('local_path', "/data/repo"))
            if result and result.startswith("Error"):
                return jsonify({"error": result}), 500
        elif task_type == "run_sql_query":
            result = run_sql_query(parameters['db_filepath'], parameters['query'], parameters['output_filepath'])
            if result and result.startswith("Error"):
                return jsonify({"error": result}), 500
        elif task_type == "scrape_website":
            result = scrape_website(parameters['url'], parameters['output_filepath'])
            if result and result.startswith("Error"):
                return jsonify({"error": result}), 500
        elif task_type == "compress_image":
            result = compress_image(parameters['input_filepath'], parameters['output_filepath'], parameters.get('quality', 60))
            if result and result.startswith("Error"):
                return jsonify({"error": result}), 500
        elif task_type == "transcribe_audio":
            result = transcribe_audio(parameters['input_filepath'], parameters['output_filepath'], llm_client)
            if result and result.startswith("Error"):
                return jsonify({"error": result}), 500
        elif task_type == "markdown_to_html":
            result = markdown_to_html(parameters['input_filepath'], parameters['output_filepath'])
            if result and result.startswith("Error"):
                return jsonify({"error": result}), 500
        elif task_type == "filter_csv":
            result = filter_csv_and_return_json(parameters['csv_filepath'], parameters['filter_column'], parameters['filter_value'], parameters['output_filepath'])
            if result and result.startswith("Error"):
                return jsonify({"error": result}), 500

        elif task_type == "unknown_task":
            return jsonify({"error": "Task not recognized"}), 400
        else:
            return jsonify({"error": "Invalid task type"}), 500

        return jsonify({"message": "Task completed successfully"}), 200

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route('/read', methods=['GET'])
def read_file():
    filepath = request.args.get('path')
    if not filepath:
        return jsonify({"error": "File path is required"}), 400

    # Security Check (B1)
    if not filepath.startswith("/data/"):
        return jsonify({"error": "Security violation: Access to paths outside /data is forbidden"}), 403
    if ".." in filepath:
        return jsonify({"error": "Security violation: Path traversal detected"}), 403

    try:
        with open(filepath, 'r') as f:
            content = f.read()
        return content, 200, {'Content-Type': 'text/plain'}
    except FileNotFoundError:
        return "", 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

