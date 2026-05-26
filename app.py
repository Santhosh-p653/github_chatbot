import sys
import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_owner_and_repo(url):
    """Parses GitHub repository link to find the owner and repository names."""
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url.strip())
    if match:
        # Strip trailing .git if present
        owner = match.group(1)
        repo = match.group(2).replace(".git", "")
        return owner, repo
    return None, None

def fetch_repo_files(owner, repo, path="", files_data=None):
    """Recursively fetches all text/code files from a public repository via GitHub API."""
    if files_data is None:
        files_data = []

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    # Standard user-agent headers to keep GitHub's API routing happy
    headers = {"User-Agent": "Cosine-Similarity-Repo-Bot"}
    
    response = requests.get(api_url, headers=headers)
    
    if response.status_code != 200:
        return files_data

    items = response.json()
    # If the repository is completely empty or pointing straight to a single file edge-case
    if not isinstance(items, list):
        items = [items]

    # Extensions we want to scrape and index for text matching
    valid_extensions = ('.py', '.md', '.json', '.txt', '.js', '.ts', '.html', '.css', '.go', '.cpp', '.h', '.yml', '.yaml')

    for item in items:
        if item['type'] == 'file' and item['name'].endswith(valid_extensions):
            # Fetch the raw code/text directly using GitHub's distributed download URL
            raw_content = requests.get(item['download_url'], headers=headers).text
            files_data.append({
                "path": item['path'],
                "content": raw_content
            })
            print(f" -> Indexed: {item['path']}")
        elif item['type'] == 'dir':
            # Drill deeper down the repository directory tree
            fetch_repo_files(owner, repo, item['path'], files_data)
            
    return files_data

def start_github_bot():
    print("=========================================================")
    print("       GITHUB REPOSITORY NLP CLI INTERACTIVE CHATBOT     ")
    print("=========================================================\n")
    
    repo_url = input("Enter Public GitHub Repo URL (e.g., https://github.com/user/repo): ").strip()
    owner, repo = extract_owner_and_repo(repo_url)
    
    if not owner or not repo:
        print("Error: Invalid GitHub URL structure.")
        return

    print(f"\nFetching and analyzing code vectors from '{owner}/{repo}'... Hang tight.")
    repo_files = fetch_repo_files(owner, repo)
    
    if not repo_files:
        print("No readable text/code files found or API rate limit hit.")
        return

    # Extract clean file text segments for our NLP vector matrix
    corpus = [f"File: {f['path']}\n\nContents:\n{f['content']}" for f in repo_files]
    file_paths = [f['path'] for f in repo_files]
    file_contents = [f['content'] for f in repo_files]

    # Initialize TF-IDF Vectorizer
    # We strip common english terms and analyze custom code structural tokens via alphanumeric n-grams
    vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'(?u)\b\w+\b')
    tfidf_matrix = vectorizer.fit_transform(corpus)

    print(f"\nSuccessfully indexed {len(repo_files)} structural code/text assets!")
    print("You can now ask questions about files, functions, or README instructions.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_query = input("Ask about Repo: ").strip()
            if user_query.lower() in ['exit', 'quit', 'bye']:
                print("Exiting Session. Happy coding!")
                break
            
            if not user_query:
                continue

            # Project user string into the repository's vector dimension space
            query_vector = vectorizer.transform([user_query])
            similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
            
            best_match_idx = similarities.argsort()[-1]
            highest_score = similarities[best_match_idx]

            # Threshold configurations
            if highest_score > 0.05:
                matched_path = file_paths[best_match_idx]
                matched_text = file_contents[best_match_idx]
                
                print(f"\n[Bot]: Found most relevant match in: {matched_path} (Confidence Similarity: {highest_score:.2f})")
                print("---------------------------------------------------------")
                
                # Truncate content preview so it doesn't flood the terminal CLI
                lines = matched_text.split('\n')
                preview = '\n'.join(lines[:25])
                print(preview)
                if len(lines) > 25:
                    print(f"\n... [{len(lines)-25} structural code lines truncated. Open the file to read in full.]")
                print("---------------------------------------------------------\n")
            else:
                print("\n[Bot]: I couldn't find code snippets or documentation text matching that query inside this repository.\n")

        except (KeyboardInterrupt, EOFError):
            print("\nSession killed. Goodbye!")
            sys.exit()

if __name__ == "__main__":
    start_github_bot()
