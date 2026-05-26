import sys
import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_owner_and_repo(url):
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url.strip())
    if match:
        owner = match.group(1)
        repo = match.group(2).replace(".git", "")
        return owner, repo
    return None, None


def fetch_repo_files(owner, repo, path="", files_data=None):
    if files_data is None:
        files_data = []

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {"User-Agent": "Repo-ChatBot"}

    response = requests.get(api_url, headers=headers)

    if response.status_code != 200:
        return files_data

    items = response.json()
    if not isinstance(items, list):
        items = [items]

    valid_extensions = ('.py', '.md', '.json', '.txt', '.js', '.ts',
                        '.html', '.css', '.go', '.cpp', '.h', '.yml', '.yaml')

    for item in items:
        if item['type'] == 'file' and item['name'].endswith(valid_extensions):
            raw_content = requests.get(item['download_url'], headers=headers).text

            files_data.append({
                "path": item['path'],
                "content": raw_content
            })

            print(f"Indexed: {item['path']}")

        elif item['type'] == 'dir':
            fetch_repo_files(owner, repo, item['path'], files_data)

    return files_data


# ---------------- CHATBOT RESPONSE LAYER ---------------- #

def interpret_score(score):
    if score > 0.3:
        return "Very High Relevance"
    elif score > 0.15:
        return "Moderate Relevance"
    elif score > 0.05:
        return "Low Relevance"
    else:
        return "Weak Match"


def format_response(query, results):
    output = []
    output.append("\n=========================================================")
    output.append("🤖 GITHUB REPO CHATBOT RESPONSE")
    output.append("=========================================================")
    output.append(f"🧠 Query: {query}")
    output.append("---------------------------------------------------------")

    for i, r in enumerate(results):
        path, score, content = r

        lines = content.split('\n')
        preview = '\n'.join(lines[:15])

        output.append(f"\n🏆 Rank #{i+1}")
        output.append(f"📁 File: {path}")
        output.append(f"📊 Similarity: {score:.2f} ({interpret_score(score)})")
        output.append("📄 Preview:")
        output.append(preview)

        if len(lines) > 15:
            output.append(f"... ({len(lines)-15} lines hidden)")

        output.append("---------------------------------------------------------")

    output.append("\n=========================================================\n")
    return "\n".join(output)


# ---------------- MAIN BOT ---------------- #

def start_github_bot():
    print("=========================================================")
    print("        GITHUB REPOSITORY CHATBOT (TF-IDF ENGINE)       ")
    print("=========================================================\n")

    repo_url = input("Enter GitHub Repo URL: ").strip()
    owner, repo = extract_owner_and_repo(repo_url)

    if not owner or not repo:
        print("Invalid GitHub URL.")
        return

    print(f"\nFetching repo: {owner}/{repo} ...\n")

    repo_files = fetch_repo_files(owner, repo)

    if not repo_files:
        print("No files found or API error.")
        return

    corpus = [f"{f['path']}\n{f['content']}" for f in repo_files]
    file_paths = [f['path'] for f in repo_files]
    file_contents = [f['content'] for f in repo_files]

    vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'(?u)\b\w+\b')
    tfidf_matrix = vectorizer.fit_transform(corpus)

    print(f"\nIndexed {len(repo_files)} files successfully!")
    print("Ask questions about the repository (type 'exit' to quit)\n")

    while True:
        try:
            user_query = input("Ask Repo: ").strip()

            if user_query.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                break

            if not user_query:
                continue

            query_vec = vectorizer.transform([user_query])
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

            # ---------------- TOP-K RESULTS ---------------- #
            top_k = 3
            top_indices = similarities.argsort()[-top_k:][::-1]

            results = []
            for idx in top_indices:
                score = similarities[idx]
                if score > 0.05:
                    results.append((
                        file_paths[idx],
                        score,
                        file_contents[idx]
                    ))

            if results:
                print(format_response(user_query, results))
            else:
                print("\n🤖 No relevant match found in this repository.\n")

        except (KeyboardInterrupt, EOFError):
            print("\nSession closed.")
            sys.exit()


if __name__ == "__main__":
    start_github_bot()