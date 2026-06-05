import os
import requests
from colorama import Fore, Style, init
import config

init(autoreset=True)


DEFAULT_MODEL = "llama-3.1-8b-instant"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def resolve_groq_api_key():
    key = config.GROQ_API_KEY
    if key:
        return key

    # Optional compatibility: allow reading from local config.py if present.
    try:
        import config

        return getattr(config, "GROQ_API_KEY", None)
    except Exception:
        return None


def fallback_extractive_summary(text, sentence_count=3):
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    if not sentences:
        return text.strip()
    return ". ".join(sentences[:sentence_count]) + ("." if len(sentences) >= sentence_count else "")


def query_groq(text, min_length, max_length, model_name=DEFAULT_MODEL):
    api_key = resolve_groq_api_key()
    if not api_key:
        print(Fore.YELLOW + Style.BRIGHT + "GROQ_API_KEY is missing. Falling back to local summary.")
        return {"fallback": True, "error": "Missing GROQ_API_KEY"}

    system_prompt = (
        "You are a precise text summarizer. "
        f"Return a clear summary between {min_length} and {max_length} words. "
        "Do not add facts not present in the input."
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        print(Fore.YELLOW + Style.BRIGHT + f"\nGroq API request failed: {exc}")
        print(Fore.YELLOW + "Using local fallback summary instead.")
        return {"fallback": True, "error": str(exc)}


def summarize_text(text, min_length, max_length, model_name=DEFAULT_MODEL):
    print(Fore.BLUE + Style.BRIGHT + f"\nRunning summarization with model: {model_name}")
    result = query_groq(text, min_length, max_length, model_name=model_name)

    if isinstance(result, dict) and result.get("fallback"):
        return fallback_extractive_summary(text)

    try:
        return result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        return fallback_extractive_summary(text)


def main():
    print(Fore.YELLOW + Style.BRIGHT + "Hi there. What is your name?")
    user_name = input("Your name: ").strip() or "User"
    print(Fore.GREEN + f"Welcome, {user_name}. Let's summarize your text.")

    print(Fore.YELLOW + Style.BRIGHT + "\nPlease enter the text you want to summarize:")
    user_text = input("> ").strip()

    if not user_text:
        print(Fore.RED + Style.BRIGHT + "No text provided. Exiting.")
        return

    print(Fore.YELLOW + "\nOptional: Enter a Groq model name, or press Enter for default.")
    print(Fore.CYAN + f"Default model: {DEFAULT_MODEL}")
    model_choice = input("Model: ").strip() or DEFAULT_MODEL

    print(Fore.YELLOW + "\nChoose summarization style:")
    print("1. Standard summary")
    print("2. Enhanced summary")
    style_choice = input("Enter 1 or 2: ").strip()

    if style_choice == "2":
        min_length, max_length = 80, 100
        print(Fore.BLUE + "Using enhanced summarization settings.")
    else:
        min_length, max_length = 50, 80
        print(Fore.BLUE + "Using standard summarization settings.")

    summary = summarize_text(user_text, min_length, max_length, model_name=model_choice)

    if summary:
        print(Fore.GREEN + Style.BRIGHT + f"\nSummary for {user_name}:")
        print(Fore.GREEN + summary)
    else:
        print(Fore.RED + Style.BRIGHT + "Unable to generate summary.")


if __name__ == "__main__":
    main()
