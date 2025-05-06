"""
moduł zawierający przykłady użycia api simplellm dla deweloperów.
"""

from llm_core import SimpleLLM


def przykład_prostego_użycia():
    """prosty przykład bezpośredniego użycia simplellm."""
    try:
        # inicjalizacja modelu z repozytorium
        model = SimpleLLM(
            repo_id="TheBloke/CodeLlama-70B-Python-GGUF",
            filename="codellama-70b-python.Q2_K.gguf",
            n_gpu_layers=-1,
            verbose=True
        )

        # generowanie tekstu
        output = model.generate(
            "Once upon a time,",
            max_tokens=512,
            echo=True
        )

        print(output)

    except Exception as e:
        print(f"Wystąpił błąd: {e}")


def przykład_generowania_kodu():
    """przykład generowania kodu python."""
    try:
        # inicjalizacja mniejszego modelu do generowania kodu
        model = SimpleLLM(
            repo_id="TheBloke/Phi-2-GGUF",
            filename="phi-2.Q4_K_M.gguf",
            verbose=True
        )

        # formatowanie prompta dla generowania kodu
        prompt = """Napisz funkcję w Pythonie, która sprawdza czy liczba jest liczbą pierwszą."""

        print("Prompt:", prompt)
        print("\nGenerowanie kodu...")

        # generowanie odpowiedzi
        output = model.generate(
            prompt,
            max_tokens=512,
            temperature=0.2  # niższa temperatura dla bardziej spójnego kodu
        )

        print("\nWygenerowany kod:")
        print(output)

    except Exception as e:
        print(f"Wystąpił błąd: {e}")


def przykład_strumieniowania():
    """przykład strumieniowego generowania odpowiedzi."""
    try:
        # inicjalizacja mniejszego modelu dla szybszego działania
        model = SimpleLLM(
            repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            verbose=True
        )

        # prompt w formacie instrukcji
        prompt = """<s>[INST] Wyjaśnij teorię względności Einsteina w prostych słowach [/INST]"""

        print("Prompt:", prompt)
        print("\nGenerowanie (strumieniowo):")

        # strumieniowe generowanie
        for chunk in model.generate(
                prompt,
                max_tokens=1024,
                temperature=0.7,
                stream=True
        ):
            print(chunk, end="", flush=True)

        print("\n\nGenerowanie zakończone.")

    except Exception as e:
        print(f"Wystąpił błąd: {e}")


if __name__ == "__main__":
    print("Wybierz przykład do uruchomienia:")
    print("1. Proste użycie")
    print("2. Generowanie kodu")
    print("3. Strumieniowanie odpowiedzi")

    choice = input("Twój wybór (1-3): ")

    if choice == "1":
        przykład_prostego_użycia()
    elif choice == "2":
        przykład_generowania_kodu()
    elif choice == "3":
        przykład_strumieniowania()
    else:
        print("Nieprawidłowy wybór")