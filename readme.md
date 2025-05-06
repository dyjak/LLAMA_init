# SimpleLLM - Interfejs dla lokalnych modeli LLM

Prosty interfejs dla lokalnych modeli językowych (LLM) w formacie GGUF, wykorzystujący bibliotekę llama-cpp-python.

## Struktura projektu

- `main.py` - główny plik uruchamiający aplikację (CLI lub GUI)
- `config.py` - konfiguracja i zmienne środowiskowe
- `llm_core.py` - podstawowa klasa SimpleLLM do interakcji z modelami
- `llm_interface.py` - podstawowy interfejs programistyczny
- `cli.py` - interfejs linii poleceń
- `llm_gui.py` - interfejs graficzny
- `examples.py` - przykłady użycia API

## Wymagania

- Python 3.8 lub nowszy
- llama-cpp-python (instalowane automatycznie)
- tkinter (dla interfejsu graficznego)

## Instalacja

Nie wymaga specjalnej instalacji. Pobierz repozytorium i uruchom `main.py`.

```
python main.py --gui  # uruchamia interfejs graficzny
python main.py --cli  # uruchamia interfejs wiersza poleceń
```

## Funkcje

- Łatwe ładowanie modeli lokalnych w formacie GGUF
- Pobieranie i korzystanie z predefiniowanych modeli z Hugging Face
- Dwa tryby działania: "chat" i "complete"
- Konfigurowalny system prompt w trybie chat
- Strumieniowe generowanie odpowiedzi
- Dostępny interfejs graficzny oraz wiersza poleceń

## Przykłady użycia

### Interfejs graficzny

```
python main.py --gui
```

### Interfejs wiersza poleceń

```
python main.py --cli --model path/to/model.gguf
```

### Użycie API w kodzie

```python
from llm_core import SimpleLLM

# Inicjalizacja modelu
model = SimpleLLM(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    verbose=True
)

# Generowanie tekstu
output = model.generate(
    "Once upon a time,",
    max_tokens=512,
    temperature=0.7
)

print(output)
```

## Konfiguracja

Możesz dostosować domyślne parametry w pliku `config.py`:

- Ścieżki do plików modeli
- Parametry generowania tekstu
- Predefiniowane modele