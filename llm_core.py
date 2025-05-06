import os
import sys
import time
from typing import Dict, List, Optional, Union, Generator

# sprawdzamy czy mamy zainstalowaną bibliotekę llama-cpp-python
try:
    from llama_cpp import Llama
except ImportError:
    print("Instalowanie wymaganych bibliotek...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python"])
    from llama_cpp import Llama


class SimpleLLM:
    def __init__(
            self,
            model_path: Optional[str] = None,
            repo_id: Optional[str] = None,
            filename: Optional[str] = None,
            context_size: int = 4096,
            n_gpu_layers: int = -1,
            n_threads: Optional[int] = None,
            verbose: bool = False
    ):
        """
        inicjalizuje prosty interfejs do modelu llm.

        args:
            model_path: ścieżka do lokalnego pliku modelu (gguf format)
            repo_id: id repozytorium dla modeli predefiniowanych
            filename: nazwa pliku w repozytorium
            context_size: rozmiar kontekstu dla modelu
            n_gpu_layers: liczba warstw do wykonania na gpu (-1 dla wszystkich)
            n_threads: liczba wątków cpu do użycia
            verbose: czy wyświetlać szczegółowe informacje
        """
        # jeśli nie podano liczby wątków, użyj wszystkich dostępnych
        if n_threads is None:
            import multiprocessing
            n_threads = multiprocessing.cpu_count()

        self.verbose = verbose
        start_time = time.time()

        # ładowanie modelu z repozytorium lub lokalnego pliku
        if repo_id and filename:
            if self.verbose:
                print(f"Pobieranie modelu z repozytorium: {repo_id}")
                print(f"Plik: {filename}")
                print(f"Kontekst: {context_size}, GPU warstwy: {n_gpu_layers}, Wątki: {n_threads}")

            # inicjalizacja modelu z predefiniowanego repozytorium
            self.llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_ctx=context_size,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
            )
            self.model_name = filename

        elif model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model nie znaleziony: {model_path}")

            if self.verbose:
                print(f"Ładowanie modelu: {model_path}")
                print(f"Kontekst: {context_size}, GPU warstwy: {n_gpu_layers}, Wątki: {n_threads}")

            # inicjalizacja modelu z lokalnego pliku
            self.llm = Llama(
                model_path=model_path,
                n_ctx=context_size,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
            )
            self.model_name = os.path.basename(model_path)

        else:
            raise ValueError("Musisz podać albo ścieżkę do modelu (model_path) albo repo_id i filename")

        if self.verbose:
            load_time = time.time() - start_time
            print(f"Model załadowany w {load_time:.2f} sekund")

    def generate(
            self,
            prompt: str,
            max_tokens: int = 512,
            temperature: float = 0.7,
            top_p: float = 0.95,
            stream: bool = False,
            stop: List[str] = None,
            echo: bool = False
    ) -> Union[str, Generator[str, None, None], dict]:
        """
        generuje odpowiedź na podstawie podanego prompta.

        args:
            prompt: tekst wejściowy dla modelu
            max_tokens: maksymalna liczba tokenów do wygenerowania
            temperature: temperatura generowania (wyższa = bardziej losowo)
            top_p: parametr próbkowania nucleus
            stream: czy strumieniować odpowiedź
            stop: lista sekwencji, które zatrzymują generowanie
            echo: czy załączyć prompt w wyjściu

        returns:
            wygenerowany tekst, generator tekstu lub pełny słownik odpowiedzi
        """
        if self.verbose:
            print(f"Generowanie z parametrami: max_tokens={max_tokens}, temp={temperature}, top_p={top_p}")

        if stream:
            return self._stream_generate(prompt, max_tokens, temperature, top_p, stop, echo)
        else:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                echo=echo,
            )
            # zwróć pełny słownik odpowiedzi lub tylko wygenerowany tekst
            return output if echo else output["choices"][0]["text"]

    def _stream_generate(
            self,
            prompt: str,
            max_tokens: int,
            temperature: float,
            top_p: float,
            stop: List[str] = None,
            echo: bool = False
    ) -> Generator[str, None, None]:
        """generuje odpowiedź w trybie strumieniowym."""
        for output in self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                stream=True,
                echo=echo,
        ):
            chunk = output["choices"][0]["text"]
            yield chunk

    def get_info(self) -> Dict:
        """zwraca podstawowe informacje o modelu."""
        return {
            "model_name": self.model_name,
            "context_size": self.llm.n_ctx(),
            "embedding_size": self.llm.n_embd(),
            "vocabulary_size": self.llm.n_vocab(),
        }