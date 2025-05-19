import os
import sys
import time
from typing import Dict, List, Optional, Union, Generator, Any

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
            model_path: str,
            context_size: int = 4096,
            n_gpu_layers: int = -1,
            n_threads: Optional[int] = None,
            batch_size: int = 512,
            f16_kv: bool = True,
            logits_all: bool = False,
            vocab_only: bool = False,
            use_mmap: bool = True,
            use_mlock: bool = False,
            embedding: bool = False,
            rope_scaling_type: Optional[str] = None,
            rope_freq_base: float = 10000.0,
            rope_freq_scale: float = 1.0,
            verbose: bool = False
    ):
        """
        Inicjalizuje prosty interfejs do modelu LLM.

        Args:
            model_path: ścieżka do lokalnego pliku modelu (GGUF format)
            context_size: rozmiar kontekstu dla modelu
            n_gpu_layers: liczba warstw do wykonania na GPU (-1 dla wszystkich)
            n_threads: liczba wątków CPU do użycia
            batch_size: rozmiar partii przy przetwarzaniu
            f16_kv: czy używać half-precision dla key/value cache
            logits_all: czy obliczać logity dla wszystkich tokenów
            vocab_only: czy ładować tylko słownik modelu
            use_mmap: czy używać memory mapping przy ładowaniu modelu
            use_mlock: czy zablokować model w pamięci RAM
            embedding: czy używać modelu do embeddings
            rope_scaling_type: typ skalowania RoPE ('linear', 'yarn' lub None)
            rope_freq_base: bazowa częstotliwość dla RoPE
            rope_freq_scale: skala częstotliwości dla RoPE
            verbose: czy wyświetlać szczegółowe informacje
        """
        # Jeśli nie podano liczby wątków, użyj wszystkich dostępnych
        if n_threads is None:
            import multiprocessing
            n_threads = multiprocessing.cpu_count()

        self.verbose = verbose
        start_time = time.time()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model nie znaleziony: {model_path}")

        if self.verbose:
            print(f"Ładowanie modelu: {model_path}")
            print(f"Parametry: kontekst={context_size}, GPU warstwy={n_gpu_layers}, "
                  f"wątki={n_threads}, batch={batch_size}")

        # Przygotowanie parametrów RoPE
        rope_scaling = None
        if rope_scaling_type:
            rope_scaling = {
                "type": rope_scaling_type,
                "factor": rope_freq_scale
            }

        # Inicjalizacja modelu z lokalnego pliku
        self.llm = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_batch=batch_size,
            f16_kv=f16_kv,
            logits_all=logits_all,
            vocab_only=vocab_only,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            embedding=embedding,
            rope_scaling=rope_scaling,
            rope_freq_base=rope_freq_base,
        )

        self.model_path = model_path
        self.model_name = os.path.basename(model_path)

        if self.verbose:
            load_time = time.time() - start_time
            print(f"Model załadowany w {load_time:.2f} sekund")

    def generate(
            self,
            prompt: str,
            max_tokens: int = 512,
            temperature: float = 0.7,
            top_p: float = 0.95,
            top_k: int = 40,
            repeat_penalty: float = 1.1,
            presence_penalty: float = 0.0,
            frequency_penalty: float = 0.0,
            stream: bool = False,
            stop: List[str] = None,
            echo: bool = False
    ) -> Union[str, Generator[str, None, None], dict]:
        """
        Generuje odpowiedź na podstawie podanego prompta.

        Args:
            prompt: tekst wejściowy dla modelu
            max_tokens: maksymalna liczba tokenów do wygenerowania
            temperature: temperatura generowania (wyższa = bardziej losowo)
            top_p: parametr próbkowania nucleus
            top_k: liczba top tokenów do rozważenia
            repeat_penalty: kara za powtarzanie tokenów
            presence_penalty: kara za obecność tokenów w tekście wejściowym
            frequency_penalty: kara za częstość występowania tokenów
            stream: czy strumieniować odpowiedź
            stop: lista sekwencji, które zatrzymują generowanie
            echo: czy załączyć prompt w wyjściu

        Returns:
            wygenerowany tekst, generator tekstu lub pełny słownik odpowiedzi
        """
        if self.verbose:
            print(f"Generowanie z parametrami: max_tokens={max_tokens}, temp={temperature}, "
                  f"top_p={top_p}, top_k={top_k}, repeat_penalty={repeat_penalty}")

        if stream:
            return self._stream_generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                stop=stop,
                echo=echo
            )
        else:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                stop=stop,
                echo=echo,
            )
            # Zwróć pełny słownik odpowiedzi lub tylko wygenerowany tekst
            return output if echo else output["choices"][0]["text"]

    def _stream_generate(
            self,
            prompt: str,
            max_tokens: int,
            temperature: float,
            top_p: float,
            top_k: int,
            repeat_penalty: float,
            presence_penalty: float,
            frequency_penalty: float,
            stop: List[str] = None,
            echo: bool = False
    ) -> Generator[str, None, None]:
        """Generuje odpowiedź w trybie strumieniowym."""
        for output in self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                stop=stop,
                stream=True,
                echo=echo,
        ):
            chunk = output["choices"][0]["text"]
            yield chunk

    def get_info(self) -> Dict[str, Any]:
        """Zwraca podstawowe informacje o modelu."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "context_size": self.llm.n_ctx(),
            "embedding_size": self.llm.n_embd(),
            "vocabulary_size": self.llm.n_vocab(),
            "n_gpu_layers": getattr(self.llm, "n_gpu_layers", "nieznane"),
            "n_threads": getattr(self.llm, "n_threads", "nieznane"),
        }

    def get_tokenizer(self):
        """Zwraca tokenizer modelu."""
        return self.llm

    def tokenize(self, text: str) -> List[int]:
        """Tokenizuje tekst, zwracając listę tokenów."""
        return self.llm.tokenize(text.encode())

    def detokenize(self, tokens: List[int]) -> str:
        """Detokenizuje listę tokenów, zwracając tekst."""
        return self.llm.detokenize(tokens).decode("utf-8", errors="replace")

    def get_token_embedding(self, token_id: int) -> List[float]:
        """Zwraca embedding dla danego tokenu."""
        return self.llm.get_embedding(token_id)