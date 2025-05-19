import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator, Union

from llm_core import SimpleLLM
from config import config  # Importujemy instancję Config, nie moduł


class SimpleLLMInterface:
    def __init__(self):
        """Interfejs użytkownika dla SimpleLLM."""
        self.model = None
        self.history = []
        self.current_model_params = {}

    def load_model(
            self,
            model_path: str,
            **kwargs
    ) -> bool:
        """
        Ładuje model z podanej ścieżki.

        Args:
            model_path: ścieżka do lokalnego pliku modelu
            **kwargs: dodatkowe parametry dla modelu

        Returns:
            True jeśli model został pomyślnie załadowany, False w przeciwnym razie
        """
        try:
            if not os.path.exists(model_path):
                print(f"Nie znaleziono pliku modelu: {model_path}")
                return False

            # Zapisz ścieżkę do ostatnio używanych modeli
            recent_models = config.config.get("recent_models", [])
            if model_path in recent_models:
                recent_models.remove(model_path)
            recent_models.insert(0, model_path)
            # Ogranicz listę do 10 ostatnich modeli
            config.config["recent_models"] = recent_models[:10]

            # Aktualizuj ostatni używany katalog
            config.config["last_models_dir"] = os.path.dirname(model_path)

            # Załaduj model z parametrami
            # Pobierz domyślne parametry z konfiguracji i nadpisz je przekazanymi argumentami
            model_params = config.config.get("model", {}).copy()
            model_params.update(kwargs)

            # Zapisz aktualne parametry modelu
            self.current_model_params = model_params

            # Utwórz instancję modelu
            self.model = SimpleLLM(
                model_path=model_path,
                context_size=model_params.get("context_size", 4096),
                n_gpu_layers=model_params.get("n_gpu_layers", -1),
                n_threads=model_params.get("n_cpu_threads"),
                batch_size=model_params.get("batch_size", 512),
                f16_kv=model_params.get("f16_kv", True),
                logits_all=model_params.get("logits_all", False),
                vocab_only=model_params.get("vocab_only", False),
                use_mmap=model_params.get("use_mmap", True),
                use_mlock=model_params.get("use_mlock", False),
                embedding=model_params.get("embedding", False),
                rope_scaling_type=model_params.get("rope_scaling_type"),
                rope_freq_base=model_params.get("rope_freq_base", 10000.0),
                rope_freq_scale=model_params.get("rope_freq_scale", 1.0),
                verbose=True
            )

            # Zapisz konfigurację
            config.save_config()

            model_info = self.model.get_info()
            print(f"Informacje o modelu:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")
            return True

        except Exception as e:
            print(f"Błąd podczas ładowania modelu: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    def chat(
            self,
            prompt: str,
            system_prompt: str = None,
            **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """
        Przeprowadza interakcję z modelem w stylu czatu.

        Args:
            prompt: Tekst wprowadzony przez użytkownika
            system_prompt: Prompt systemowy definiujący zachowanie modelu
            **kwargs: Dodatkowe parametry generowania

        Returns:
            Wygenerowana odpowiedź lub generator odpowiedzi
        """
        if self.model is None:
            print("Najpierw załaduj model używając load_model()")
            return ""

        # Jeśli nie podano system_prompt, użyj domyślnego z konfiguracji
        if system_prompt is None:
            system_prompt = config.config.get("system_prompt", "Jesteś pomocnym asystentem AI.")

        # Formatowanie prompta zgodnie z formatem instrukcji dla modeli
        formatted_prompt = f"""<s>[INST] {system_prompt}

{prompt} [/INST]"""

        # Pobierz parametry generowania z konfiguracji i nadpisz je przekazanymi argumentami
        generation_params = config.config.get("generation", {}).copy()
        generation_params.update(kwargs)

        # Usuń parametry, które nie są używane przez model.generate()
        stream = generation_params.pop("stream", False)

        return self.model.generate(
            formatted_prompt,
            stream=stream,
            **generation_params
        )

    def complete(
            self,
            prompt: str,
            **kwargs
    ):
        """
        Przeprowadza proste uzupełnienie tekstu bez formatowania czatu.

        Args:
            prompt: Tekst wprowadzony przez użytkownika
            **kwargs: Dodatkowe parametry generowania

        Returns:
            Wygenerowana odpowiedź lub generator odpowiedzi
        """
        if self.model is None:
            print("Najpierw załaduj model używając load_model()")
            return ""

        # Pobierz parametry generowania z konfiguracji i nadpisz je przekazanymi argumentami
        generation_params = config.config.get("generation", {}).copy()
        generation_params.update(kwargs)

        return self.model.generate(
            prompt,
            **generation_params
        )

    def find_local_models(self, models_dir: str = None) -> List[Path]:
        """
        Wyszukuje lokalne modele w formacie GGUF.

        Args:
            models_dir: Katalog z modelami

        Returns:
            Lista ścieżek do znalezionych modeli
        """
        if models_dir is None:
            models_dir = config.config.get("last_models_dir", os.path.expanduser("~/models"))

        if not os.path.exists(models_dir):
            return []

        # Wyszukaj pliki GGUF rekurencyjnie we wszystkich podkatalogach
        gguf_files = []
        for root, _, files in os.walk(models_dir):
            for file in files:
                if file.endswith(".gguf"):
                    gguf_files.append(Path(os.path.join(root, file)))

        return gguf_files

    def get_recent_models(self) -> List[str]:
        """
        Zwraca listę ostatnio używanych modeli.

        Returns:
            Lista ścieżek do ostatnio używanych modeli
        """
        recent_models = config.config.get("recent_models", [])
        if recent_models is None:
            return []
        return recent_models

    def save_current_config(self) -> bool:
        """
        Zapisuje aktualną konfigurację.

        Returns:
            True jeśli zapisano pomyślnie, False w przeciwnym razie
        """
        return config.save_config()

    def update_model_params(self, params: Dict[str, Any]) -> None:
        """
        Aktualizuje parametry modelu w konfiguracji.

        Args:
            params: Słownik z nowymi parametrami
        """
        if "model" not in config.config:
            config.config["model"] = {}

        for key, value in params.items():
            config.config["model"][key] = value

    def update_generation_params(self, params: Dict[str, Any]) -> None:
        """
        Aktualizuje parametry generowania w konfiguracji.

        Args:
            params: Słownik z nowymi parametrami
        """
        if "generation" not in config.config:
            config.config["generation"] = {}

        for key, value in params.items():
            config.config["generation"][key] = value

    def set_system_prompt(self, system_prompt: str) -> None:
        """
        Ustawia domyślny system prompt.

        Args:
            system_prompt: Nowy system prompt
        """
        config.config["system_prompt"] = system_prompt