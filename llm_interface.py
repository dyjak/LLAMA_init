import os
from pathlib import Path
from typing import Optional, List, Dict

from llm_core import SimpleLLM
import config


class SimpleLLMInterface:
    def __init__(self):
        """interfejs użytkownika dla simplellm."""
        self.model = None
        self.model_path = None
        self.history = []

    def load_model(
            self,
            model_path: Optional[str] = None,
            repo_id: Optional[str] = None,
            filename: Optional[str] = None,
            **kwargs
    ) -> bool:
        """
        ładuje model z podanej ścieżki lub z predefiniowanego repozytorium.

        args:
            model_path: ścieżka do lokalnego pliku modelu
            repo_id: id repozytorium dla modeli predefiniowanych
            filename: nazwa pliku w repozytorium
            **kwargs: dodatkowe parametry dla modelu (context_size, n_gpu_layers, itd.)
        """
        try:
            if repo_id and filename:
                self.model = SimpleLLM(
                    repo_id=repo_id,
                    filename=filename,
                    verbose=True,
                    **kwargs
                )
                print(f"Model załadowany: {filename} z repozytorium {repo_id}")
            elif model_path:
                self.model = SimpleLLM(
                    model_path=model_path,
                    verbose=True,
                    **kwargs
                )
                self.model_path = model_path
                print(f"Model załadowany: {self.model.model_name}")
            else:
                print("Musisz podać ścieżkę do modelu lub dane repozytorium")
                return False

            model_info = self.model.get_info()
            print(f"Informacje o modelu:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")
            return True
        except Exception as e:
            print(f"Błąd podczas ładowania modelu: {e}")
            return False

    def chat(
            self,
            prompt: str,
            system_prompt: str = config.DEFAULT_SYSTEM_PROMPT,
            max_tokens: int = config.DEFAULT_MAX_TOKENS,
            temperature: float = config.DEFAULT_TEMPERATURE,
            stream: bool = True
    ) -> str:
        """przeprowadza interakcję z modelem w stylu czatu."""
        if self.model is None:
            print("Najpierw załaduj model używając load_model()")
            return ""

        # formatowanie prompta zgodnie z formatem instrukcji dla modeli llama
        formatted_prompt = f"""<s>[INST] {system_prompt}

{prompt} [/INST]"""

        if stream:
            print("\nOdpowiedź:")
            full_response = ""
            for chunk in self.model.generate(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
            ):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n")
            return full_response
        else:
            response = self.model.generate(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            print(f"\nOdpowiedź:\n{response}\n")
            return response

    def complete(
            self,
            prompt: str,
            max_tokens: int = config.DEFAULT_MAX_TOKENS,
            temperature: float = config.DEFAULT_TEMPERATURE,
            echo: bool = False,
            stream: bool = True
    ):
        """przeprowadza proste uzupełnienie tekstu bez formatowania czatu."""
        if self.model is None:
            print("Najpierw załaduj model używając load_model()")
            return ""

        if stream:
            print("\nWyjście:")
            full_response = ""
            for chunk in self.model.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    echo=echo,
                    stream=True
            ):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n")
            return full_response
        else:
            output = self.model.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=echo,
                stream=False
            )

            if isinstance(output, dict):
                print(f"\nWyjście:\n{output['choices'][0]['text']}\n")
                return output
            else:
                print(f"\nWyjście:\n{output}\n")
                return output

    def find_local_models(self, models_dir: str = config.DEFAULT_MODELS_DIR) -> List[Path]:
        """wyszukuje lokalne modele w formacie gguf."""
        if not os.path.exists(models_dir):
            return []

        return list(Path(models_dir).glob("*.gguf"))

    def get_predefined_models(self) -> List[Dict]:
        """zwraca listę predefiniowanych modeli."""
        return config.PREDEFINED_MODELS