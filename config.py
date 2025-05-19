import os
import json
import multiprocessing
from typing import Dict, Any

# Domyślna lokalizacja pliku konfiguracyjnego
CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".simplellm_config.json")

# Domyślne parametry modelu
DEFAULT_CONTEXT_SIZE = 4096
DEFAULT_N_GPU_LAYERS = -1  # -1 oznacza użycie wszystkich dostępnych warstw na GPU
DEFAULT_N_CPU_THREADS = multiprocessing.cpu_count()
DEFAULT_BATCH_SIZE = 512
DEFAULT_F16_KV = True  # Użycie half-precision dla key/value cache
DEFAULT_LOGITS_ALL = False
DEFAULT_VOCAB_ONLY = False
DEFAULT_USE_MMAP = True
DEFAULT_USE_MLOCK = False
DEFAULT_EMBEDDING = False
DEFAULT_ROPE_SCALING_TYPE = None  # None, "linear", "yarn"
DEFAULT_ROPE_FREQ_BASE = 10000.0
DEFAULT_ROPE_FREQ_SCALE = 1.0

# Domyślne parametry generowania tekstu
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 40
DEFAULT_REPEAT_PENALTY = 1.1
DEFAULT_PRESENCE_PENALTY = 0.0
DEFAULT_FREQUENCY_PENALTY = 0.0

# Domyślny katalog z modelami lokalnymi
DEFAULT_MODELS_DIR = os.path.expanduser("~/models")

# Domyślny system prompt dla trybu czatu
DEFAULT_SYSTEM_PROMPT = "Jesteś pomocnym asystentem AI."


class Config:
    """Klasa zarządzająca konfiguracją aplikacji."""

    def __init__(self):
        """Inicjalizacja konfiguracji z domyślnymi wartościami."""
        self.config = {
            # Parametry modelu
            "model": {
                "context_size": DEFAULT_CONTEXT_SIZE,
                "n_gpu_layers": DEFAULT_N_GPU_LAYERS,
                "n_cpu_threads": DEFAULT_N_CPU_THREADS,
                "batch_size": DEFAULT_BATCH_SIZE,
                "f16_kv": DEFAULT_F16_KV,
                "logits_all": DEFAULT_LOGITS_ALL,
                "vocab_only": DEFAULT_VOCAB_ONLY,
                "use_mmap": DEFAULT_USE_MMAP,
                "use_mlock": DEFAULT_USE_MLOCK,
                "embedding": DEFAULT_EMBEDDING,
                "rope_scaling_type": DEFAULT_ROPE_SCALING_TYPE,
                "rope_freq_base": DEFAULT_ROPE_FREQ_BASE,
                "rope_freq_scale": DEFAULT_ROPE_FREQ_SCALE
            },
            # Parametry generowania
            "generation": {
                "max_tokens": DEFAULT_MAX_TOKENS,
                "temperature": DEFAULT_TEMPERATURE,
                "top_p": DEFAULT_TOP_P,
                "top_k": DEFAULT_TOP_K,
                "repeat_penalty": DEFAULT_REPEAT_PENALTY,
                "presence_penalty": DEFAULT_PRESENCE_PENALTY,
                "frequency_penalty": DEFAULT_FREQUENCY_PENALTY,
                "stream": True  # Dodana domyślna wartość dla parametru stream
            },
            # Ostatnio używane modele
            "recent_models": [],
            # Ostatnio używany katalog modeli
            "last_models_dir": DEFAULT_MODELS_DIR,
            # Domyślny system prompt
            "system_prompt": DEFAULT_SYSTEM_PROMPT
        }

        # Wczytaj istniejącą konfigurację, jeśli istnieje
        self.load_config()

        # Upewnij się, że kluczowe listy istnieją
        if "recent_models" not in self.config or self.config["recent_models"] is None:
            self.config["recent_models"] = []

    def load_config(self, config_file: str = CONFIG_FILE) -> bool:
        """Wczytuje konfigurację z pliku JSON."""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Aktualizacja konfiguracji, zachowując domyślne wartości dla brakujących kluczy
                    self._update_nested_dict(self.config, loaded_config)
                return True
            return False
        except Exception as e:
            print(f"Błąd podczas wczytywania konfiguracji: {e}")
            return False

    def save_config(self, config_file: str = CONFIG_FILE) -> bool:
        """Zapisuje konfigurację do pliku JSON."""
        try:
            # Upewnij się, że katalog docelowy istnieje
            os.makedirs(os.path.dirname(config_file), exist_ok=True)

            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Błąd podczas zapisywania konfiguracji: {e}")
            return False

    def get(self, section: str, key: str = None) -> Any:
        """Pobiera wartość z konfiguracji."""
        if key is None:
            # Sprawdź czy section jest bezpośrednio w konfiguracji
            if section in self.config:
                return self.config.get(section, {})
            # Jeśli nie, spróbuj pobrać jako zwykłą wartość
            return section

        # Normalne pobieranie section/key
        section_dict = self.config.get(section, {})
        if isinstance(section_dict, dict):
            return section_dict.get(key)
        return None

    def set(self, section: str, key: str, value: Any) -> None:
        """Ustawia wartość w konfiguracji."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value

    def update_section(self, section: str, values: Dict[str, Any]) -> None:
        """Aktualizuje całą sekcję konfiguracji."""
        if section not in self.config:
            self.config[section] = {}
        self._update_nested_dict(self.config[section], values)

    def add_recent_model(self, model_path: str) -> None:
        """Dodaje model do listy ostatnio używanych."""
        if "recent_models" not in self.config:
            self.config["recent_models"] = []

        if model_path in self.config["recent_models"]:
            self.config["recent_models"].remove(model_path)
        self.config["recent_models"].insert(0, model_path)
        # Ogranicz listę do 10 ostatnich modeli
        self.config["recent_models"] = self.config["recent_models"][:10]

    def get_model_params(self) -> Dict[str, Any]:
        """Zwraca wszystkie parametry modelu jako słownik."""
        return self.config.get("model", {}).copy()

    def get_generation_params(self) -> Dict[str, Any]:
        """Zwraca wszystkie parametry generowania jako słownik."""
        return self.config.get("generation", {}).copy()

    def _update_nested_dict(self, d: Dict, u: Dict) -> None:
        """Rekurencyjnie aktualizuje słownik, zachowując strukturę zagnieżdżeń."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v


# Stwórz globalną instancję konfiguracji
config = Config()


# Funkcje modułowe delegujące do instancji config
def add_recent_model(model_path: str) -> None:
    """Dodaje model do listy ostatnio używanych."""
    config.add_recent_model(model_path)


def set(section: str, key: str, value: Any) -> None:
    """Ustawia wartość w konfiguracji."""
    config.set(section, key, value)


def get(section: str, key: str = None) -> Any:
    """Pobiera wartość z konfiguracji."""
    return config.get(section, key)


def update_section(section: str, values: Dict[str, Any]) -> None:
    """Aktualizuje całą sekcję konfiguracji."""
    config.update_section(section, values)


def save_config() -> bool:
    """Zapisuje konfigurację do pliku."""
    return config.save_config()


def get_model_params() -> Dict[str, Any]:
    """Zwraca parametry modelu."""
    return config.get_model_params()


def get_generation_params() -> Dict[str, Any]:
    """Zwraca parametry generowania."""
    return config.get_generation_params()


def load_config(config_file: str = CONFIG_FILE) -> bool:
    """Wczytuje konfigurację z pliku."""
    return config.load_config(config_file)