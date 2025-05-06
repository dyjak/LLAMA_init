import os
import multiprocessing

# konfiguracja ścieżek dla huggingface
HF_HOME = "D:/LOCAL/HugginFace_GGUFs"
os.environ["HF_HOME"] = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_HOME

# domyślne parametry modelu
DEFAULT_CONTEXT_SIZE = 4096
DEFAULT_N_GPU_LAYERS = -1  # -1 oznacza użycie wszystkich dostępnych warstw na GPU
DEFAULT_N_CPU_THREADS = multiprocessing.cpu_count()

# domyślne parametry generowania tekstu
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95

# domyślny katalog z modelami lokalnymi
DEFAULT_MODELS_DIR = os.path.expanduser("~/models")

# predefiniowane modele dla łatwiejszego wyboru
PREDEFINED_MODELS = [
    {"repo_id": "TheBloke/CodeLlama-70B-Python-GGUF", "filename": "codellama-70b-python.Q2_K.gguf"},
    {"repo_id": "TheBloke/Llama-2-7B-Chat-GGUF", "filename": "llama-2-7b-chat.Q4_K_M.gguf"},
    {"repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf"},
    {"repo_id": "TheBloke/Phi-2-GGUF", "filename": "phi-2.Q4_K_M.gguf"}
]

# domyślny system prompt dla trybu czatu
DEFAULT_SYSTEM_PROMPT = "Jesteś pomocnym asystentem AI."