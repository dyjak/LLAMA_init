import sys
import argparse
import os


def main():
    """
    Główna funkcja programu pozwalająca na wybór między interfejsem graficznym a konsolowym.
    """
    parser = argparse.ArgumentParser(description="AdvancedLLM - Zaawansowany interfejs do lokalnych modeli LLM")
    parser.add_argument("--gui", action="store_true", help="Uruchom interfejs graficzny")
    parser.add_argument("--cli", action="store_true", help="Uruchom interfejs wiersza poleceń")

    # Argumenty dla interfejsu wiersza poleceń
    parser.add_argument("--model", type=str, help="Ścieżka do modelu GGUF")
    parser.add_argument("--ctx_size", type=int, help="Rozmiar kontekstu")
    parser.add_argument("--gpu_layers", type=int, help="Liczba warstw GPU (-1 = wszystkie)")
    parser.add_argument("--threads", type=int, help="Liczba wątków CPU")
    parser.add_argument("--mode", type=str, choices=["chat", "complete"], help="Tryb pracy: chat lub complete")
    parser.add_argument("--config", type=str, help="Ścieżka do pliku konfiguracyjnego JSON")

    args = parser.parse_args()

    # Jeśli nie podano jawnie interfejsu, domyślnie uruchom GUI
    if not (args.gui or args.cli):
        args.gui = True

    # Załaduj konfigurację z pliku, jeśli podano
    if args.config:
        if os.path.exists(args.config):
            from config import config
            config.load_config(args.config)
        else:
            print(f"Plik konfiguracyjny {args.config} nie istnieje.")
            sys.exit(1)

    if args.gui:
        try:
            from llm_gui import run_gui
            print("Uruchamianie interfejsu graficznego...")
            run_gui()
        except ImportError as e:
            print(f"Błąd podczas importowania modułu GUI: {e}")
            print("Upewnij się, że masz zainstalowany tkinter lub uruchom program w trybie CLI z opcją --cli")
            sys.exit(1)
        except Exception as e:
            print(f"Wystąpił błąd podczas uruchamiania interfejsu graficznego: {e}")
            sys.exit(1)
    else:  # CLI
        try:
            from cli import run_cli
            print("Uruchamianie interfejsu wiersza poleceń...")

            # Przygotuj argumenty dla CLI
            cli_args = {
                "model_path": args.model,
                "context_size": args.ctx_size,
                "n_gpu_layers": args.gpu_layers,
                "n_threads": args.threads,
                "mode": args.mode
            }

            # Usuń None wartości
            cli_args = {k: v for k, v in cli_args.items() if v is not None}

            run_cli(**cli_args)
        except ImportError as e:
            print(f"Błąd podczas importowania modułu CLI: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Wystąpił błąd podczas uruchamiania interfejsu wiersza poleceń: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()