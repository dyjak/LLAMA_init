import sys
import argparse
from llm_gui import run_gui


def main():
    """
    główna funkcja programu pozwalająca na wybór między interfejsem graficznym a konsolowym.
    """
    parser = argparse.ArgumentParser(description="SimpleLLM - Prosty interfejs do lokalnych modeli LLM")
    parser.add_argument("--gui", action="store_true", help="Uruchom interfejs graficzny")
    parser.add_argument("--cli", action="store_true", help="Uruchom interfejs wiersza poleceń")

    # argumenty dla interfejsu wiersza poleceń
    parser.add_argument("--model", type=str, help="Ścieżka do modelu GGUF")
    parser.add_argument("--repo_id", type=str, help="ID repozytorium dla modeli predefiniowanych")
    parser.add_argument("--filename", type=str, help="Nazwa pliku w repozytorium")
    parser.add_argument("--ctx_size", type=int, help="Rozmiar kontekstu")
    parser.add_argument("--gpu_layers", type=int, help="Liczba warstw GPU (-1 = wszystkie)")
    parser.add_argument("--threads", type=int, help="Liczba wątków CPU")
    parser.add_argument("--mode", type=str, choices=["chat", "complete"], help="Tryb pracy: chat lub complete")

    args = parser.parse_args()

    # jeśli nie podano jawnie interfejsu, domyślnie uruchom gui
    if not (args.gui or args.cli):
        args.gui = True

    if args.gui:
        try:
            from llm_gui import run_gui
            print("Uruchamianie interfejsu graficznego...")
            run_gui()
        except ImportError as e:
            print(f"Błąd podczas importowania modułu GUI: {e}")
            print("Upewnij się, że masz zainstalowany tkinter lub uruchom program w trybie CLI z opcją --cli")
            sys.exit(1)
    else:  # cli
        from cli import run_cli
        print("Uruchamianie interfejsu wiersza poleceń...")
        run_cli()


if __name__ == "__main__":
    main()