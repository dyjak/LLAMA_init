import argparse
import os
from pathlib import Path

from llm_interface import SimpleLLMInterface
import config


def run_cli():
    """uruchamia interfejs wiersza poleceń dla simplellm."""
    parser = argparse.ArgumentParser(description="Prosty interfejs LLM")
    parser.add_argument("--model", type=str, help="Ścieżka do modelu GGUF")
    parser.add_argument("--repo_id", type=str, help="ID repozytorium dla modeli predefiniowanych")
    parser.add_argument("--filename", type=str, help="Nazwa pliku w repozytorium")
    parser.add_argument("--ctx_size", type=int, default=config.DEFAULT_CONTEXT_SIZE, help="Rozmiar kontekstu")
    parser.add_argument("--gpu_layers", type=int, default=config.DEFAULT_N_GPU_LAYERS,
                        help="Liczba warstw GPU (-1 = wszystkie)")
    parser.add_argument("--threads", type=int, default=config.DEFAULT_N_CPU_THREADS, help="Liczba wątków CPU")
    parser.add_argument("--mode", type=str, choices=["chat", "complete"], default="chat",
                        help="Tryb pracy: chat (z formatowaniem czatu) lub complete (surowy prompt)")

    args = parser.parse_args()

    interface = SimpleLLMInterface()

    # sprawdź, czy podano argumenty dla modelu predefiniowanego
    if args.repo_id and args.filename:
        print(f"Ładowanie modelu z repozytorium {args.repo_id}...")
        success = interface.load_model(
            repo_id=args.repo_id,
            filename=args.filename,
            context_size=args.ctx_size,
            n_gpu_layers=args.gpu_layers,
            n_threads=args.threads
        )
    elif args.model:
        model_path = args.model
        success = interface.load_model(
            model_path=model_path,
            context_size=args.ctx_size,
            n_gpu_layers=args.gpu_layers,
            n_threads=args.threads
        )
    else:
        # szukamy modeli lokalnych
        local_models = interface.find_local_models()

        if local_models:
            print(f"Szukam modeli w {config.DEFAULT_MODELS_DIR}...")
            print("Dostępne lokalne modele:")
            for i, model in enumerate(local_models):
                print(f"L{i + 1}. {model.name}")

        # pokazujemy predefiniowane modele
        predefined_models = interface.get_predefined_models()
        print("\nDostępne modele predefiniowane:")
        for i, model in enumerate(predefined_models):
            print(f"P{i + 1}. {model['repo_id']} - {model['filename']}")

        print("\nWybierz model (np. L1 dla lokalnego, P1 dla predefiniowanego) lub podaj własną ścieżkę/repo: ")
        choice = input().strip()

        if choice.startswith('L'):
            try:
                idx = int(choice[1:]) - 1
                if 0 <= idx < len(local_models):
                    model_path = str(local_models[idx])
                    success = interface.load_model(
                        model_path=model_path,
                        context_size=args.ctx_size,
                        n_gpu_layers=args.gpu_layers,
                        n_threads=args.threads
                    )
                else:
                    print("Nieprawidłowy wybór")
                    return
            except (ValueError, IndexError):
                print("Nieprawidłowy wybór")
                return

        elif choice.startswith('P'):
            try:
                idx = int(choice[1:]) - 1
                if 0 <= idx < len(predefined_models):
                    model = predefined_models[idx]
                    success = interface.load_model(
                        repo_id=model["repo_id"],
                        filename=model["filename"],
                        context_size=args.ctx_size,
                        n_gpu_layers=args.gpu_layers,
                        n_threads=args.threads
                    )
                else:
                    print("Nieprawidłowy wybór")
                    return
            except (ValueError, IndexError):
                print("Nieprawidłowy wybór")
                return

        elif os.path.exists(choice):
            success = interface.load_model(
                model_path=choice,
                context_size=args.ctx_size,
                n_gpu_layers=args.gpu_layers,
                n_threads=args.threads
            )
        else:
            # zakładamy, że użytkownik podał własne repo i filename w formacie repo:filename
            if ":" in choice:
                repo_id, filename = choice.split(":", 1)
                success = interface.load_model(
                    repo_id=repo_id,
                    filename=filename,
                    context_size=args.ctx_size,
                    n_gpu_layers=args.gpu_layers,
                    n_threads=args.threads
                )
            else:
                print("Nieprawidłowy format. Użyj 'repo_id:filename' dla modeli predefiniowanych")
                return

    if not success:
        return

    print("\nWitaj w prostym interfejsie LLM!")
    print(f"Aktualny tryb: {args.mode}")
    print("Wpisz 'q' aby wyjść, 'mode' aby zmienić tryb")

    mode = args.mode

    if mode == "chat":
        system_prompt = config.DEFAULT_SYSTEM_PROMPT
        print(f"\nAktualny system prompt: {system_prompt}")
        print("Chcesz zmienić system prompt? (t/n): ", end="")
        if input().strip().lower() == 't':
            print("Podaj nowy system prompt: ", end="")
            system_prompt = input().strip()

    while True:
        if mode == "chat":
            print("\nWprowadź prompt (lub 'q' aby wyjść, 'mode' aby zmienić tryb): ", end="")
        else:
            print("\nWprowadź tekst do uzupełnienia (lub 'q' aby wyjść, 'mode' aby zmienić tryb): ", end="")

        prompt = input().strip()

        if prompt.lower() == 'q':
            break
        elif prompt.lower() == 'mode':
            mode = "complete" if mode == "chat" else "chat"
            print(f"Tryb zmieniony na: {mode}")
            if mode == "chat":
                system_prompt = config.DEFAULT_SYSTEM_PROMPT
                print(f"\nAktualny system prompt: {system_prompt}")
                print("Chcesz zmienić system prompt? (t/n): ", end="")
                if input().strip().lower() == 't':
                    print("Podaj nowy system prompt: ", end="")
                    system_prompt = input().strip()
            continue

        print("Generowanie...")
        if mode == "chat":
            interface.chat(prompt, system_prompt=system_prompt)
        else:
            interface.complete(prompt)


if __name__ == "__main__":
    run_cli()