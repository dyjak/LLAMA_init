import argparse
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

from llm_interface import SimpleLLMInterface
from config import config


def load_or_select_model(interface: SimpleLLMInterface, model_path: Optional[str] = None) -> bool:
    """
    Ładuje model z podanej ścieżki lub pozwala użytkownikowi wybrać model.

    Args:
        interface: Interfejs SimpleLLM
        model_path: Opcjonalna ścieżka do modelu

    Returns:
        True jeśli model został załadowany, False w przeciwnym razie
    """
    if model_path:
        # Jeśli podano model, spróbuj go załadować
        if not os.path.exists(model_path):
            print(f"Nie znaleziono pliku modelu: {model_path}")
            return False

        model_params = config.get_model_params()
        print(f"Ładowanie modelu: {model_path}...")
        return interface.load_model(model_path=model_path, **model_params)
    else:
        # Pokaż ostatnio używane modele
        recent_models = interface.get_recent_models()
        if recent_models:
            print("\nOstatnio używane modele:")
            for i, model in enumerate(recent_models):
                if os.path.exists(model):
                    print(f"{i + 1}. {model}")

        # Szukaj modeli w katalogu
        models_dir = config.get("last_models_dir")
        if not models_dir or not os.path.exists(models_dir):
            models_dir = os.path.expanduser("~/models")

        if os.path.exists(models_dir):
            local_models = interface.find_local_models(models_dir)

            if local_models:
                print(f"\nModele znalezione w {models_dir}:")
                for i, model in enumerate(local_models):
                    print(f"L{i + 1}. {model.name}")

        # Poproś użytkownika o wybór modelu
        print("\nWybierz model (numer z listy), podaj ścieżkę do modelu lub naciśnij Enter, aby przeszukać katalog: ")
        choice = input().strip()

        if not choice:
            # Jeśli użytkownik nie podał wyboru, pokaż okno dialogowe
            print("Podaj ścieżkę do katalogu z modelami: ")
            models_dir = input().strip()

            if not models_dir:
                models_dir = os.path.expanduser("~/models")

            if not os.path.exists(models_dir):
                print(f"Katalog {models_dir} nie istnieje.")
                return False

            local_models = interface.find_local_models(models_dir)

            if not local_models:
                print(f"Nie znaleziono modeli w katalogu {models_dir}.")
                return False

            print(f"\nModele znalezione w {models_dir}:")
            for i, model in enumerate(local_models):
                print(f"{i + 1}. {model.name}")

            print("\nWybierz model (numer): ")
            choice = input().strip()

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(local_models):
                    model_path = str(local_models[idx])
                else:
                    print("Nieprawidłowy wybór.")
                    return False
            except ValueError:
                print("Nieprawidłowy wybór.")
                return False
        elif choice.isdigit():
            # Jeśli użytkownik podał numer, wybierz z listy ostatnio używanych modeli
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(recent_models):
                    model_path = recent_models[idx]
                else:
                    print("Nieprawidłowy wybór.")
                    return False
            except ValueError:
                print("Nieprawidłowy wybór.")
                return False
        elif choice.startswith('L') and choice[1:].isdigit():
            # Jeśli użytkownik podał Lx, wybierz z listy lokalnych modeli
            try:
                idx = int(choice[1:]) - 1
                if 0 <= idx < len(local_models):
                    model_path = str(local_models[idx])
                else:
                    print("Nieprawidłowy wybór.")
                    return False
            except ValueError:
                print("Nieprawidłowy wybór.")
                return False
        else:
            # Traktuj wybór jako ścieżkę do modelu
            model_path = choice
            if not os.path.exists(model_path):
                print(f"Nie znaleziono pliku modelu: {model_path}")
                return False

        # Załaduj wybrany model
        model_params = config.get_model_params()
        print(f"Ładowanie modelu: {model_path}...")
        return interface.load_model(model_path=model_path, **model_params)


def edit_parameters(param_group: str) -> Dict[str, Any]:
    """
    Pozwala użytkownikowi edytować parametry modelu lub generowania.

    Args:
        param_group: Grupa parametrów do edycji ('model' lub 'generation')

    Returns:
        Słownik zaktualizowanych parametrów
    """
    if param_group == "model":
        params = config.get_model_params()
        print("\n=== Edycja parametrów modelu ===")
    else:
        params = config.get_generation_params()
        print("\n=== Edycja parametrów generowania ===")

    # Wyświetl aktualne parametry
    print("\nAktualne parametry:")
    for name, value in params.items():
        print(f"{name} = {value}")

    # Edycja parametrów
    print("\nPodaj parametry do zmiany w formacie 'parametr=wartość' (Enter aby zakończyć):")

    while True:
        line = input().strip()
        if not line:
            break

        try:
            name, value = line.split('=', 1)
            name = name.strip()
            value = value.strip()

            if name not in params:
                print(f"Nieznany parametr: {name}")
                continue

            # Próba konwersji wartości do odpowiedniego typu
            current_value = params[name]
            if isinstance(current_value, bool):
                if value.lower() in ('true', 't', '1', 'yes', 'y'):
                    params[name] = True
                elif value.lower() in ('false', 'f', '0', 'no', 'n'):
                    params[name] = False
                else:
                    print(f"Nieprawidłowa wartość bool dla {name}: {value}")
                    continue
            elif isinstance(current_value, int):
                try:
                    params[name] = int(value)
                except ValueError:
                    print(f"Nieprawidłowa wartość int dla {name}: {value}")
                    continue
            elif isinstance(current_value, float):
                try:
                    params[name] = float(value)
                except ValueError:
                    print(f"Nieprawidłowa wartość float dla {name}: {value}")
                    continue
            elif current_value is None and name == "rope_scaling_type":
                if value.lower() in ('none', 'brak', 'null'):
                    params[name] = None
                else:
                    params[name] = value
            else:
                params[name] = value

            print(f"Zmieniono {name} = {params[name]}")

        except ValueError:
            print("Nieprawidłowy format. Użyj 'parametr=wartość'.")

    return params


def run_cli(model_path: Optional[str] = None, **kwargs):
    """
    Uruchamia interfejs wiersza poleceń dla SimpleLLM.

    Args:
        model_path: Opcjonalna ścieżka do modelu
        **kwargs: Dodatkowe parametry dla modelu
    """
    interface = SimpleLLMInterface()

    # Aktualizuj parametry modelu na podstawie argumentów wiersza poleceń
    if kwargs:
        model_params = config.get_model_params()
        for key, value in kwargs.items():
            if key in model_params:
                # Konwersja na odpowiedni typ
                if isinstance(model_params[key], bool):
                    model_params[key] = bool(value)
                elif isinstance(model_params[key], int):
                    model_params[key] = int(value)
                elif isinstance(model_params[key], float):
                    model_params[key] = float(value)
                else:
                    model_params[key] = value
        config.update_section("model", model_params)

    # Załaduj model lub pozwól użytkownikowi wybrać
    if not load_or_select_model(interface, model_path):
        print("Nie udało się załadować modelu. Wyjście.")
        return

    mode = kwargs.get("mode", "chat")
    if mode not in ["chat", "complete"]:
        mode = "chat"

    # Ustawienia generowania
    generation_params = config.get_generation_params()

    # Pobierz system prompt dla trybu chat
    if mode == "chat":
        system_prompt = config.get("system_prompt")
        print(f"\nAktualny system prompt: {system_prompt}")
        print("Chcesz zmienić system prompt? (t/n): ", end="")
        if input().strip().lower() in ('t', 'tak', 'y', 'yes'):
            print("Podaj nowy system prompt: ", end="")
            system_prompt = input().strip()
            config.set("system_prompt", system_prompt)
            config.save_config()

    print("\n=== SimpleLLM CLI ===")
    print(f"Tryb: {mode}")
    print("Dostępne komendy:")
    print("  q - wyjście")
    print("  mode - zmiana trybu")
    print("  params - edycja parametrów generowania")
    print("  model - edycja parametrów modelu")
    print("  save - zapisz konfigurację")
    print("  load - załaduj nowy model")

    while True:
        if mode == "chat":
            print("\nWprowadź prompt (lub komendę): ", end="")
        else:
            print("\nWprowadź tekst do uzupełnienia (lub komendę): ", end="")

        prompt = input().strip()

        if prompt.lower() == 'q':
            break
        elif prompt.lower() == 'mode':
            mode = "complete" if mode == "chat" else "chat"
            print(f"Tryb zmieniony na: {mode}")
            if mode == "chat":
                system_prompt = config.get("system_prompt")
                print(f"\nAktualny system prompt: {system_prompt}")
                print("Chcesz zmienić system prompt? (t/n): ", end="")
                if input().strip().lower() in ('t', 'tak', 'y', 'yes'):
                    print("Podaj nowy system prompt: ", end="")
                    system_prompt = input().strip()
                    config.set("system_prompt", system_prompt)
                    config.save_config()
        elif prompt.lower() == 'params':
            updated_params = edit_parameters("generation")
            interface.update_generation_params(updated_params)
            generation_params = updated_params
            print("Parametry generowania zaktualizowane.")
        elif prompt.lower() == 'model':
            updated_params = edit_parameters("model")
            interface.update_model_params(updated_params)
            print("Parametry modelu zaktualizowane. Zmiany będą widoczne po ponownym załadowaniu modelu.")
        elif prompt.lower() == 'save':
            if interface.save_current_config():
                print("Konfiguracja zapisana.")
            else:
                print("Błąd podczas zapisywania konfiguracji.")
        elif prompt.lower() == 'load':
            if load_or_select_model(interface):
                print("Model załadowany pomyślnie.")
            else:
                print("Nie udało się załadować modelu.")
        else:
            # Generowanie odpowiedzi
            print("Generowanie...")
            if mode == "chat":
                response = interface.chat(
                    prompt,
                    system_prompt=system_prompt,
                    **generation_params
                )
                if not generation_params.get("stream", True):
                    print(f"\nOdpowiedź:\n{response}\n")
            else:
                response = interface.complete(
                    prompt,
                    **generation_params
                )
                if not generation_params.get("stream", True):
                    if isinstance(response, dict):
                        response = response["choices"][0]["text"]
                    print(f"\nWygenerowany tekst:\n{response}\n")


if __name__ == "__main__":
    run_cli()