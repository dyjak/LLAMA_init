import os
import threading
import tkinter as tk
# import tk
from tkinter import ttk, scrolledtext, filedialog, messagebox, Frame
from typing import Dict, Any, Optional, List

import json

from llm_interface import SimpleLLMInterface
from config import config

from ttkthemes import ThemedTk



class SettingsPanel(ttk.Frame):
    """Panel ustawień z zakładkami dla różnych kategorii parametrów."""

    def __init__(self, parent, interface, expanded=False):
        super().__init__(parent)
        self.parent = parent
        self.interface = interface
        self.expanded = expanded
        self.toggle_direction = "left"

        self.model_params = {}
        self.generation_params = {}

        # Utwórz zakładki
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Zakładka parametrów modelu
        self.model_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.model_frame, text="Model")

        # Zakładka parametrów generowania
        self.generation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.generation_frame, text="Generowanie")

        # Zakładka ustawień interfejsu
        self.interface_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.interface_frame, text="Interfejs")

        # Przycisk zwijania/rozwijania panelu
        self.toggle_button = ttk.Button(self, text="<<", command=self.toggle_panel, width=3)
        self.toggle_button.pack(side="top", anchor="ne", padx=5, pady=5)

        # Wypełnij zakładki
        self.setup_model_tab()
        self.setup_generation_tab()
        self.setup_interface_tab()

        # Pobierz parametry z konfiguracji
        self.load_config_values()

    def setup_model_tab(self):
        """Tworzy kontrolki dla parametrów modelu."""
        # Utwórz ramkę przewijania
        canvas = tk.Canvas(self.model_frame, height=400)
        scrollbar = ttk.Scrollbar(self.model_frame, orient="vertical", command=canvas.yview)
        self.scrollable_model_frame = ttk.Frame(canvas)

        self.scrollable_model_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_model_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Parametry modelu
        params = [
            ("context_size", "Rozmiar kontekstu", "int", 512, 32768),
            ("n_gpu_layers", "Liczba warstw GPU (-1 = wszystkie)", "int", -1, 100),
            ("n_cpu_threads", "Liczba wątków CPU", "int", 1, 32),
            ("batch_size", "Rozmiar partii", "int", 1, 2048),
            ("f16_kv", "Używaj half-precision dla KV cache", "bool"),
            ("logits_all", "Obliczaj logity dla wszystkich tokenów", "bool"),
            ("vocab_only", "Ładuj tylko słownik modelu", "bool"),
            ("use_mmap", "Używaj memory mapping", "bool"),
            ("use_mlock", "Zablokuj model w RAM", "bool"),
            ("embedding", "Używaj jako model embeddingu", "bool"),
            ("rope_scaling_type", "Typ skalowania RoPE", "choice",
             ["Brak", "linear", "yarn"]),
            ("rope_freq_base", "Bazowa częstotliwość RoPE", "float", 100.0, 100000.0),
            ("rope_freq_scale", "Skala częstotliwości RoPE", "float", 0.1, 10.0),
        ]

        # Utwórz kontrolki dla każdego parametru
        for i, (param_name, label_text, param_type, *args) in enumerate(params):
            frame = ttk.Frame(self.scrollable_model_frame)
            frame.pack(fill="x", padx=5, pady=2)

            ttk.Label(frame, text=label_text, width=35).pack(side="left")

            if param_type == "int":
                min_val, max_val = args
                var = tk.IntVar()
                self.model_params[param_name] = var

                # Dodaj spinner z wartościami min i max
                spinner = ttk.Spinbox(frame, from_=min_val, to=max_val, textvariable=var, width=10)
                spinner.pack(side="left", padx=5)

            elif param_type == "float":
                min_val, max_val = args
                var = tk.DoubleVar()
                self.model_params[param_name] = var

                # Dodaj spinner z wartościami min i max
                spinner = ttk.Spinbox(frame, from_=min_val, to=max_val, increment=0.1, textvariable=var, width=10)
                spinner.pack(side="left", padx=5)

            elif param_type == "bool":
                var = tk.BooleanVar()
                self.model_params[param_name] = var

                # Dodaj przełącznik
                switch = ttk.Checkbutton(frame, variable=var)
                switch.pack(side="left", padx=5)

            elif param_type == "choice":
                options = args[0]
                var = tk.StringVar()
                self.model_params[param_name] = var

                # Dodaj combobox
                combo = ttk.Combobox(frame, textvariable=var, values=options, width=15)
                combo.pack(side="left", padx=5)

        # Przycisk zapisz
        save_frame = ttk.Frame(self.scrollable_model_frame)
        save_frame.pack(fill="x", padx=5, pady=10)
        ttk.Button(save_frame, text="Zapisz parametry modelu", command=self.save_model_params).pack(pady=5)

    def setup_generation_tab(self):
        """Tworzy kontrolki dla parametrów generowania."""
        # Utwórz ramkę przewijania
        canvas = tk.Canvas(self.generation_frame, height=400)
        scrollbar = ttk.Scrollbar(self.generation_frame, orient="vertical", command=canvas.yview)
        self.scrollable_generation_frame = ttk.Frame(canvas)

        self.scrollable_generation_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_generation_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Parametry generowania
        params = [
            ("max_tokens", "Maksymalna liczba tokenów", "int", 1, 32768),
            ("temperature", "Temperatura", "float", 0.0, 2.0),
            ("top_p", "Top-p (nucleus sampling)", "float", 0.0, 1.0),
            ("top_k", "Top-k", "int", 1, 100),
            ("repeat_penalty", "Kara za powtórzenia", "float", 0.0, 2.0),
            ("presence_penalty", "Kara za obecność", "float", 0.0, 2.0),
            ("frequency_penalty", "Kara za częstość", "float", 0.0, 2.0),
            ("stream", "Tryb strumieniowy", "bool"),
        ]

        # Utwórz kontrolki dla każdego parametru
        for i, (param_name, label_text, param_type, *args) in enumerate(params):
            frame = ttk.Frame(self.scrollable_generation_frame)
            frame.pack(fill="x", padx=5, pady=2)

            ttk.Label(frame, text=label_text, width=25).pack(side="left")

            if param_type == "int":
                min_val, max_val = args
                var = tk.IntVar()
                self.generation_params[param_name] = var

                # Dodaj spinner z wartościami min i max
                spinner = ttk.Spinbox(frame, from_=min_val, to=max_val, textvariable=var, width=10)
                spinner.pack(side="left", padx=5)

            elif param_type == "float":
                min_val, max_val = args
                var = tk.DoubleVar()
                self.generation_params[param_name] = var

                # Dodaj spinner z wartościami min i max
                spinner = ttk.Spinbox(frame, from_=min_val, to=max_val, increment=0.05, textvariable=var, width=10)
                spinner.pack(side="left", padx=5)

            elif param_type == "bool":
                var = tk.BooleanVar()
                self.generation_params[param_name] = var

                # Dodaj przełącznik
                switch = ttk.Checkbutton(frame, variable=var)
                switch.pack(side="left", padx=5)

        # Przycisk zapisz
        save_frame = ttk.Frame(self.scrollable_generation_frame)
        save_frame.pack(fill="x", padx=5, pady=10)
        ttk.Button(save_frame, text="Zapisz parametry generowania", command=self.save_generation_params).pack(pady=5)

    def setup_interface_tab(self):
        """Tworzy kontrolki dla ustawień interfejsu."""
        # System prompt
        prompt_frame = ttk.LabelFrame(self.interface_frame, text="System Prompt")
        prompt_frame.pack(fill="x", padx=5, pady=5, expand=False)

        self.system_prompt_var = tk.StringVar()
        system_text = scrolledtext.ScrolledText(prompt_frame, wrap=tk.WORD, height=5)
        system_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Przypisz wartość do widgetu tekstu
        def update_system_prompt(*args):
            system_text.delete("1.0", tk.END)
            system_text.insert("1.0", self.system_prompt_var.get())

        def save_system_prompt():
            # Pobierz tekst z widgetu i zapisz
            prompt = system_text.get("1.0", tk.END).strip()
            self.system_prompt_var.set(prompt)
            self.interface.set_system_prompt(prompt)
            messagebox.showinfo("Zapisano", "System prompt został zapisany")

        self.system_prompt_var.trace_add("write", update_system_prompt)

        ttk.Button(prompt_frame, text="Zapisz system prompt", command=save_system_prompt).pack(pady=5)

        # Ostatnio używane modele
        recent_frame = ttk.LabelFrame(self.interface_frame, text="Ostatnio używane modele")
        recent_frame.pack(fill="both", padx=5, pady=5, expand=True)

        # Lista ostatnio używanych modeli
        self.recent_models_list = tk.Listbox(recent_frame, height=10)
        self.recent_models_list.pack(fill="both", expand=True, padx=5, pady=5)

        ttk.Button(recent_frame, text="Załaduj wybrany model",
                   command=self.load_selected_model).pack(side="left", padx=5, pady=5)
        ttk.Button(recent_frame, text="Odśwież listę",
                   command=self.refresh_recent_models).pack(side="left", padx=5, pady=5)

    def load_config_values(self):
        """Wczytuje wartości z konfiguracji do kontrolek."""
        try:
            # Parametry modelu
            for param_name, var in self.model_params.items():
                try:
                    value = config.get("model", param_name)
                    if param_name == "rope_scaling_type" and value is None:
                        var.set("Brak")
                    elif value is not None:
                        var.set(value)
                except Exception as e:
                    print(f"Błąd przy wczytywaniu parametru modelu {param_name}: {e}")

            # Parametry generowania
            for param_name, var in self.generation_params.items():
                try:
                    value = config.get("generation", param_name)
                    if value is not None:
                        var.set(value)
                except Exception as e:
                    print(f"Błąd przy wczytywaniu parametru generowania {param_name}: {e}")

            # System prompt
            try:
                system_prompt = config.get("system_prompt", "")
                if system_prompt is None:
                    system_prompt = ""
                self.system_prompt_var.set(system_prompt)
            except Exception as e:
                print(f"Błąd przy wczytywaniu system prompt: {e}")
                self.system_prompt_var.set("")

            # Odśwież listę ostatnio używanych modeli
            self.refresh_recent_models()
        except Exception as e:
            import traceback
            print(f"Błąd podczas ładowania wartości konfiguracji: {e}")
            print(traceback.format_exc())

    def save_model_params(self):
        """Zapisuje parametry modelu do konfiguracji."""
        model_params = {}
        for param_name, var in self.model_params.items():
            value = var.get()
            # Konwersja "Brak" na None dla rope_scaling_type
            if param_name == "rope_scaling_type" and value == "Brak":
                value = None
            model_params[param_name] = value

        self.interface.update_model_params(model_params)
        messagebox.showinfo("Zapisano", "Parametry modelu zostały zapisane")

    def save_generation_params(self):
        """Zapisuje parametry generowania do konfiguracji."""
        generation_params = {}
        for param_name, var in self.generation_params.items():
            generation_params[param_name] = var.get()

        self.interface.update_generation_params(generation_params)
        messagebox.showinfo("Zapisano", "Parametry generowania zostały zapisane")

    def toggle_panel(self):
        """Przełącza panel ustawień między stanem rozwiniętym a zwiniętym."""
        self.expanded = not self.expanded

        if self.expanded:
            self.toggle_button.config(text="<<")
            # Zamiast wywołania parent.panel_visible, zmieniamy bezpośrednio wygląd panelu
            self.pack(fill="both", expand=True)
        else:
            self.toggle_button.config(text=">>")
            # Zmniejsz panel, pokazując tylko przycisk
            self.pack(fill="none", expand=False, side="left")


    def refresh_recent_models(self):
        """Odświeża listę ostatnio używanych modeli."""
        self.recent_models_list.delete(0, tk.END)
        recent_models = self.interface.get_recent_models()

        if recent_models is None:
            return

        for model_path in recent_models:
            if os.path.exists(model_path):
                self.recent_models_list.insert(tk.END, model_path)

    def load_selected_model(self):
        """Ładuje wybrany model z listy ostatnio używanych."""
        selection = self.recent_models_list.curselection()
        if selection:
            model_path = self.recent_models_list.get(selection[0])

            # Pobierz aktualne parametry modelu z konfiguracji
            model_params = {}
            for param_name, var in self.model_params.items():
                value = var.get()
                # Konwersja "Brak" na None dla rope_scaling_type
                if param_name == "rope_scaling_type" and value == "Brak":
                    value = None
                model_params[param_name] = value

            # Wywołaj funkcję ładowania modelu w głównym oknie
            self.parent.load_model(model_path, model_params)


class LLMApp(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.root.title("synergiAI 1.0.1")
        self.root.geometry("1800x800")

        # Próba zastosowania motywu, jeśli jest dostępny
        try:
            from ttkthemes import ThemedStyle
            self.style = ThemedStyle(root)
            self.style.set_theme("arc")  # "arc", "equilux", "breeze" - nowoczesne motywy
        except ImportError:
            print("Biblioteka ttkthemes nie jest zainstalowana. Używam domyślnego stylu.")
            self.style = ttk.Style()

        self.pack(fill="both", expand=True)

        self.interface = SimpleLLMInterface()
        self.model_loaded = False
        self.mode = tk.StringVar(value="chat")
        self.chat_history = []
        self.attached_files = []

        # Podział na główne panele: lewy (ustawienia), środkowy (czat), prawy (szczegóły)
        self.main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill="both", expand=True, padx=5, pady=5)

        # Lewy panel - ustawienia
        self.settings_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.settings_frame, weight=1)

        # Środkowy panel - czat
        self.chat_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.chat_frame, weight=3)

        # Prawy panel - szczegóły modelu i kontekst
        self.details_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.details_frame, weight=1)

        # Inicjalizacja paneli
        self.setup_settings_panel()
        self.setup_chat_panel()
        self.setup_details_panel()

    def setup_settings_panel(self):
        """Konfiguracja panelu ustawień (lewy panel)"""
        settings_label = ttk.Label(self.settings_frame, text="Ustawienia", font=("TkDefaultFont", 12, "bold"))
        settings_label.pack(pady=10)

        # Panel ustawień
        self.settings_panel = SettingsPanel(self.settings_frame, self.interface, expanded=True)
        self.settings_panel.pack(fill="both", expand=True, padx=5, pady=5)

        # Przyciski konfiguracji
        config_buttons_frame = ttk.Frame(self.settings_frame)
        config_buttons_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(
            config_buttons_frame,
            text="Zapisz konfigurację",
            command=self.save_config_to_file
        ).pack(side="left", padx=5, pady=5)

        ttk.Button(
            config_buttons_frame,
            text="Wczytaj konfigurację",
            command=self.load_config_from_file
        ).pack(side="left", padx=5, pady=5)

        ttk.Button(
            config_buttons_frame,
            text="Ustaw jako domyślną",
            command=self.save_as_default_config
        ).pack(side="left", padx=5, pady=5)

    def setup_chat_panel(self):
        """Konfiguracja panelu czatu (środkowy panel)"""
        # Górny panel - informacje o modelu
        model_frame = ttk.LabelFrame(self.chat_frame, text="Model")
        model_frame.pack(fill="x", padx=5, pady=5)

        # Przyciski sterujące
        button_frame = ttk.Frame(model_frame)
        button_frame.pack(side="left", padx=5, pady=5)

        ttk.Button(button_frame, text="Załaduj model", command=self.browse_model).pack(side="left", padx=5)

        # Tryb pracy
        mode_frame = ttk.Frame(model_frame)
        mode_frame.pack(side="left", padx=20, pady=5)

        ttk.Label(mode_frame, text="Tryb:").pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="Chat", variable=self.mode, value="chat").pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="Complete", variable=self.mode, value="complete").pack(side="left", padx=5)

        self.model_info_label = ttk.Label(model_frame, text="Brak załadowanego modelu")
        self.model_info_label.pack(side="left", padx=10, pady=5)

        # Środkowy panel - historia czatu
        chat_history_frame = ttk.Frame(self.chat_frame)
        chat_history_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Etykieta "Historia czatu"
        history_label = ttk.Label(chat_history_frame, text="Historia czatu", font=("TkDefaultFont", 10, "bold"))
        history_label.pack(anchor="w", pady=(0, 5))

        # Historia czatu
        self.history_text = scrolledtext.ScrolledText(chat_history_frame, wrap=tk.WORD, height=15,
                                                      font=("TkDefaultFont", 10))
        self.history_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.history_text.config(state="disabled")  # Tylko do odczytu

        # Konfiguracja tagów dla różnych stylów tekstu
        self.history_text.tag_configure("user", foreground="#0066cc", font=("TkDefaultFont", 10, "bold"))
        self.history_text.tag_configure("assistant", foreground="#009933", font=("TkDefaultFont", 10))
        self.history_text.tag_configure("system", foreground="#cc0000", font=("TkDefaultFont", 10, "italic"))
        self.history_text.tag_configure("file", foreground="#993399", font=("TkDefaultFont", 10, "italic"))

        # Dolny panel - wprowadzanie tekstu
        input_frame = ttk.LabelFrame(self.chat_frame, text="Wprowadź prompt")
        input_frame.pack(fill="x", padx=5, pady=5)

        # Pole wprowadzania tekstu
        self.input_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=5, font=("TkDefaultFont", 10))
        self.input_text.pack(fill="x", expand=True, padx=5, pady=5)

        # Pasek przycisków
        buttons_frame = ttk.Frame(input_frame)
        buttons_frame.pack(fill="x", padx=5, pady=5)

        # Przyciski
        ttk.Button(buttons_frame, text="Generuj", command=self.generate_text).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="Dołącz plik", command=self.attach_file).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="Wyczyść", command=self.clear_output).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="Zapisz historię", command=self.save_chat_history).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="Wczytaj historię", command=self.load_chat_history).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="Wyjdź", command=self.root.quit).pack(side="right", padx=5)

    def setup_details_panel(self):
        """Konfiguracja panelu szczegółów (prawy panel)"""
        details_label = ttk.Label(self.details_frame, text="Szczegóły", font=("TkDefaultFont", 12, "bold"))
        details_label.pack(pady=10)

        # Panel dołączonych plików
        files_frame = ttk.LabelFrame(self.details_frame, text="Dołączone pliki")
        files_frame.pack(fill="x", padx=5, pady=5)

        # Lista dołączonych plików
        self.files_listbox = tk.Listbox(files_frame, height=8)
        self.files_listbox.pack(fill="x", expand=True, padx=5, pady=5)

        # Przyciski zarządzania plikami
        files_buttons_frame = ttk.Frame(files_frame)
        files_buttons_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(files_buttons_frame, text="Usuń plik", command=self.remove_file).pack(side="left", padx=5)
        ttk.Button(files_buttons_frame, text="Wyczyść wszystkie", command=self.clear_files).pack(side="left", padx=5)

        # Panel kontekstu
        context_frame = ttk.LabelFrame(self.details_frame, text="Kontekst")
        context_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Pole kontekstu
        self.context_text = scrolledtext.ScrolledText(context_frame, wrap=tk.WORD, height=10,
                                                      font=("TkDefaultFont", 10))
        self.context_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Przyciski kontekstu
        context_buttons_frame = ttk.Frame(context_frame)
        context_buttons_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(context_buttons_frame, text="Zastosuj kontekst", command=self.apply_context).pack(side="left",
                                                                                                     padx=5)
        ttk.Button(context_buttons_frame, text="Wyczyść kontekst", command=self.clear_context).pack(side="left", padx=5)

    def panel_visible(self, visible):
        """Przełącza widoczność panelu ustawień."""
        if visible:
            self.settings_frame.pack(side="left", fill="y", padx=5, pady=5)
        else:
            self.settings_frame.pack_forget()

    def save_config_to_file(self):
        """Zapisuje kompletną konfigurację aplikacji do pliku."""
        file_path = filedialog.asksaveasfilename(
            title="Zapisz konfigurację",
            defaultextension=".json",
            filetypes=[("Pliki JSON", "*.json"), ("Wszystkie pliki", "*.*")]
        )

        if not file_path:
            return

        try:
            # Zbierz wszystkie parametry modelu
            model_config = {}
            for param_name, var in self.settings_panel.model_params.items():
                value = var.get()
                # Konwersja "Brak" na None dla rope_scaling_type
                if param_name == "rope_scaling_type" and value == "Brak":
                    value = None
                model_config[param_name] = value

            # Zbierz wszystkie parametry generowania
            generation_config = {}
            for param_name, var in self.settings_panel.generation_params.items():
                generation_config[param_name] = var.get()

            # Pobierz system prompt i inne ustawienia interfejsu
            system_prompt = self.settings_panel.system_prompt_var.get() if hasattr(self.settings_panel,
                                                                                   'system_prompt_var') else ""

            # Zbierz ostatnio używane modele z konfiguracji
            recent_models = self.interface.get_recent_models()

            # Pobierz ostatnio używany katalog modeli
            last_models_dir = config.config.get("last_models_dir", "")

            # Utwórz pełną konfigurację
            full_config = {
                "model": model_config,
                "generation": generation_config,
                "system_prompt": system_prompt,
                "recent_models": recent_models,
                "last_models_dir": last_models_dir,
                # Dodaj tutaj inne ustawienia, które chcesz zapisać
                "ui_settings": {
                    "window_size": f"{self.root.winfo_width()}x{self.root.winfo_height()}",
                    "panel_expanded": self.settings_panel.expanded,
                    "mode": self.mode.get()
                }
            }

            # Zapisz do pliku
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(full_config, f, indent=4, ensure_ascii=False)

            # Również zaktualizuj standardową konfigurację
            config.config["model"] = model_config
            config.config["generation"] = generation_config
            config.config["system_prompt"] = system_prompt
            config.save_config()

            messagebox.showinfo("Sukces", "Konfiguracja została zapisana do pliku.")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się zapisać konfiguracji: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def load_config_from_file(self):
        """Wczytuje pełną konfigurację aplikacji z pliku."""
        file_path = filedialog.askopenfilename(
            title="Wczytaj konfigurację",
            filetypes=[("Pliki JSON", "*.json"), ("Wszystkie pliki", "*.*")]
        )

        if not file_path:
            return

        try:
            # Wczytaj konfigurację z pliku
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)

            # Sprawdź, czy plik ma podstawowe sekcje
            if not all(key in loaded_config for key in ["model", "generation"]):
                messagebox.showwarning("Uwaga", "Plik konfiguracyjny ma niepełną strukturę.")
                return

            # Zastosuj ustawienia modelu
            for param_name, value in loaded_config["model"].items():
                if param_name in self.settings_panel.model_params:
                    var = self.settings_panel.model_params[param_name]
                    # Obsługa specjalnego przypadku dla rope_scaling_type
                    if param_name == "rope_scaling_type" and value is None:
                        var.set("Brak")
                    else:
                        try:
                            var.set(value)
                        except:
                            print(f"Nie można ustawić {param_name} na {value}")

            # Zastosuj ustawienia generowania
            for param_name, value in loaded_config["generation"].items():
                if param_name in self.settings_panel.generation_params:
                    try:
                        self.settings_panel.generation_params[param_name].set(value)
                    except:
                        print(f"Nie można ustawić {param_name} na {value}")

            # Zastosuj system prompt
            if "system_prompt" in loaded_config and hasattr(self.settings_panel, 'system_prompt_var'):
                self.settings_panel.system_prompt_var.set(loaded_config["system_prompt"])

            # Wczytaj ostatnio używane modele
            if "recent_models" in loaded_config:
                # Aktualizuj listę w konfiguracji
                config.config["recent_models"] = loaded_config["recent_models"]
                # Odśwież listę wyświetlanych modeli
                self.settings_panel.refresh_recent_models()

            # Zastosuj katalog modeli
            if "last_models_dir" in loaded_config:
                config.config["last_models_dir"] = loaded_config["last_models_dir"]

            # Zastosuj ustawienia UI
            if "ui_settings" in loaded_config:
                ui_settings = loaded_config["ui_settings"]

                # Ustaw rozmiar okna
                if "window_size" in ui_settings:
                    try:
                        self.root.geometry(ui_settings["window_size"])
                    except:
                        pass

                # Ustaw tryb
                if "mode" in ui_settings:
                    self.mode.set(ui_settings["mode"])

                # Ustaw stan rozwinięcia panelu
                if "panel_expanded" in ui_settings:
                    if ui_settings["panel_expanded"] != self.settings_panel.expanded:
                        self.settings_panel.toggle_panel()

            # Zaktualizuj główną konfigurację
            config.config["model"] = loaded_config["model"]
            config.config["generation"] = loaded_config["generation"]
            if "system_prompt" in loaded_config:
                config.config["system_prompt"] = loaded_config["system_prompt"]
            config.save_config()

            messagebox.showinfo("Sukces", "Pełna konfiguracja została wczytana z pliku.")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się wczytać konfiguracji: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def save_as_default_config(self):
        """Zapisuje aktualną konfigurację jako domyślną."""
        try:
            # Pobierz wszystkie parametry modelu
            model_config = {}
            for param_name, var in self.settings_panel.model_params.items():
                value = var.get()
                if param_name == "rope_scaling_type" and value == "Brak":
                    value = None
                model_config[param_name] = value

            # Pobierz wszystkie parametry generowania
            generation_config = {}
            for param_name, var in self.settings_panel.generation_params.items():
                generation_config[param_name] = var.get()

            # Pobierz system prompt
            system_prompt = self.settings_panel.system_prompt_var.get() if hasattr(self.settings_panel,
                                                                                   'system_prompt_var') else ""

            # Zaktualizuj główną konfigurację
            config.config["model"] = model_config
            config.config["generation"] = generation_config
            config.config["system_prompt"] = system_prompt
            config.config["ui_settings"] = {
                "window_size": f"{self.root.winfo_width()}x{self.root.winfo_height()}",
                "panel_expanded": self.settings_panel.expanded,
                "mode": self.mode.get()
            }

            # Zapisz do standardowego pliku konfiguracyjnego
            if config.save_config():
                messagebox.showinfo("Sukces", "Aktualna konfiguracja została zapisana jako domyślna.")
            else:
                messagebox.showerror("Błąd", "Nie udało się zapisać konfiguracji jako domyślnej.")
        except Exception as e:
            messagebox.showerror("Błąd", f"Wystąpił problem: {str(e)}")

    def browse_model(self):
        """Otwiera okno wyboru pliku modelu."""
        last_dir = config.get("last_models_dir")
        if not last_dir or not os.path.exists(last_dir):
            last_dir = os.path.expanduser("~")

        file_path = filedialog.askopenfilename(
            title="Wybierz plik modelu",
            initialdir=last_dir,
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")]
        )

        if file_path:
            # Pobierz parametry modelu z konfiguracji
            model_params = {}
            for param_name, var in self.settings_panel.model_params.items():
                value = var.get()
                # Konwersja "Brak" na None dla rope_scaling_type
                if param_name == "rope_scaling_type" and value == "Brak":
                    value = None
                model_params[param_name] = value

            self.load_model(file_path, model_params)

    def load_model(self, model_path, model_params=None):
        """Ładuje model z podanej ścieżki."""
        self.model_info_label.config(text="Ładowanie modelu...")
        self.root.update()

        if model_params is None:
            model_params = config.get_model_params()

        def load_in_thread():
            success = self.interface.load_model(
                model_path=model_path,
                **model_params
            )
            self.root.after(0, self.update_model_info, success)

        thread = threading.Thread(target=load_in_thread)
        thread.daemon = True
        thread.start()

    def update_model_info(self, success):
        """Aktualizuje etykietę z informacjami o modelu."""
        if success:
            model_info = self.interface.model.get_info()
            model_name = model_info.get("model_name", "Nieznany model")
            context_size = model_info.get("context_size", "Nieznany")

            self.model_info_label.config(
                text=f"Model: {model_name} (Kontekst: {context_size})"
            )
            self.model_loaded = True

            # Dodaj informacje do historii czatu
            self.add_to_history(f"System: Załadowano model {model_name}", "system")

            # Odśwież listę ostatnio używanych modeli
            self.settings_panel.refresh_recent_models()

            messagebox.showinfo("Sukces", "Model został pomyślnie załadowany")
        else:
            self.model_info_label.config(text="Brak załadowanego modelu")
            self.model_loaded = False
            messagebox.showerror("Błąd", "Nie udało się załadować modelu")

    def generate_text(self):
        """Generuje tekst na podstawie wprowadzonego prompta."""
        if not self.model_loaded:
            messagebox.showwarning("Ostrzeżenie", "Najpierw załaduj model!")
            return

        prompt = self.input_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Ostrzeżenie", "Wprowadź prompt!")
            return

        # Dodaj prompt do historii
        self.add_to_history(f"Ty: {prompt}", "user")
        self.chat_history.append({"role": "user", "content": prompt})

        # Wyczyść pole wprowadzania
        self.input_text.delete("1.0", tk.END)

        # Przygotuj kontekst z załączonych plików, jeśli istnieją
        file_context = ""
        if self.attached_files:
            file_context = "Zawartość załączonych plików:\n\n"
            for file_info in self.attached_files:
                file_context += f"--- {file_info['name']} ---\n{file_info['content']}\n\n"

            # Dodaj informację o dołączonych plikach do historii
            file_names = ", ".join([file_info['name'] for file_info in self.attached_files])
            self.add_to_history(f"Dołączone pliki: {file_names}", "file")

        # Pobierz dodatkowy kontekst z pola kontekstu
        additional_context = self.context_text.get("1.0", tk.END).strip()
        if additional_context:
            if file_context:
                file_context += "\nDodatkowy kontekst:\n" + additional_context
            else:
                file_context = "Dodatkowy kontekst:\n" + additional_context

        # Połącz prompt i kontekst
        full_prompt = prompt
        if file_context:
            full_prompt = file_context + "\n\n" + prompt

        # Ustaw stan "Generowanie..."
        self.model_info_label.config(text=f"{self.model_info_label.cget('text')} (Generowanie...)")
        self.root.update()

        # Pobierz parametry generowania z konfiguracji
        generation_params = {}
        for param_name, var in self.settings_panel.generation_params.items():
            generation_params[param_name] = var.get()

        def generate_in_thread():
            mode = self.mode.get()
            full_response = ""

            try:
                if mode == "chat":
                    if generation_params.get("stream", True):
                        for chunk in self.interface.chat(full_prompt, **generation_params):
                            full_response += chunk
                            self.add_to_history_streaming(chunk, "assistant")
                    else:
                        full_response = self.interface.chat(full_prompt, **generation_params)
                        self.add_to_history(f"Model: {full_response}", "assistant")
                else:  # tryb complete
                    if generation_params.get("stream", True):
                        for chunk in self.interface.complete(full_prompt, **generation_params):
                            full_response += chunk
                            self.add_to_history_streaming(chunk, "assistant")
                    else:
                        response = self.interface.complete(full_prompt, **generation_params)
                        if isinstance(response, dict):
                            full_response = response["choices"][0]["text"]
                        else:
                            full_response = response
                        self.add_to_history(f"Model: {full_response}", "assistant")

                # Dodaj odpowiedź do historii chatu
                self.chat_history.append({"role": "assistant", "content": full_response})

                # Przywróć normalny stan etykiety modelu
                self.root.after(0, lambda: self.model_info_label.config(
                    text=self.model_info_label.cget('text').replace(" (Generowanie...)", "")
                ))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Błąd generowania", str(e)))
                self.root.after(0, lambda: self.model_info_label.config(
                    text=self.model_info_label.cget('text').replace(" (Generowanie...)", "")
                ))

        thread = threading.Thread(target=generate_in_thread)
        thread.daemon = True
        thread.start()

    def add_to_history(self, text, tag=""):
        """Dodaje tekst do historii czatu."""
        self.history_text.config(state="normal")
        self.history_text.insert(tk.END, text + "\n\n", tag)
        self.history_text.see(tk.END)
        self.history_text.config(state="disabled")

    def add_to_history_streaming(self, chunk, tag=""):
        """Dodaje kawałki odpowiedzi do historii czatu w trybie strumieniowym."""
        self.history_text.config(state="normal")

        # Sprawdź, czy to pierwszy kawałek odpowiedzi w nowej sekwencji
        if not hasattr(self, '_streaming_started') or not self._streaming_started:
            # Dodaj prefiks "Model: " tylko raz na początku sekwencji
            self.history_text.insert(tk.END, "Model: ", tag)
            self._streaming_started = True
            self._last_chunk = ""

        # Dodaj kawałek tekstu
        self.history_text.insert(tk.END, chunk, tag)
        self.history_text.see(tk.END)
        self._last_chunk = chunk

        # Sprawdź, czy to może być koniec odpowiedzi
        if chunk.endswith(("\n", ".", "!", "?")):
            # Jeśli mamy znak końca zdania lub nową linię, dodaj niewielkie opóźnienie
            # przed uznaniem sekwencji za zakończoną
            self.root.after(100, self._check_streaming_end)

        self.history_text.config(state="disabled")

    def _check_streaming_end(self):
        """Sprawdza, czy strumieniowanie powinno zostać uznane za zakończone."""
        # Jeśli przez pewien czas nie było nowych chunków, uznaj sekwencję za zakończoną
        self.history_text.config(state="normal")
        self.history_text.insert(tk.END, "\n\n")
        self._streaming_started = False
        self.history_text.config(state="disabled")

    def attach_file(self):
        """Dołącza plik do aktualnej konwersacji."""
        file_path = filedialog.askopenfilename(
            title="Wybierz plik do dołączenia",
            filetypes=[
                ("Pliki tekstowe", "*.txt"),
                ("Pliki HTML", "*.html;*.htm"),
                ("Pliki PDF", "*.pdf"),
                ("Dokumenty", "*.docx;*.doc"),
                ("Wszystkie pliki", "*.*")
            ]
        )

        if not file_path:
            return

        file_name = os.path.basename(file_path)
        file_content = ""

        try:
            # Obsługa różnych typów plików
            if file_path.lower().endswith('.pdf'):
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page_num in range(len(reader.pages)):
                            page = reader.pages[page_num]
                            text += page.extract_text() + "\n"
                        file_content = text
                except ImportError:
                    messagebox.showwarning("Brak biblioteki",
                                           "Do obsługi PDF wymagana jest biblioteka PyPDF2. Zainstaluj ją używając: pip install PyPDF2")
                    return

            elif file_path.lower().endswith(('.docx', '.doc')):
                try:
                    from docx import Document
                    document = Document(file_path)
                    paragraphs = [p.text for p in document.paragraphs]
                    file_content = '\n'.join(paragraphs)
                except ImportError:
                    messagebox.showwarning("Brak biblioteki",
                                           "Do obsługi DOCX wymagana jest biblioteka python-docx. Zainstaluj ją używając: pip install python-docx")
                    return

            elif file_path.lower().endswith(('.html', '.htm')):
                try:
                    from bs4 import BeautifulSoup
                    with open(file_path, 'r', encoding='utf-8') as file:
                        soup = BeautifulSoup(file.read(), 'html.parser')
                        file_content = soup.get_text()
                except ImportError:
                    # Jeśli nie ma BeautifulSoup, po prostu wczytaj jako tekst
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                        file_content = file.read()

            else:  # Domyślna obsługa plików tekstowych
                with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                    file_content = file.read()

            # Dodaj plik do listy dołączonych plików
            self.attached_files.append({
                'name': file_name,
                'path': file_path,
                'content': file_content
            })

            # Zaktualizuj listę plików
            self.update_files_list()

            # Informacja o dołączeniu pliku
            messagebox.showinfo("Sukces", f"Plik {file_name} został dołączony do konwersacji.")

        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się wczytać pliku: {str(e)}")

    def update_files_list(self):
        """Aktualizuje listę dołączonych plików."""
        self.files_listbox.delete(0, tk.END)
        for file_info in self.attached_files:
            self.files_listbox.insert(tk.END, file_info['name'])

    def remove_file(self):
        """Usuwa wybrany plik z listy dołączonych plików."""
        selection = self.files_listbox.curselection()
        if selection:
            index = selection[0]
            removed_file = self.attached_files.pop(index)
            self.update_files_list()
            messagebox.showinfo("Informacja", f"Plik {removed_file['name']} został usunięty.")

    def clear_files(self):
        """Usuwa wszystkie dołączone pliki."""
        if self.attached_files:
            self.attached_files = []
            self.update_files_list()
            messagebox.showinfo("Informacja", "Wszystkie dołączone pliki zostały usunięte.")

    def apply_context(self):
        """Zastosowuje kontekst do konwersacji."""
        context = self.context_text.get("1.0", tk.END).strip()
        if context:
            self.add_to_history(f"System: Dodano kontekst konwersacji", "system")
        else:
            messagebox.showinfo("Informacja", "Brak kontekstu do dodania.")

    def clear_context(self):
        """Czyści pole kontekstu."""
        self.context_text.delete("1.0", tk.END)

    def clear_output(self):
        """Czyści pola tekstowe."""
        self.input_text.delete("1.0", tk.END)

        # Pytanie czy wyczyścić historię
        if self.chat_history:
            response = messagebox.askyesno("Potwierdź", "Czy chcesz wyczyścić całą historię czatu?")
            if response:
                self.history_text.config(state="normal")
                self.history_text.delete("1.0", tk.END)
                self.history_text.config(state="disabled")
                self.chat_history = []

    def save_chat_history(self):
        """Zapisuje historię czatu do pliku."""
        if not self.chat_history:
            messagebox.showinfo("Informacja", "Brak historii czatu do zapisania.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Zapisz historię czatu",
            defaultextension=".json",
            filetypes=[("Pliki JSON", "*.json"), ("Wszystkie pliki", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(self.chat_history, file, ensure_ascii=False, indent=2)
                messagebox.showinfo("Sukces", "Historia czatu została zapisana.")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się zapisać historii: {str(e)}")

    def load_chat_history(self):
        """Wczytuje historię czatu z pliku."""
        file_path = filedialog.askopenfilename(
            title="Wczytaj historię czatu",
            filetypes=[("Pliki JSON", "*.json"), ("Wszystkie pliki", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    loaded_history = json.load(file)

                # Sprawdź poprawność formatu
                if not isinstance(loaded_history, list):
                    raise ValueError("Nieprawidłowy format historii.")

                # Wyczyść aktualną historię
                self.history_text.config(state="normal")
                self.history_text.delete("1.0", tk.END)
                self.history_text.config(state="disabled")

                # Wczytaj nową historię
                self.chat_history = loaded_history

                # Wyświetl wczytaną historię
                for entry in self.chat_history:
                    role = entry.get('role', '')
                    content = entry.get('content', '')

                    if role == 'user':
                        self.add_to_history(f"Ty: {content}", "user")
                    elif role == 'assistant':
                        self.add_to_history(f"Model: {content}", "assistant")
                    elif role == 'system':
                        self.add_to_history(f"System: {content}", "system")

                messagebox.showinfo("Sukces", "Historia czatu została wczytana.")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się wczytać historii: {str(e)}")



def run_gui():
    """Uruchamia interfejs graficzny."""
    try:
        root = ThemedTk(theme="equilux")  # Inne dostępne motywy: "equilux", "breeze", "black", "clearlooks"
        app = LLMApp(root)
        root.mainloop()
    except Exception as e:
        import traceback
        print(f"Błąd podczas uruchamiania GUI: {e}")
        print(traceback.format_exc())  # Wyświetli pełny stack trace


if __name__ == "__main__":
    run_gui()