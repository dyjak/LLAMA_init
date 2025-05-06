import sys
import os
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from typing import Optional

from llm_interface import SimpleLLMInterface
import config


class LLMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple LLM Interface")
        self.root.geometry("900x700")

        self.interface = SimpleLLMInterface()

        self.model_loaded = False
        self.stream_output = tk.BooleanVar(value=True)
        self.temperature = tk.DoubleVar(value=config.DEFAULT_TEMPERATURE)
        self.max_tokens = tk.IntVar(value=config.DEFAULT_MAX_TOKENS)
        self.mode = tk.StringVar(value="chat")
        self.system_prompt = tk.StringVar(value=config.DEFAULT_SYSTEM_PROMPT)

        self.create_model_frame()
        self.create_settings_frame()
        self.create_input_frame()
        self.create_output_frame()
        self.create_control_buttons()

    def create_model_frame(self):
        model_frame = ttk.LabelFrame(self.root, text="Model")
        model_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(model_frame, text="Załaduj lokalny model",
                   command=self.load_local_model).pack(side="left", padx=5, pady=5)

        ttk.Button(model_frame, text="Wybierz predefiniowany model",
                   command=self.show_predefined_models).pack(side="left", padx=5, pady=5)

        self.model_info_label = ttk.Label(model_frame, text="Brak załadowanego modelu")
        self.model_info_label.pack(side="left", padx=10, pady=5)

    def create_settings_frame(self):
        settings_frame = ttk.LabelFrame(self.root, text="Ustawienia")
        settings_frame.pack(fill="x", padx=10, pady=5)

        mode_frame = ttk.Frame(settings_frame)
        mode_frame.pack(fill="x", padx=5, pady=2)

        ttk.Label(mode_frame, text="Tryb:").pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="Chat", variable=self.mode,
                        value="chat", command=self.toggle_system_prompt).pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="Complete", variable=self.mode,
                        value="complete", command=self.toggle_system_prompt).pack(side="left", padx=5)

        self.system_prompt_frame = ttk.Frame(settings_frame)
        self.system_prompt_frame.pack(fill="x", padx=5, pady=2)

        ttk.Label(self.system_prompt_frame, text="System Prompt:").pack(side="left", padx=5)
        system_entry = ttk.Entry(self.system_prompt_frame, textvariable=self.system_prompt, width=50)
        system_entry.pack(side="left", padx=5, fill="x", expand=True)

        params_frame = ttk.Frame(settings_frame)
        params_frame.pack(fill="x", padx=5, pady=2)

        ttk.Label(params_frame, text="Temperatura:").pack(side="left", padx=5)
        temp_scale = ttk.Scale(params_frame, from_=0.0, to=1.0, variable=self.temperature,
                               orient="horizontal", length=150)
        temp_scale.pack(side="left", padx=5)
        ttk.Label(params_frame, textvariable=self.temperature).pack(side="left", padx=5)

        ttk.Label(params_frame, text="Max Tokenów:").pack(side="left", padx=5)
        tokens_combo = ttk.Combobox(params_frame, textvariable=self.max_tokens,
                                    values=[128, 256, 512, 1024, 2048], width=5)
        tokens_combo.pack(side="left", padx=5)

        ttk.Checkbutton(params_frame, text="Stream Output", variable=self.stream_output).pack(side="left", padx=10)

    def create_input_frame(self):
        input_frame = ttk.LabelFrame(self.root, text="Wprowadź prompt")
        input_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.input_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=10)
        self.input_text.pack(fill="both", expand=True, padx=5, pady=5)

    def create_output_frame(self):
        output_frame = ttk.LabelFrame(self.root, text="Wygenerowany tekst")
        output_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=10)
        self.output_text.pack(fill="both", expand=True, padx=5, pady=5)

    def create_control_buttons(self):
        buttons_frame = ttk.Frame(self.root)
        buttons_frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(buttons_frame, text="Generuj", command=self.generate_text).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="Wyczyść", command=self.clear_output).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="Wyjdź", command=self.root.quit).pack(side="right", padx=5)

    def toggle_system_prompt(self):
        state = "normal" if self.mode.get() == "chat" else "disabled"
        for child in self.system_prompt_frame.winfo_children():
            child.configure(state=state)

    def load_local_model(self):
        file_path = filedialog.askopenfilename(
            title="Wybierz plik modelu",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")]
        )

        if file_path:
            self.load_model(model_path=file_path)

    def show_predefined_models(self):
        models_window = tk.Toplevel(self.root)
        models_window.title("Wybierz predefiniowany model")
        models_window.geometry("600x400")

        models = config.PREDEFINED_MODELS

        list_frame = ttk.Frame(models_window)
        list_frame.pack(fill="both", expand=True, padx=10, pady=10)

        columns = ("repo_id", "filename")
        tree = ttk.Treeview(list_frame, columns=columns, show="headings")
        tree.heading("repo_id", text="Repozytorium")
        tree.heading("filename", text="Nazwa pliku")
        tree.column("repo_id", width=300)
        tree.column("filename", width=250)

        for model in models:
            tree.insert("", "end", values=(model["repo_id"], model["filename"]))

        tree.pack(fill="both", expand=True)

        def load_selected_model():
            selection = tree.selection()
            if selection:
                item = tree.item(selection[0])
                values = item["values"]
                repo_id, filename = values
                models_window.destroy()
                self.load_model(repo_id=repo_id, filename=filename)

        ttk.Button(models_window, text="Załaduj wybrany model",
                   command=load_selected_model).pack(pady=10)
        ttk.Button(models_window, text="Anuluj",
                   command=models_window.destroy).pack(pady=5)

    def load_model(self, model_path: Optional[str] = None, repo_id: Optional[str] = None,
                   filename: Optional[str] = None):
        try:
            self.model_info_label.config(text="Ładowanie modelu...")
            self.root.update()

            params = {
                "context_size": config.DEFAULT_CONTEXT_SIZE,
                "n_gpu_layers": config.DEFAULT_N_GPU_LAYERS,
                "n_threads": config.DEFAULT_N_CPU_THREADS
            }

            def load_in_thread():
                success = self.interface.load_model(
                    model_path=model_path,
                    repo_id=repo_id,
                    filename=filename,
                    **params
                )
                self.root.after(0, self.update_model_info, success)

            thread = threading.Thread(target=load_in_thread)
            thread.daemon = True
            thread.start()

        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się załadować modelu: {str(e)}")
            self.model_info_label.config(text="Brak załadowanego modelu")

    def update_model_info(self, success: bool):
        if success:
            model_info = self.interface.model.get_info()
            self.model_info_label.config(text=f"Model: {model_info['model_name']}")
            self.model_loaded = True
            messagebox.showinfo("Sukces", "Model został pomyślnie załadowany")
        else:
            self.model_info_label.config(text="Brak załadowanego modelu")
            self.model_loaded = False

    def generate_text(self):
        if not self.model_loaded:
            messagebox.showwarning("Ostrzeżenie", "Najpierw załaduj model!")
            return

        prompt = self.input_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Ostrzeżenie", "Wprowadź prompt!")
            return

        self.output_text.delete("1.0", tk.END)
        self.root.update()

        params = {
            "max_tokens": self.max_tokens.get(),
            "temperature": self.temperature.get(),
            "stream": self.stream_output.get()
        }

        def generate_in_thread():
            mode = self.mode.get()

            if mode == "chat":
                system_prompt = self.system_prompt.get()
                if self.stream_output.get():
                    for chunk in self.get_response_stream(prompt, system_prompt):
                        self.update_output(chunk)
                else:
                    response = self.interface.chat(
                        prompt,
                        system_prompt=system_prompt,
                        stream=False,
                        **params
                    )
                    self.root.after(0, lambda: self.output_text.insert(tk.END, response))
            else:
                if self.stream_output.get():
                    for chunk in self.get_completion_stream(prompt):
                        self.update_output(chunk)
                else:
                    response = self.interface.complete(
                        prompt,
                        stream=False,
                        **params
                    )
                    if isinstance(response, dict):
                        response = response["choices"][0]["text"]
                    self.root.after(0, lambda: self.output_text.insert(tk.END, response))

        thread = threading.Thread(target=generate_in_thread)
        thread.daemon = True
        thread.start()

    def get_response_stream(self, prompt, system_prompt):
        formatted_prompt = f"""<s>[INST] {system_prompt}\n\n{prompt} [/INST]"""
        return self.interface.model.generate(
            formatted_prompt,
            max_tokens=self.max_tokens.get(),
            temperature=self.temperature.get(),
            stream=True
        )

    def get_completion_stream(self, prompt):
        return self.interface.model.generate(
            prompt,
            max_tokens=self.max_tokens.get(),
            temperature=self.temperature.get(),
            stream=True
        )

    def update_output(self, text):
        self.root.after(0, lambda: self.output_text.insert(tk.END, text))
        self.root.after(0, lambda: self.output_text.see(tk.END))

    def clear_output(self):
        self.input_text.delete("1.0", tk.END)
        self.output_text.delete("1.0", tk.END)


# 🔧 Poprawna, niezagnieżdżona funkcja uruchamiająca GUI
def run_gui():
    root = tk.Tk()
    app = LLMApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
