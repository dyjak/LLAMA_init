from PyInstaller.utils.hooks import collect_all

hiddenimports = ['llama_cpp', 'PyPDF2', 'docx', 'bs4', 'ttkthemes', 'json', 'multiprocessing']

datas, binaries, hiddenimports_llama = collect_all('llama_cpp')
hiddenimports.extend(hiddenimports_llama)


#pyinstaller --onefile --windowed --icon=icon.ico --name=synergiAI --additional-hooks-dir=setup.py --debug=all main.py
