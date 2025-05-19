# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_dynamic_libs

block_cipher = None

# Znajdź biblioteki DLL dla llama_cpp
llama_cpp_binaries = collect_dynamic_libs('llama_cpp')

a = Analysis(['main.py'],
             pathex=[],
             binaries=llama_cpp_binaries,
             datas=[],
             hiddenimports=['llama_cpp'],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
          cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='SynergiAI',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None)