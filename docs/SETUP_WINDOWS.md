# Configuração no Windows 11

## 1. Verificar Python

Abra o VSCode, use **Terminal > New Terminal** e execute:

```powershell
py --version
```

O recomendado é Python 3.11. Se o comando não funcionar, instale pelo site
oficial `python.org` e marque a opção para adicionar o Python ao PATH.

## 2. Extensões do VSCode

Instale as extensões oficiais da Microsoft:

- Python;
- Jupyter.

## 3. Ambiente virtual

Na pasta do projeto:

```powershell
py -3.11 -m venv .venv
```

Ative:

```powershell
.\.venv\Scripts\Activate.ps1
```

Se aparecer bloqueio de execução, use apenas para a sessão atual:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Depois repita a ativação.

## 4. Instalar e testar

```powershell
python -m pip install --upgrade pip
```

```powershell
pip install -r requirements.txt
```

```powershell
pip install -e .
```

```powershell
python scripts/check_environment.py
```

```powershell
pytest
```

## 5. GPU

A RX 7600 continuará útil para aplicações gráficas, mas o caminho mais estável
para DINOv2 e SAM 2 neste projeto será o Google Colab com GPU NVIDIA. Não tente
instalar CUDA para a placa AMD. O ambiente local será usado para as etapas
leves e para manter o repositório organizado.

