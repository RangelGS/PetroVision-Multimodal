# PetroVision Multimodal

Prova de conceito para curadoria, análise e classificação de imagens
petrográficas multimodais. O projeto foi planejado para demonstrar competências
em Python científico, visão computacional, modelos fundacionais, aprendizado
auto-supervisionado, integração de dados e reprodutibilidade.

> Estado atual: estrutura inicial funcional. As etapas de catálogo e controle
> de qualidade já estão implementadas. A extração com DINOv2, a fusão PPL/XPL
> e a segmentação com SAM 2 serão executadas depois que o subconjunto de dados
> reais estiver preparado.

## Pergunta de pesquisa

Representações extraídas por um modelo visual auto-supervisionado conseguem
separar categorias petrográficas? A combinação de imagens em luz plano-
polarizada (PPL) e luz polarizada cruzada (XPL) melhora o desempenho em relação
ao uso de apenas uma modalidade?

## Etapas

1. Catalogar imagens, classes, modalidades e divisões de treino/validação/teste.
2. Verificar brilho, contraste e nitidez com OpenCV.
3. Extrair embeddings com DINOv2 usando PyTorch.
4. Agrupar embeddings sem rótulos com HDBSCAN.
5. Treinar um classificador linear sobre os embeddings.
6. Comparar PPL, XPL e fusão PPL + XPL.
7. Produzir máscaras exploratórias com SAM 2 e indicadores quantitativos.
8. Documentar métricas, limitações e reprodutibilidade.

## Estrutura esperada dos dados

```text
data/raw/
├── PPL/
│   ├── train/<classe>/*.jpg
│   ├── val/<classe>/*.jpg
│   └── test/<classe>/*.jpg
└── XPL/
    ├── train/<classe>/*.jpg
    ├── val/<classe>/*.jpg
    └── test/<classe>/*.jpg
```

Imagens PPL e XPL correspondentes devem ter o mesmo nome de arquivo dentro da
mesma classe e divisão. O script de catálogo registra pares encontrados e
também preserva imagens sem par.

## Instalação no Windows 11

Use Python 3.11. Abra a pasta no VSCode e execute, no terminal PowerShell, um
comando por vez:

```powershell
py -3.11 -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
python -m pip install --upgrade pip
```

```powershell
pip install -r requirements.txt
```

```powershell
pip install -e .
```

Se o PowerShell bloquear a ativação, consulte `docs/SETUP_WINDOWS.md`.

## Primeiros comandos

Verificar o ambiente:

```powershell
python scripts/check_environment.py
```

Depois de colocar as imagens em `data/raw`, criar o catálogo:

```powershell
python scripts/build_catalog.py
```

Executar o controle de qualidade:

```powershell
python scripts/run_quality_control.py
```

Rodar os testes do núcleo inicial:

```powershell
pytest
```

## Hardware

O catálogo e o controle de qualidade rodam localmente. Como a GPU AMD RX 7600
não oferece o fluxo CUDA esperado pelo SAM 2 no Windows, as etapas pesadas serão
preparadas para Google Colab com GPU. O VSCode continuará sendo usado para
organização, Git, documentação e desenvolvimento.

## Dados e integridade científica

O conjunto de referência planejado é o DeepCarbonate, publicado com 55.786
imagens, 22 categorias litológicas e diferentes modos ópticos. Para esta prova
de conceito será usado um subconjunto estratificado, preservando as divisões
oficiais e registrando a origem de cada arquivo.

Resultados de segmentação do SAM 2 serão descritos como regiões candidatas, não
como identificação mineral validada. Interpretação geológica exige validação de
especialistas.

## Autoria

Projeto desenvolvido por Rodrigo Rangel Goes e Silva. Bibliotecas, modelos e
dados externos devem permanecer citados no relatório e no repositório.

