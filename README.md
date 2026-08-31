# PetroVision Multimodal

[![Abrir no Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RangelGS/PetroVision-Multimodal/blob/main/notebooks/PetroVision_DINOv2_Colab.ipynb)

Prova de conceito para curadoria, análise e classificação de imagens
petrográficas multimodais. O projeto foi planejado para demonstrar competências
em Python científico, visão computacional, modelos fundacionais, aprendizado
auto-supervisionado, integração de dados e reprodutibilidade.

> Estado atual — v0.5.0: catálogo, controle de qualidade, preparação dos dados e
> pipeline DINOv2 estão implementados. O projeto seleciona 336 imagens PPL/XPL
> do DeepCarbonate, equilibradas por modalidade, classe e divisão, com
> proveniência e hashes. As 336 foram aceitas após triagem automática e revisão
> visual documentada. A etapa de modelos extrai representações congeladas,
> avalia probes lineares, agrupamentos e alinhamento entre protótipos PPL/XPL.

## Pergunta de pesquisa

Representações extraídas por um modelo visual auto-supervisionado conseguem
separar categorias petrográficas? Elas permanecem consistentes quando o domínio
óptico muda entre luz plano-polarizada (PPL) e luz polarizada cruzada (XPL)?

## Etapas

1. Catalogar imagens, classes, modalidades e divisões de treino/validação/teste.
2. Verificar brilho, contraste e nitidez com OpenCV.
3. Extrair embeddings com DINOv2 usando PyTorch.
4. Agrupar embeddings sem rótulos com HDBSCAN.
5. Treinar classificadores lineares sobre embeddings congelados.
6. Comparar PPL e XPL e medir o alinhamento entre protótipos de classe.
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

O ZIP não fornece uma chave confiável para ligar cada imagem PPL à sua suposta
correspondente XPL. Por isso, as modalidades são tratadas como subconjuntos não
pareados. A seleção é independente, equilibrada e preserva as divisões oficiais.

## Instalação no Windows 11

Use Python 3.11 ou 3.12. Abra a pasta no VSCode e execute, no terminal
PowerShell, um comando por vez. O exemplo abaixo usa a versão 3.12:

```powershell
py -V:3.12 -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
python -m pip install --upgrade pip
```

```powershell
python -m pip install -r requirements.txt
```

```powershell
python -m pip install -e .
```

Se o PowerShell bloquear a ativação, consulte `docs/SETUP_WINDOWS.md`.

## Primeiros comandos

Verificar o ambiente:

```powershell
python scripts/check_environment.py
```

Antes do download, simular a seleção no arquivo remoto:

```powershell
python scripts/download_subset.py --dry-run
```

Se a simulação confirmar 336 imagens, baixar o recorte:

```powershell
python scripts/download_subset.py --yes
```

O download gera `metadata/subset_manifest.csv` com a origem, o tamanho e o
SHA-256 de cada imagem. Depois, criar o catálogo:

```powershell
python scripts/build_catalog.py
```

O manifesto também é auditado contra conteúdos repetidos. Duas entradas
idênticas com rótulos conflitantes foram documentadas, colocadas em quarentena
e substituídas. Consulte `metadata/DATASET_ISSUES.md`.

Executar o controle de qualidade:

```powershell
python scripts/run_quality_control.py
```

Os limites automáticos funcionam como triagem, não como exclusão. Casos
sinalizados devem receber decisão visual em
`metadata/quality_manual_review.csv`. Nesta amostra, 330 imagens passaram
automaticamente e 6 imagens escuras foram mantidas após revisão por preservarem
contraste, textura, bordas e escala. Consulte `metadata/QUALITY_REVIEW.md`.

Rodar os testes do núcleo inicial:

```powershell
python -m pytest
```

## DINOv2 no Google Colab

A execução pesada foi organizada no notebook
[`notebooks/PetroVision_DINOv2_Colab.ipynb`](notebooks/PetroVision_DINOv2_Colab.ipynb).
Abra-o no Google Colab, selecione uma GPU e execute as células em ordem. O
notebook clona o repositório, reconstrói o subconjunto, extrai 336 vetores com o
checkpoint congelado `facebook/dinov2-small` e gera um pacote de resultados.

O pipeline usa o token global `CLS` (384 dimensões), normalização L2 e uma
revisão fixa do checkpoint. Não há fine-tuning do DINOv2. O classificador linear
seleciona seu hiperparâmetro apenas na validação e usa o teste somente na
avaliação final. São executados seis cenários:

- treino PPL com teste PPL e XPL;
- treino XPL com teste XPL e PPL;
- treino combinado PPL+XPL com teste separado em cada modalidade.

Os resultados também incluem PCA, K-Means avaliado contra classe e modalidade,
matriz de similaridade entre protótipos e matrizes de confusão. PPL e XPL
continuam tratados como domínios não pareados. Consulte
[`docs/DINOV2_COLAB.md`](docs/DINOV2_COLAB.md).

Execução manual no Colab, após preparar os dados:

```bash
python scripts/extract_dinov2_embeddings.py --device cuda
python scripts/analyze_dinov2_embeddings.py
```

## Hardware

O catálogo e o controle de qualidade rodam localmente. Como a GPU AMD RX 7600
não oferece o fluxo CUDA esperado pelo SAM 2 no Windows, as etapas pesadas serão
preparadas para Google Colab com GPU. O VSCode continuará sendo usado para
organização, Git, documentação e desenvolvimento.

## Dados e integridade científica

O conjunto de referência é o DeepCarbonate, publicado com 55.786 imagens, 22
categorias litológicas e diferentes modos ópticos. A prova de conceito usa um
subconjunto estratificado das classes `class10` (Cemented fracture), `class13`
(Micritic limestone), `class17` (Oolite) e `class22` (Pore). Para cada classe e
modalidade são selecionadas 30 imagens de treino, 7 de validação e 5 de teste.
Isso totaliza 336 imagens: 168 PPL e 168 XPL. A validação é limitada pelas 7
imagens XPL de Oolite disponíveis; aplicar o mesmo limite a todos os grupos
mantém o recorte equilibrado. Arquivos `_ARS` são excluídos.

O artigo informa que PPL e XPL foram capturadas simultaneamente, mas o ZIP não
publica uma chave de pareamento individual. O projeto não associa imagens por
ordem nem presume que nomes coincidentes sejam pares reais. A análise
multimodal será feita em nível de domínio e de protótipos de classe.

O arquivo de origem possui 30,92 GB, mas o script usa requisições parciais para
transferir somente as imagens selecionadas. O rótulo vem do diretório oficial
`classN` e do `classmap.txt`, não do nome histórico da imagem.

Os dados são disponibilizados pelos autores sob CC BY-NC-ND 4.0. As imagens
brutas ficam fora do Git; este repositório versiona o código, o plano de
seleção, o manifesto de proveniência e os resultados. Consulte
`metadata/DATASET.md`.

Resultados futuros de segmentação do SAM 2 serão descritos como regiões candidatas, não
como identificação mineral validada. Interpretação geológica exige validação de
especialistas.

## Autoria

Projeto desenvolvido por Rodrigo Rangel Goes e Silva. Bibliotecas, modelos e
dados externos devem permanecer citados no relatório e no repositório.
