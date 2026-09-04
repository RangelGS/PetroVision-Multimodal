# PetroVision Multimodal

[![Abrir no Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RangelGS/PetroVision-Multimodal/blob/main/notebooks/PetroVision_DINOv2_Colab.ipynb)

Prova de conceito para curadoria, análise e classificação de imagens
petrográficas multimodais. O projeto foi planejado para demonstrar competências
em Python científico, visão computacional, modelos fundacionais, aprendizado
auto-supervisionado, integração de dados e reprodutibilidade.

> Estado atual — v1.0.0: catálogo, controle de qualidade, preparação dos dados e
> pipeline DINOv2 foram implementados e executados. O projeto seleciona 336 imagens PPL/XPL
> do DeepCarbonate, equilibradas por modalidade, classe e divisão, com
> proveniência e hashes. As 336 foram aceitas após triagem automática e revisão
> visual documentada. A etapa de modelos extrai representações congeladas,
> avalia probes lineares, agrupamentos, alinhamento entre protótipos PPL/XPL e
> estabilidade por validação repetida aninhada. Os resultados reproduzíveis da
> execução final estão versionados em [`results/dinov2`](results/dinov2). A
> evolução está descrita em
> [`docs/UPDATE_V0.6.0.md`](docs/UPDATE_V0.6.0.md); a correção de integridade
> permanece documentada em [`docs/UPDATE_V0.5.2.md`](docs/UPDATE_V0.5.2.md) e a
> retomada robusta do Zenodo em [`docs/UPDATE_V0.6.1.md`](docs/UPDATE_V0.6.1.md).
> O fechamento da prova de conceito está em
> [`docs/UPDATE_V1.0.0.md`](docs/UPDATE_V1.0.0.md).

## Pergunta de pesquisa

Representações extraídas por um modelo visual auto-supervisionado conseguem
separar categorias petrográficas? Elas permanecem consistentes quando o domínio
óptico muda entre luz plano-polarizada (PPL) e luz polarizada cruzada (XPL)?

## Etapas

1. Catalogar imagens, classes, modalidades e divisões de treino/validação/teste.
2. Verificar brilho, contraste e nitidez com OpenCV.
3. Extrair embeddings com DINOv2 usando PyTorch.
4. Agrupar embeddings sem rótulos com K-Means.
5. Treinar classificadores lineares sobre embeddings congelados.
6. Comparar PPL e XPL e medir o alinhamento entre protótipos de classe.
7. Documentar métricas, estabilidade, limitações e reprodutibilidade.

Segmentação com SAM 2 permanece como continuação futura e não integra o escopo
validado da v1.0.0.

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

Falhas temporárias `504` do Zenodo recebem até cinco tentativas com espera
progressiva. Uma nova execução reaproveita imagens completas já transferidas e
remove automaticamente qualquer arquivo parcial.

O download gera `metadata/subset_manifest.csv` com a origem, o tamanho e o
SHA-256 de cada imagem. Depois, criar o catálogo:

```powershell
python scripts/build_catalog.py
```

O manifesto também é auditado contra conteúdos repetidos. Duas entradas
idênticas com rótulos conflitantes foram documentadas, colocadas em quarentena
e substituídas. Consulte `metadata/DATASET_ISSUES.md`.

A v0.5.2 também impede que o mesmo `sample_id` seja reutilizado entre treino,
validação e teste dentro da mesma classe e modalidade. Uma triagem por pHash
gera `results/dinov2/tables/near_duplicate_report.csv` para revisão de possíveis
duplicatas visuais; os candidatos não são excluídos automaticamente.
As decisões visuais ficam registradas em
`metadata/near_duplicate_review.csv`.

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
continuam tratados como domínios não pareados.

Além do teste oficial fixo, a v0.6.0 executa validação repetida aninhada somente
em treino+validação: cinco dobras, cinco repetições e seleção interna de `C`.
Isso produz 25 medições por cenário para descrever média e variação sem reutilizar
o teste final. Consulte
[`docs/DINOV2_COLAB.md`](docs/DINOV2_COLAB.md).

## Resultados da execução de referência

A execução de referência da v0.6.1 foi realizada em 3 de setembro de 2026, no
commit `83ac0b3`, com uma Tesla T4. A v1.0.0 preserva o pipeline validado e
publica seus artefatos leves. O teste oficial contém apenas 20 imagens por
modalidade e deve ser interpretado em conjunto com a análise repetida no
conjunto de desenvolvimento.

| Treino | Avaliação | Macro-F1 no teste | Macro-F1 repetido (média ± DP) |
|---|---|---:|---:|
| PPL | PPL | 0,488 | 0,673 ± 0,063 |
| PPL | XPL | 0,581 | 0,587 ± 0,104 |
| PPL+XPL | PPL | **0,635** | 0,675 ± 0,073 |
| PPL+XPL | XPL | 0,557 | 0,650 ± 0,088 |
| XPL | PPL | 0,534 | 0,509 ± 0,078 |
| XPL | XPL | 0,473 | **0,691 ± 0,091** |

O treinamento combinado apresentou o comportamento mais equilibrado entre os
dois domínios na validação repetida. Os agrupamentos não supervisionados foram
fracos (`ARI` de classe 0,129; `silhouette` 0,059), portanto o projeto não
afirma que as classes formem grupos naturais bem separados. Consulte o
[`relatório técnico`](docs/RELATORIO_TECNICO.md) para a interpretação completa.

Execução manual no Colab, após preparar os dados:

```bash
python scripts/extract_dinov2_embeddings.py --device cuda
python scripts/analyze_dinov2_embeddings.py
```

## Hardware

O catálogo e o controle de qualidade rodam localmente. Como a GPU AMD RX 7600
não oferece o fluxo CUDA usado neste experimento, a extração DINOv2 foi
preparada para Google Colab com GPU. O VSCode continua sendo usado para
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

A seleção reserva primeiro os identificadores de treino, depois os de validação
e por fim os de teste. Assim, cada combinação de modalidade, classe e
`sample_id` aparece em apenas uma divisão. O catálogo e a extração de embeddings
repetem essa validação como defesa adicional.

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

Uma continuação futura com SAM 2 deverá descrever as máscaras como regiões
candidatas, não como identificação mineral validada. Interpretação geológica
exige validação de especialistas.

## Entregáveis da v1.0.0

- [`Relatório técnico em Markdown`](docs/RELATORIO_TECNICO.md) e
  [`versão em PDF`](output/pdf/PetroVision_Relatorio_Tecnico_v1.0.pdf)
- [`Orientação para o Currículo Lattes`](docs/LATTES.md)
- [`Roteiro de demonstração de 2 a 3 minutos`](docs/ROTEIRO_DEMONSTRACAO.md)
- [`Relatório automático e resultados`](results/dinov2/DINOV2_REPORT.md)

## Autoria

Projeto desenvolvido por Rodrigo Rangel Goes e Silva. Bibliotecas, modelos e
dados externos devem permanecer citados no relatório e no repositório.
