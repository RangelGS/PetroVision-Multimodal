# DINOv2 no Google Colab

## Objetivo

Esta etapa verifica se um modelo visual auto-supervisionado, sem treinamento
adicional, produz representações úteis das quatro categorias petrográficas do
recorte. Também mede quanto a mudança óptica entre PPL e XPL altera essas
representações.

O checkpoint escolhido é `facebook/dinov2-small`. Ele possui 22,8 milhões de
parâmetros e gera um token global `CLS` de 384 dimensões para cada imagem. Os
pesos ficam congelados: o projeto extrai características e treina apenas
classificadores lineares sobre elas.

## Por que usar o Colab

O catálogo e o controle de qualidade continuam locais. A extração DINOv2 usa
PyTorch com CUDA e é executada no Google Colab para evitar as limitações do fluxo
AMD/Windows. O recorte tem apenas 336 imagens e cabe confortavelmente na GPU
gratuita típica do Colab.

## Execução guiada

1. Acesse <https://colab.research.google.com/>.
2. Escolha **Arquivo > Abrir notebook > GitHub**.
3. Cole `https://github.com/RangelGS/PetroVision-Multimodal`.
4. Abra `notebooks/PetroVision_DINOv2_Colab.ipynb`.
5. Selecione **Ambiente de execução > Alterar tipo de ambiente de execução**.
6. Em acelerador de hardware, escolha **GPU T4** ou outra GPU disponível.
7. Execute uma célula por vez, de cima para baixo, usando o botão de reprodução.

O download seletivo das 336 imagens pode levar aproximadamente 15 a 25 minutos.
Não feche a aba durante essa célula. O download do checkpoint ocorre apenas na
primeira execução de uma sessão.

O downloader tenta novamente erros temporários do Zenodo e usa arquivos `.part`
para impedir que uma transferência interrompida seja tratada como imagem válida.
Ao repetir a célula, imagens completas já baixadas são reaproveitadas.

### Recuperação quando a sessão do Colab expirar

O armazenamento de `/content` é temporário. Se a sessão desaparecer durante o
download, mas as 336 imagens já existirem no computador local, compacte somente
`data/raw` no PowerShell:

```powershell
Compress-Archive -LiteralPath "data\raw" -DestinationPath "$env:USERPROFILE\Downloads\PetroVision_DeepCarbonate_336_clean.zip"
```

No Colab, execute as células de instalação e testes, envie esse ZIP com
`files.upload()` e extraia preservando a estrutura relativa. A normalização de
`\\` é necessária para ZIPs criados no Windows:

```python
from google.colab import files
from pathlib import Path
from zipfile import ZipFile
import shutil

enviados = files.upload()
arquivo_zip = Path(next(iter(enviados)))

with ZipFile(arquivo_zip) as arquivo:
    for membro in arquivo.infolist():
        nome = membro.filename.replace("\\", "/")
        partes = Path(nome).parts
        if not partes or partes[0] != "raw" or ".." in partes:
            raise ValueError(f"Caminho inesperado no ZIP: {nome}")
        destino = Path("data").joinpath(*partes)
        if membro.is_dir():
            destino.mkdir(parents=True, exist_ok=True)
        else:
            destino.parent.mkdir(parents=True, exist_ok=True)
            with arquivo.open(membro) as origem, destino.open("wb") as saida:
                shutil.copyfileobj(origem, saida)

imagens = [
    caminho
    for caminho in Path("data/raw").rglob("*")
    if caminho.suffix.lower() in {".jpg", ".jpeg"}
]
print("Imagens extraídas:", len(imagens))
assert len(imagens) == 336
```

Com a mensagem `Imagens extraídas: 336`, pule a etapa de reconstrução remota e
continue no controle de qualidade. O ZIP contém dados brutos, não deve ser
adicionado ao Git e continua sujeito à licença declarada pelo DeepCarbonate.

## Artefatos gerados

O notebook baixa `PetroVision_DINOv2_results.zip`, contendo:

- `DINOV2_REPORT.md`: resumo automático das métricas;
- `tables/linear_probe_metrics.csv`: resultados intra e cross-domain;
- `tables/linear_probe_predictions.csv`: previsões de teste auditáveis;
- `tables/repeated_probe_fold_metrics.csv`: 25 medições por cenário no
  conjunto de desenvolvimento;
- `tables/repeated_probe_summary.csv`: média, desvio-padrão e faixa observada;
- `tables/clustering_metrics.csv`: ARI, NMI e silhouette;
- `tables/prototype_similarity.csv`: cosseno entre protótipos PPL/XPL;
- `tables/pca_coordinates.csv`: projeção reprodutível;
- `tables/near_duplicate_report.csv`: triagem pHash entre divisões;
- `figures/pca_class_mode.png`: visualização por classe e modalidade;
- `figures/prototype_similarity.png`: mapa de alinhamento de protótipos;
- `figures/repeated_probe_stability.png`: distribuição do macro-F1 repetido;
- `figures/linear_probe_confusions.png`: matrizes de confusão.

O arquivo de embeddings completo permanece fora do Git porque é um artefato
derivado e reproduzível. A revisão fixa do checkpoint, as versões das
bibliotecas e os hashes da execução são registrados em
`results/dinov2/tables/dinov2_embedding_run.json` durante a sessão. Esse registro
também inclui a versão do PetroVision, o commit Git e os hashes do `config.yaml`
e do manifesto do subconjunto.

## Protocolo experimental

- **Entrada:** somente imagens aprovadas pelo controle de qualidade.
- **Representação:** token `CLS` do DINOv2-small, seguido de normalização L2.
- **Seleção de hiperparâmetro:** macro-F1 na validação.
- **Ajuste final:** treino + validação da modalidade-fonte.
- **Avaliação:** teste oficial, nunca usado para escolher o classificador.
- **Estabilidade:** validação repetida aninhada com 5 dobras externas, 5
  repetições e 3 dobras internas, limitada a treino+validação.
- **Cenários:** PPL→PPL, PPL→XPL, XPL→XPL, XPL→PPL e combinado→cada domínio.
- **Não supervisionado:** K-Means com quatro grupos, comparado separadamente com
  rótulos de classe e de modalidade.
- **Multimodal:** similaridade entre médias de classe PPL e XPL, sem supor pares
  individuais inexistentes.

## Interpretação responsável

Um macro-F1 alto indica que as categorias do recorte são linearmente separáveis
nas representações congeladas; não demonstra generalização para outros
laboratórios, equipamentos ou populações de rochas. Diferenças entre resultados
intra e cross-domain indicam sensibilidade ao modo óptico. Os rótulos vêm do
DeepCarbonate e não foram revalidados por um especialista neste projeto.
O desvio-padrão das repetições é descritivo: as dobras se sobrepõem e não devem
ser interpretadas como 25 experimentos independentes nem como intervalo de
confiança do desempenho em outras bases.

## Referências técnicas

- DINOv2: <https://arxiv.org/abs/2304.07193>
- Implementação oficial: <https://github.com/facebookresearch/dinov2>
- Checkpoint: <https://huggingface.co/facebook/dinov2-small>
- Documentação Transformers: <https://huggingface.co/docs/transformers/model_doc/dinov2>
