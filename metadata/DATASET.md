# Dados: DeepCarbonate

## Fonte e licença

- Registro: DeepCarbonate, Zenodo, DOI `10.5281/zenodo.18061204`.
- Artigo: *DeepCarbonate: a global petrographic dataset of carbonate rocks for
  computer vision and geoscience applications*, Scientific Data (2026).
- Repositório dos autores: `https://github.com/ai4geology/DeepCarbonate`.
- Licença declarada para os dados: CC BY-NC-ND 4.0.

As imagens originais não são redistribuídas neste repositório. O diretório
`data/raw` está no `.gitignore`. O projeto versiona somente código, critérios de
seleção, proveniência, hashes e resultados derivados permitidos pela licença.

## Recorte desta prova de conceito

O rótulo é determinado pelo diretório oficial `classN` e pelo `classmap.txt` do
arquivo, nunca pelo texto histórico presente no nome da imagem.

| Classe | Rótulo oficial | Relação com o edital | Imagens PPL + XPL |
|---|---|---|---:|
| class10 | Cemented fracture | cimento/fratura | 84 |
| class13 | Micritic limestone | matriz | 84 |
| class17 | Oolite | grão | 84 |
| class22 | Pore | poro | 84 |

Por classe e por modalidade, são selecionadas 30 imagens de treino, 7 de
validação e 5 de teste. Isso totaliza 336 imagens. O subconjunto XPL de Oolite
possui somente 7 imagens de validação; aplicar o mesmo limite aos demais grupos
evita desbalanceamento e não move amostras entre divisões. A seleção preserva
as divisões oficiais, usa semente 42 e exclui `_ARS`.

## Modalidades não pareadas

O artigo relata aquisição simultânea em PPL e XPL, mas o ZIP reorganizado em
formato ImageNet não fornece um identificador confiável da dupla. A inspeção do
arquivo encontrou correspondências nominais raras e insuficientes. Ordenar as
pastas ou associar nomes semelhantes criaria pares sem evidência documental.

Portanto, PPL e XPL são tratados como domínios não pareados. O estudo compara
desempenho por modalidade, separação dos embeddings e proximidade dos protótipos
de cada classe. Fusão por instância somente será realizada se uma fonte futura
fornecer a tabela de correspondência original.

O comando `python scripts/download_subset.py --dry-run` valida a disponibilidade
estratificada no arquivo remoto sem baixar as imagens. O download efetivo gera
`metadata/subset_manifest.csv` com origem, caminho local, tamanho e SHA-256.
