# Atualização v0.6.0 — estabilidade dos probes lineares

## Objetivo

A versão 0.6.0 adiciona uma análise secundária de estabilidade sem alterar o
resultado principal no teste oficial. A pergunta é se as conclusões dos probes
lineares permanecem semelhantes quando as imagens de treino e validação são
redistribuídas de forma estratificada.

## Protocolo

- somente as divisões `train` e `val` participam;
- o teste oficial é ignorado por construção;
- são usadas 5 dobras externas e 5 repetições, totalizando 25 medições por
  cenário;
- cada dobra externa seleciona `C` por validação interna de 3 dobras;
- PPL e XPL recebem divisões estratificadas independentes porque não há chave de
  pareamento individual confiável;
- são avaliados os mesmos seis cenários intra, cross-domain e combinados.

## Novos artefatos

- `results/dinov2/tables/repeated_probe_fold_metrics.csv`;
- `results/dinov2/tables/repeated_probe_summary.csv`;
- `results/dinov2/figures/repeated_probe_stability.png`.

O relatório automático inclui a média, o desvio-padrão e a faixa de macro-F1.
Como as dobras repetidas se sobrepõem, o desvio-padrão é um diagnóstico
descritivo de estabilidade, não um intervalo de confiança inferencial.

## Reprodutibilidade

Os parâmetros ficam em `config.yaml` e a semente continua fixa em 42. A lógica
foi adicionada a `src/petrovision/embedding_analysis.py` e coberta por testes que
confirmam explicitamente que alterações nas linhas de teste não afetam a
validação repetida.
