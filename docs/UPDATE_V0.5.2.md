# Atualização v0.5.2 — integridade entre divisões

## Motivo

A auditoria da primeira execução DINOv2 encontrou dois `sample_id` reutilizados
entre treino e validação no estrato PPL de `class17` (Oolite): `Oolitic.20` e
`Oolitic.3`. Os hashes eram diferentes e o teste estava limpo, mas a possível
relação entre as imagens poderia influenciar a escolha do hiperparâmetro na
validação.

## Correção

- reserva determinística de identificadores na ordem treino, validação e teste;
- substituição das duas entradas de validação por `Oolitic.12` e `Oolitic.4`;
- bloqueio de conflitos no manifesto, no catálogo e na entrada do modelo;
- quarentena recuperável de arquivos pertencentes à seleção anterior;
- triagem pHash de possíveis duplicatas visuais entre divisões;
- revisão documentada de 11 candidatos pHash e substituição da duplicata visual
  de treino `XPL/class17/Oolitic.135` por `Oolitic.77`;
- registro de versão, commit e hashes de configuração e manifesto na execução;
- testes automatizados para todos esses contratos.

O pHash é usado somente para triagem. Imagens sinalizadas precisam de revisão
visual e não são excluídas automaticamente.

## Validação

```text
25 passed
336 imagens selecionadas
336 hashes SHA-256 únicos
0 identificadores reutilizados entre divisões
```

Como a validação mudou, os resultados DINOv2 da v0.5.1 permanecem apenas como
execução preliminar. O notebook deve ser executado novamente antes da publicação
do baseline definitivo.
