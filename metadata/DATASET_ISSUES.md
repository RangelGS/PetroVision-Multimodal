# Problemas conhecidos e decisões de curadoria

## Conteúdo duplicado com rótulos conflitantes

O manifesto inicial de 336 imagens revelou 335 hashes SHA-256 únicos. Duas
entradas eram idênticas byte a byte, apesar de estarem associadas a modalidades
e classes distintas:

- `PPL/train/class10/Fracture.999.jpg` — Cemented fracture;
- `XPL/train/class13/Micritic_limestone.1837.jpg` — Micritic limestone.

SHA-256 compartilhado:
`912453ff8d9317a3da06b2aed79387cd597d7f8e68101ec46a93f4ca5fcc415d`.

As duas pertenciam ao treino, portanto não havia vazamento para validação ou
teste. Entretanto, o conflito de rótulos introduziria ruído supervisionado e
não existe evidência suficiente para escolher qual anotação é correta.

## Decisão

As duas entradas foram excluídas da seleção, movidas para uma quarentena local
recuperável e substituídas por novas imagens dos mesmos estratos. O pipeline
agora bloqueia a escrita de qualquer manifesto que contenha SHA-256 repetido.

Essa decisão trata somente integridade computacional e não constitui revisão
geológica das classes originais.

## Identificadores reutilizados entre treino e validação

Uma auditoria posterior encontrou dois nomes reutilizados no estrato PPL de
`class17` (Oolite):

- `Oolitic.20`, presente no treino e na validação;
- `Oolitic.3`, presente no treino e na validação.

Os quatro arquivos tinham hashes SHA-256 distintos, portanto não eram cópias
exatas. Ainda assim, a reutilização do identificador poderia representar
imagens relacionadas e influenciar a seleção de hiperparâmetros na validação.
O conjunto de teste não tinha identificadores repetidos.

## Decisão da v0.5.2

As duas entradas de validação foram substituídas deterministicamente por
`Oolitic.12` e `Oolitic.4`. A seleção agora reserva identificadores entre as
divisões e o manifesto, o catálogo e a entrada do modelo bloqueiam qualquer
novo conflito. Uma auditoria perceptual por pHash também sinaliza candidatos
visualmente próximos para revisão manual, sem exclusão automática.

## Duplicata visual entre treino e teste

A triagem pHash da seleção corrigida sinalizou 11 pares. A revisão visual
confirmou 10 falsos positivos causados por fundos escuros, barras de escala ou
layouts semelhantes. Um par representa o mesmo campo petrográfico com diferença
de brilho/processamento:

- `XPL/test/class17/10.jpg`;
- `XPL/train/class17/Oolitic.135.jpg`.

Para preservar o conjunto de teste oficial, a entrada de treino `Oolitic.135`
foi excluída e substituída deterministicamente por `Oolitic.77`. As decisões dos
11 pares estão registradas em `metadata/near_duplicate_review.csv`.
