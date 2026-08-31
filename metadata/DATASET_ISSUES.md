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
