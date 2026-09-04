# Atualização v1.0.0

## Fechamento da prova de conceito

A v1.0.0 publica os resultados validados do pipeline DINOv2 e fecha o primeiro
escopo demonstrável do PetroVision Multimodal.

## Resultados incorporados

- 336 imagens e embeddings DINOv2-small de 384 dimensões;
- 4 classes e 2 modalidades ópticas;
- 6 cenários de classificação linear;
- teste oficial mantido fora da seleção de hiperparâmetros;
- 25 avaliações repetidas por cenário, totalizando 150 medições;
- PCA, K-Means, similaridade de protótipos e matrizes de confusão;
- proveniência com commit, revisão do modelo, versões e hashes.

O treinamento combinado PPL+XPL foi a alternativa mais equilibrada na
validação repetida: macro-F1 médio de 0,675 em PPL e 0,650 em XPL. O melhor
macro-F1 no teste oficial foi 0,635, no cenário combinado avaliado em PPL.

## Documentação final

- relatório técnico em Markdown e PDF;
- diário da execução de referência;
- orientação para cadastro no Currículo Lattes;
- roteiro de demonstração de 2 a 3 minutos;
- procedimento de recuperação do Colab por ZIP local;
- README atualizado com métricas e limitações.

## Reprodutibilidade

A execução de referência foi produzida com a v0.6.1 no commit `83ac0b3`. Esse
registro não foi alterado retroativamente: a v1.0.0 publica os artefatos da
execução e atualiza a documentação e a versão do projeto.

## Limite da versão

O projeto usa um DINOv2 pré-treinado e congelado; não treinou o modelo
fundacional. As modalidades PPL e XPL são não pareadas, os rótulos não foram
revalidados por especialista e não há teste externo. Segmentação com SAM 2 é
uma possibilidade futura, não parte da entrega validada da v1.0.0.
