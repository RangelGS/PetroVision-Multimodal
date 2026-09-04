# Plano de execução até a candidatura

## Estado da primeira versão

Os marcos necessários para a primeira versão demonstrável foram concluídos. A
v1.0.0 fecha uma prova de conceito DINOv2 reproduzível; segmentação com SAM 2
permanece como continuação futura e não bloqueia a publicação.

## Marco 1 - Base reprodutível: concluído

- ambiente Python, testes e Git configurados;
- origem e licença dos dados documentadas;
- catálogo, manifesto, hashes e controle de qualidade implementados;
- auditoria contra vazamento de identidades e conteúdo entre divisões.

## Marco 2 - Representações visuais: concluído

- subconjunto estratificado de 336 imagens preparado;
- embeddings DINOv2-small extraídos no Colab;
- PCA bidimensional produzido;
- K-Means avaliado com ARI, NMI e silhouette;
- checkpoint, versões, hashes e ambiente registrados.

O plano inicial mencionava UMAP e HDBSCAN. A implementação final usa PCA e
K-Means, suficientes para o diagnóstico exploratório deste recorte e mais
simples de reproduzir.

## Marco 3 - Comparação multimodal: concluído

- classificadores lineares avaliados em seis cenários;
- PPL, XPL e treinamento combinado comparados;
- macro-F1, acurácia balanceada e matrizes de confusão gerados;
- validação repetida aninhada executada com 25 medições por cenário;
- protótipos de classe comparados entre PPL e XPL.

As modalidades não são pareadas individualmente. Portanto, o projeto não
afirma fusão de pares petrográficos; usa treinamento combinado e comparação em
nível de domínio e de protótipo.

## Marco 4 - Segmentação exploratória: continuação futura

- avaliar SAM 2 em um recorte separado;
- tratar máscaras como regiões candidatas, não como minerais validados;
- submeter qualquer interpretação geológica a especialista.

## Marco 5 - Evidências para a seleção: concluído

- README e diário experimental revisados;
- relatório técnico curto produzido em Markdown e PDF;
- roteiro de demonstração de dois a três minutos preparado;
- versão estável v1.0.0 preparada;
- orientação de cadastro responsável no Currículo Lattes documentada.
