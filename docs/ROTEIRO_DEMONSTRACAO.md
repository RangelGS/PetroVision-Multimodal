# Roteiro de demonstração - 2 a 3 minutos

## 0:00-0:20 - abertura

**Tela:** página inicial do repositório.

**Fala:**

> Este é o PetroVision Multimodal, uma prova de conceito reproduzível para
> curadoria e análise de imagens petrográficas PPL e XPL. O objetivo é avaliar
> se representações DINOv2 mantêm informação útil quando o modo óptico muda.

## 0:20-0:50 - dados e integridade

**Tela:** tabela do README e arquivos `metadata/DATASET.md` e
`metadata/near_duplicate_review.csv`.

**Fala:**

> Selecionei 336 imagens do DeepCarbonate, distribuídas igualmente entre PPL e
> XPL e quatro classes. O pipeline registra origem e SHA-256, bloqueia conteúdo
> repetido entre treino, validação e teste e exige revisão visual dos candidatos
> detectados por pHash. Todas as 336 imagens foram aprovadas.

## 0:50-1:20 - método

**Tela:** notebook Colab e diagrama textual do README.

**Fala:**

> Usei o DINOv2-small pré-treinado com pesos congelados. Cada imagem gera um
> vetor CLS de 384 dimensões, normalizado por L2. Sobre esses vetores treinei
> regressões logísticas em seis cenários: dentro de cada modalidade,
> transferência entre modalidades e treinamento combinado. O teste oficial
> ficou separado da seleção de hiperparâmetros.

## 1:20-1:55 - resultados

**Tela:** `linear_probe_confusions.png` e tabela de métricas.

**Fala:**

> No teste oficial, o melhor macro-F1 foi 0,635 no treinamento combinado
> avaliado em PPL. Como há somente cinco imagens por classe em cada modalidade,
> também executei validação repetida aninhada. O treinamento combinado obteve
> macro-F1 médio de 0,675 em PPL e 0,650 em XPL, sendo a alternativa mais
> equilibrada.

## 1:55-2:25 - estabilidade e limites

**Tela:** `repeated_probe_stability.png`, PCA e protótipos.

**Fala:**

> A variação entre repetições mostra que o conjunto ainda é pequeno. O K-Means
> apresentou agrupamento fraco, apesar de existir sinal para classificação
> linear. A similaridade entre protótipos PPL e XPL da mesma classe foi maior do
> que entre classes diferentes, mas a margem é limitada. Portanto, não interpreto
> o resultado como diagnóstico geológico nem como generalização universal.

## 2:25-2:45 - encerramento

**Tela:** estrutura do repositório, testes e relatório PDF.

**Fala:**

> A entrega principal é o fluxo auditável: dados, testes, notebook Colab,
> proveniência, métricas, figuras e relatório técnico. O projeto aplica um
> modelo existente de forma reprodutível; não criou nem treinou o DINOv2.

## Checklist antes de gravar

- usar resolução mínima de 1080p;
- aumentar o zoom do navegador e do editor;
- ocultar notificações e dados pessoais;
- ensaiar até ficar abaixo de três minutos;
- não mostrar o ZIP bruto das imagens;
- publicar legenda ou transcrição junto ao vídeo.
