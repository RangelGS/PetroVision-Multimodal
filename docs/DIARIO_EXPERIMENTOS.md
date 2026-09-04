# Diário de experimentos

Registre cada execução importante. Não apague resultados ruins: explique o que
foi aprendido e qual alteração foi feita depois.

## Modelo de registro

- Data e hora:
- Objetivo:
- Versão do código/commit:
- Dados e quantidade de imagens:
- Divisão treino/validação/teste:
- Modelo e parâmetros:
- Semente aleatória:
- Hardware/ambiente:
- Métricas:
- Resultado observado:
- Problemas ou possíveis vieses:
- Próxima ação:

## Execução de referência - fechamento v1.0.0

- Data e hora: 3 de setembro de 2026, 22:03 UTC.
- Objetivo: avaliar representações DINOv2 em quatro classes petrográficas e
  medir estabilidade entre os domínios ópticos PPL e XPL.
- Versão do código/commit: PetroVision 0.6.1, commit
  `83ac0b325f81c2f4722593c6df42f196353840de`; resultados incorporados à versão
  documental 1.0.0.
- Dados: 336 imagens DeepCarbonate aprovadas, sendo 168 PPL e 168 XPL.
- Divisão por modalidade: 120 treino, 28 validação e 20 teste.
- Modelo: `facebook/dinov2-small`, revisão
  `ed25f3a31f01632728cabb09d1542f84ab7b0056`, pesos congelados, token CLS de
  384 dimensões e normalização L2.
- Ambiente: Google Colab, CUDA, Tesla T4, PyTorch 2.11.0+cu128 e Transformers
  5.16.1.
- Avaliação oficial: seis cenários; melhor macro-F1 0,635 no treino combinado
  avaliado em PPL.
- Estabilidade: 25 medições por cenário, total de 150; treino combinado com
  macro-F1 médio 0,675 em PPL e 0,650 em XPL.
- Agrupamento: ARI de classe 0,129, NMI de classe 0,186 e silhouette 0,059.
- Alinhamento: similaridade média de protótipos da mesma classe 0,953, contra
  0,874 entre classes diferentes; gap 0,079.
- Resultado observado: existe sinal útil para classificação linear, mas não há
  separação natural forte dos embeddings. O treino combinado foi a alternativa
  mais equilibrada no recorte.
- Limitações: apenas quatro classes, cinco imagens de teste por classe e modo,
  uma única fonte, PPL/XPL não pareados e ausência de validação petrográfica
  independente.
- Próxima ação: publicar a v1.0.0, gravar a demonstração curta e registrar a
  produção técnica no Currículo Lattes com linguagem responsável.
