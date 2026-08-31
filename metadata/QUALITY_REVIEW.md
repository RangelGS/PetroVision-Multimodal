# Revisão de qualidade das imagens

## Resultado

As 336 imagens foram lidas com sucesso. A triagem automática marcou seis casos
por `baixa_luminosidade`; nenhum deles apresentou baixo contraste, possível
desfoque ou erro de leitura.

As seis imagens foram inspecionadas visualmente em resolução original. Todas
preservam estruturas, bordas, textura e escala, portanto foram mantidas. Essa
decisão avalia somente a qualidade visual e não substitui validação geológica
dos rótulos.

## Decisão metodológica

O limite de brilho permanece como alerta sensível, mas deixa de representar
reprovação automática. Imagens petrográficas podem ser legitimamente escuras em
função da modalidade óptica e da própria classe. Quatro alertas pertencem à
classe de poros, mostrando que uma exclusão automática poderia introduzir viés.

O pipeline agora diferencia:

- `automatic_pass`: passou por todos os limites automáticos;
- `manual_keep`: recebeu alerta e foi mantida após inspeção visual;
- `manual_exclude`: rejeitada após inspeção documentada;
- `pending_review`: alerta ainda sem decisão;
- `read_error`: arquivo ilegível.

As decisões reproduzíveis estão em `metadata/quality_manual_review.csv`. As
imagens originais e a galeria de revisão permanecem fora do Git.
