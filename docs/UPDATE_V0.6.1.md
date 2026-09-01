# Atualização v0.6.1 — retomada robusta do Zenodo

## Motivo

O endpoint de arquivos do Zenodo pode responder temporariamente com
`504 Gateway Time-out` durante as requisições parciais do arquivo DeepCarbonate. A
seleção e os dados permanecem corretos, mas uma única falha interrompia o
download completo.

## Alteração

- cada imagem recebe até cinco tentativas em erros `RemoteIOError`;
- a espera cresce progressivamente entre as tentativas e é limitada a 30
  segundos;
- arquivos são escritos primeiro com o sufixo `.part` e promovidos atomicamente
  somente depois da conferência de tamanho;
- arquivos parciais são removidos mesmo quando todas as tentativas falham;
- uma nova execução preserva e verifica imagens completas já baixadas.

Os parâmetros `download_retry_attempts` e `download_retry_backoff_seconds`
ficam declarados em `config.yaml`. Testes automatizados simulam um erro `504`,
confirmam a retomada e verificam a limpeza após esgotar as tentativas.
