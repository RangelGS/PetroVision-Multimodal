# Relatório automático — DINOv2

## Classificadores lineares

| training_domain   | evaluation_mode   | relation      |   selected_c |   validation_macro_f1 |   accuracy |   balanced_accuracy |   macro_f1 |   fit_samples |   test_samples |
|:------------------|:------------------|:--------------|-------------:|----------------------:|-----------:|--------------------:|-----------:|--------------:|---------------:|
| PPL               | PPL               | within_domain |        0.010 |                 0.554 |      0.550 |               0.550 |      0.488 |           148 |             20 |
| PPL               | XPL               | cross_domain  |        0.010 |                 0.554 |      0.600 |               0.600 |      0.581 |           148 |             20 |
| PPL+XPL           | PPL               | combined      |       10.000 |                 0.644 |      0.650 |               0.650 |      0.635 |           296 |             20 |
| PPL+XPL           | XPL               | combined      |       10.000 |                 0.644 |      0.550 |               0.550 |      0.557 |           296 |             20 |
| XPL               | PPL               | cross_domain  |        0.010 |                 0.623 |      0.550 |               0.550 |      0.534 |           148 |             20 |
| XPL               | XPL               | within_domain |        0.010 |                 0.623 |      0.550 |               0.550 |      0.473 |           148 |             20 |

## Estabilidade no conjunto de desenvolvimento

| training_domain   | evaluation_mode   | relation      |   stability_runs |   selected_c_median |   inner_validation_macro_f1_mean |   accuracy_mean |   accuracy_std |   balanced_accuracy_mean |   balanced_accuracy_std |   macro_f1_mean |   macro_f1_std |   macro_f1_min |   macro_f1_max |
|:------------------|:------------------|:--------------|-----------------:|--------------------:|---------------------------------:|----------------:|---------------:|-------------------------:|------------------------:|----------------:|---------------:|---------------:|---------------:|
| PPL               | PPL               | within_domain |               25 |               1.000 |                            0.653 |           0.681 |          0.059 |                    0.681 |                   0.061 |           0.673 |          0.063 |          0.524 |          0.765 |
| PPL               | XPL               | cross_domain  |               25 |               1.000 |                            0.653 |           0.595 |          0.103 |                    0.595 |                   0.104 |           0.587 |          0.104 |          0.343 |          0.850 |
| PPL+XPL           | PPL               | combined      |               25 |               1.000 |                            0.665 |           0.683 |          0.069 |                    0.682 |                   0.071 |           0.675 |          0.073 |          0.531 |          0.829 |
| PPL+XPL           | XPL               | combined      |               25 |               1.000 |                            0.665 |           0.659 |          0.085 |                    0.661 |                   0.086 |           0.650 |          0.088 |          0.413 |          0.830 |
| XPL               | PPL               | cross_domain  |               25 |              10.000 |                            0.688 |           0.530 |          0.067 |                    0.531 |                   0.066 |           0.509 |          0.078 |          0.325 |          0.618 |
| XPL               | XPL               | within_domain |               25 |              10.000 |                            0.688 |           0.699 |          0.087 |                    0.701 |                   0.086 |           0.691 |          0.091 |          0.527 |          0.864 |

> Validação aninhada com 5 dobras e 5 repetições. As linhas do teste oficial foram excluídas. O desvio-padrão é descritivo porque as dobras repetidas se sobrepõem.

## Diagnóstico de agrupamento

|   samples |   clusters |   adjusted_rand_class |   nmi_class |   adjusted_rand_mode |   nmi_mode |   silhouette_clusters |
|----------:|-----------:|----------------------:|------------:|---------------------:|-----------:|----------------------:|
|   336.000 |      4.000 |                 0.129 |       0.186 |                0.004 |      0.011 |                 0.059 |

## Alinhamento de protótipos

|   samples |   embedding_dimensions |   pca_explained_variance_2d |   same_class_mean_cosine |   different_class_mean_cosine |   prototype_alignment_gap |
|----------:|-----------------------:|----------------------------:|-------------------------:|------------------------------:|--------------------------:|
|   336.000 |                384.000 |                       0.171 |                    0.953 |                         0.874 |                     0.079 |

> Resultados exploratórios em um subconjunto pequeno. O teste foi mantido fora da seleção de hiperparâmetros. PPL e XPL são domínios não pareados.
