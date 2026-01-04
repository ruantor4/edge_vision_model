# Edge Vision Model – Modelagem e Análise de Algoritmos de Detecção de Objetos

Aplicação desenvolvida em **[Python 3.11](https://docs.python.org/pt-br/3.11/contents.html)** para realizar **modelagem, treinamento, avaliação e análise comparativa de algoritmos de detecção de objetos**, como parte de uma **Prova de Conceito (PoC)** em **Visão Computacional**.

O projeto tem como objetivo transformar os insumos produzidos pelo **Edge Vision EDA** em **modelos de detecção treinados e avaliados**, seguindo uma **pipeline profissional, reprodutível e auditável**, permitindo comparar diferentes arquiteturas e **subsidiar a escolha técnica do modelo mais adequado** para uso posterior em sistemas de inferência em **computação de borda (edge computing)**.

Este repositório é **exclusivamente dedicado à etapa de modelagem e análise de algoritmos**, não realizando **Análise Exploratória de Dados (EDA)**, **inferência em tempo real** ou **integração com sistemas finais**, as quais são tratadas em projetos correlatos do ecossistema Edge Vision.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Objetivos do Projeto

- Preparar datasets de detecção de objetos de forma controlada e rastreável.
- Treinar diferentes algoritmos de detecção de objetos.
- Avaliar o desempenho dos modelos em conjunto de teste independente.
- Analisar métricas de qualidade e trade-offs computacionais.
- Comparar arquiteturas de forma justa e reprodutível.
- Selecionar modelos candidatos para uso em ambientes de edge computing.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Estratégia de Modelagem

A estratégia adotada neste projeto segue práticas profissionais de engenharia de Machine Learning:

- **Separação clara entre EDA, modelagem e inferência**
- **Uso de dataset externo imutável**, previamente analisado no projeto Edge Vision EDA
- **Preparação controlada dos dados**, específica para cada algoritmo
- **Treinamento isolado por arquitetura**
- **Avaliação em conjunto de teste nunca utilizado durante o treinamento**
- **Comparação objetiva baseada em métricas padronizadas**

Cada algoritmo é treinado e avaliado sob **condições equivalentes**, garantindo uma comparação técnica justa.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Algoritmos Avaliados

Os seguintes algoritmos de detecção de objetos são utilizados e comparados:

- **YOLO (You Only Look Once)**  
  Arquitetura de estágio único, otimizada para inferência rápida, especialmente adequada para aplicações em edge computing.

- **SSD (Single Shot Detector)**  
  Detector de estágio único com foco em simplicidade e desempenho intermediário entre velocidade e precisão.

- **Faster R-CNN**  
  Arquitetura de dois estágios, com maior custo computacional, utilizada como referência de precisão.

Cada modelo é treinado com configurações explícitas e avaliado utilizando o mesmo conjunto de métricas.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Funcionalidades

| Categoria | Descrição |
|-----------|-----------|
| **Preparação de Dados** | Conversão e organização do dataset conforme requisitos de cada algoritmo. |
| **Treinamento de Modelos** | Treinamento controlado de YOLO, SSD e Faster R-CNN. |
| **Avaliação de Desempenho** | Cálculo de métricas em conjunto de teste isolado. |
| **Análise Comparativa** | Comparação objetiva de desempenho entre algoritmos. |
| **Versionamento de Artefatos** | Registro de dados preparados, pesos treinados e métricas por execução. |
| **Persistência de Resultados** | Salvamento estruturado de métricas e logs. |
| **Pipeline Reprodutível** | Execução determinística baseada em configurações versionadas. |

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Métricas de Avaliação

A avaliação dos modelos é realizada utilizando métricas amplamente aceitas na literatura e na indústria:

- **Precision**
- **Recall**
- **mAP@0.5**
- **mAP@0.5:0.95**
- **Tempo de inferência**

As métricas são definidas **antes do treinamento** e aplicadas de forma consistente a todos os modelos.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Tecnologias Utilizadas

| Categoria | Tecnologia |
|----------|------------|
| **Linguagem** | **[Python 3.11](https://docs.python.org/pt-br/3.11/contents.html)** |
| **Deep Learning** | **[YOLO](https://yolo-docs.readthedocs.io/en/latest/)**, **[SSD](https://docs.pytorch.org/vision/main/models/ssd.html)**, **[Faster R-CNN](https://docs.pytorch.org/vision/main/models/faster_rcnn.html)** |
| **Análise Numérica** | **[NumPy](https://numpy.org/doc/)** |
| **Avaliação** | **[pycocotools](https://github.com/cocodataset/cocoapi)**, **[métricas COCO](https://cocodataset.org/)** |
| **Visualização** | **[Matplotlib](https://matplotlib.org/)** |
| **Logging** | **[logging](https://docs.python.org/pt-br/3/library/logging.html)** |
| **Sistema** | **[pathlib](https://docs.python.org/3/library/pathlib.html)**, **[yaml](https://pyyaml.org/wiki/PyYAMLDocumentation)**, **[json](https://docs.python.org/3/library/json.html)** |

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Estrutura de Diretórios

```bash
edge-vision-model/
├── artifacts/                          # Artefatos gerados ao longo da pipeline
│   ├── prepared_data/                  # Datasets preparados por algoritmo (YOLO / SSD / Faster R-CNN)
│   ├── models/                         # Pesos e checkpoints dos modelos treinados
│   │   ├── yolo/                       # Artefatos de treino do YOLO
│   │   │   ├── best_model.pt           # Melhor peso do YOLO (validação)
│   │   │   ├── final_model.pt          # Peso final do YOLO
│   │   │   └── training_metadata.txt   # Metadados do treinamento
│   │   ├── ssd/                        # Artefatos de treino do SSD
│   │   │   ├── best_model.pth          # Melhor peso do SSD
│   │   │   ├── final_model.pth         # Peso final do SSD
│   │   │   └── training_metadata.txt   # Metadados do treinamento
│   │   └── faster_rcnn/                # Artefatos de treino do Faster R-CNN
│   │       ├── best_model.pth          # Melhor peso do Faster R-CNN
│   │       ├── final_model.pth         # Peso final do Faster R-CNN
│   │       └── training_metadata.txt   # Metadados do treinamento
│   │
│   ├── metrics/                        # Métricas geradas pelos evaluators (CSV COCO-style)
│   │   ├── yolo_test.csv               # Métricas finais do YOLO
│   │   ├── ssd_test.csv                # Métricas finais do SSD
│   │   └── faster_rcnn_test.csv        # Métricas finais do Faster R-CNN
│   │
│   └── comparisons/                    # Resultados comparativos consolidados
│       ├── models_comparison.csv       # Comparação de métricas (AP / AR)
│       ├── models_comparison.json      # Comparação de métricas (formato técnico)
│       ├── models_cost_comparison.csv  # Comparação de custo computacional
│       ├── models_cost_comparison.json # Comparação de custo computacional (técnico)
│       ├── models_final_comparison.csv # Consolidação final (métricas + custo)
│       └── models_final_comparison.json
│
├── comparator/                         # Módulos de comparação entre modelos
│   ├── __init__.py
│   ├── model_comparator_metrics.py     # Comparação de métricas de avaliação (AP / AR)
│   ├── model_comparator_cost.py        # Comparação de custo computacional (inferência)
│   └── model_comparator_merge.py       # Consolidação final dos resultados
│
├── config/                             # Configurações centrais do projeto
│   ├── __init__.py
│   ├── settings.py                     # Configuração global (paths, flags, parâmetros)
│   ├── dataset.yaml                    # Referência ao dataset externo (RAW)
│   ├── metrics.yaml                    # Definição formal das métricas de avaliação
│   └── models/                         # Configurações específicas por modelo
│       ├── yolo.yaml                   # Configuração do YOLO
│       ├── ssd.yaml                    # Configuração do SSD
│       └── faster_rcnn.yaml            # Configuração do Faster R-CNN
│
├── core/                               # Núcleo da pipeline de dados e validações
│   ├── __init__.py
│   ├── config_validator.py             # Validação estrutural das configurações
│   ├── dataset_preparer.py             # Preparação determinística do dataset
│   └── evaluation_runner.py            # Orquestra execução dos evaluators
│
├── evaluators/                         # Avaliação individual dos modelos
│   ├── __init__.py
│   ├── base_coco_evaluator.py          # Classe base para avaliação COCO-style
│   ├── yolo_evaluator.py               # Avaliação do YOLO
│   ├── ssd_evaluator.py                # Avaliação do SSD
│   └── faster_rcnn_evaluator.py        # Avaliação do Faster R-CNN
│
├── trainers/                           # Treinamento dos modelos
│   ├── __init__.py
│   ├── yolo_trainer.py                 # Treinamento do YOLO
│   ├── ssd_trainer.py                  # Treinamento do SSD
│   └── faster_rcnn_trainer.py          # Treinamento do Faster R-CNN
│
├── utils/                              # Utilitários compartilhados
│   ├── __init__.py
│   └── logging_global.py               # Configuração global de logging
│
├── logs/                               # Logs de execução da pipeline
│   ├── edge_vision_model_YYYY-MM-DD.log
│
├── tools/                              # Scripts auxiliares e experimentais
│
├── main.py                             # Orquestrador principal da pipeline
├── run_evaluate.py                     # Execução direta da etapa de avaliação
├── run_model.py                        # Execução isolada de modelos (debug / teste)
├── main_teste.py                       # Script auxiliar de testes
│
├── requirements.txt                    # Dependências do projeto
└── README.md                           # Documentação técnica do projeto
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Dataset

O dataset utilizado neste projeto é externo ao repositório e corresponde ao dataset previamente analisado e validado no projeto Edge Vision EDA.

O acesso ao dataset é realizado exclusivamente em modo leitura, por meio de configuração definida em config/dataset.yaml.
Nenhuma modificação é realizada sobre o dataset original durante a execução deste projeto.

Os dados preparados para treinamento são gerados e versionados no diretório artifacts/prepared_data, garantindo rastreabilidade entre dados, modelos e métricas.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## Execução

### Passo 1 – Criar ambiente virtual
```bash
$ python -m venv .venv
$ source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
```

### Passo 2 – Instalar dependências
```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

### Passo 3 – Executar o EDA
```bash
$ python main.py
```
A execução realiza, de forma controlada, as etapas de preparação de dados, treinamento, avaliação e comparação dos modelos, conforme definido em configuração.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Observações Técnicas

- O projeto é focado exclusivamente em modelagem e análise de algoritmos.
- Não há qualquer etapa de EDA ou inferência em tempo real neste repositório.
- O conjunto de teste é mantido isolado e nunca utilizado durante o treinamento.
- Os modelos selecionados são utilizados em projetos correlatos do ecossistema Edge Vision.
- O `main.py` atua apenas como orquestrador da pipeline.s